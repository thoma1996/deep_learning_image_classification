import os
from pathlib import Path

import numpy as np
from ase.lattice.hexagonal import Graphene
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from structures import cut_rectangle
from tqdm import tqdm


def simulate_2d_material(atoms, shape, probe_profile, power_law):
    """
    Simulate a STEM image of a 2d material using the convolution approximation.

    Parameters
    ----------
    atoms : ASE Atoms object
        The 2d structure to simulate.
    shape : two ints
        The shape of the output image.
    probe_profile : Callable
        Function for calculating the probe profile.
    power_law : float
        The assumed Z-contrast powerlaw

    Returns
    -------
    ndarray
        Simulated STEM image.
    """

    extent = np.diag(atoms.cell)[:2]
    sampling = extent / shape

    margin = int(np.ceil(probe_profile.x[-1] / min(sampling)))
    shape_w_margin = (shape[0] + 2 * margin, shape[1] + 2 * margin)

    x = np.fft.fftfreq(shape_w_margin[0]) * shape_w_margin[1] * sampling[0]
    y = np.fft.fftfreq(shape_w_margin[1]) * shape_w_margin[1] * sampling[1]

    r = np.sqrt(x[:, None] ** 2 + y[None] ** 2)
    intensity = probe_profile(r)

    positions = atoms.positions[:, :2] / sampling

    inside = ((positions[:, 0] > -margin) &
              (positions[:, 1] > -margin) &
              (positions[:, 0] < shape[0] + margin) &
              (positions[:, 1] < shape[1] + margin))

    positions = positions[inside] + margin - .5

    array = np.zeros(shape_w_margin)
    for number in np.unique(atoms.numbers):
        temp = np.zeros(shape_w_margin)
        superpose_deltas(positions[atoms.numbers == number], temp)
        array += temp * number ** power_law

    array = np.fft.ifft2(np.fft.fft2(array) * np.fft.fft2(intensity)).real
    array = array[margin:-margin, margin:-margin]
    return array


def superpose_deltas(positions: np.ndarray, array: np.ndarray):
    """ Superpose delta functions """
    shape = array.shape[-2:]
    rounded = np.floor(positions).astype(np.int32)
    rows, cols = rounded[:, 0], rounded[:, 1]

    array[rows, cols] += (1 - (positions[:, 0] - rows)) * (1 - (positions[:, 1] - cols))
    array[(rows + 1) % shape[0], cols] += (positions[:, 0] - rows) * (1 - (positions[:, 1] - cols))
    array[rows, (cols + 1) % shape[1]] += (1 - (positions[:, 0] - rows)) * (positions[:, 1] - cols)
    array[(rows + 1) % shape[0], (cols + 1) % shape[1]] += (rows - positions[:, 0]) * (cols - positions[:, 1])


def make_random_hbn_model(extent):
    hbn = Graphene(symbol='N', latticeconstant={'a': 2.502, 'c': 12})
    hbn[0].symbol = 'B'
    rotation = np.random.rand() * 360
    hbn.rotate(rotation, 'z', rotate_cell=True)
    hbn = cut_rectangle(hbn, (0, 0), extent, margin=5)
    return hbn


def add_vacancy(atoms, number, atomic_number=None, margin=0):
    inside = ((atoms.positions[:, 0] > margin) &
              (atoms.positions[:, 1] > margin) &
              (atoms.positions[:, 0] < atoms.cell[0, 0] - margin) &
              (atoms.positions[:, 1] < atoms.cell[1, 1] - margin))

    if atomic_number is not None:
        inside = inside & (atoms.numbers == atomic_number)

    del atoms[np.random.choice(np.where(inside)[0], number, replace=False)]


def make_probe():
    gaussian = lambda x, sigma: np.exp(-x ** 2 / (2 * sigma ** 2))
    lorentz = lambda x, gamma: gamma / (np.pi * (x ** 2 + gamma ** 2))

    x = np.linspace(0, 5, 100)
    profile = gaussian(x, .4) + lorentz(x, 1)
    return interp1d(x, profile, fill_value=0, bounds_error=False)


def add_contamination(image, amount):
    if amount > 0:
        low_frequency_noise = gaussian_filter(np.random.randn(*image.shape), 10)
        low_frequency_noise -= low_frequency_noise.min()
        image += amount * low_frequency_noise


def add_noise(image, amount):
    if amount > 0:
        image[:] = np.random.poisson(image / amount).astype(np.float) * amount


presets = {'set_A':
               {'num_examples': 4000,  # Total number of examples to simulate, the test set will be 10 % of these
                'num_pixels': 48,  # Image size in pixels
                'fov': 15,  # Field of view in Angstrom
                'contamination': 0,  # Scale amount of mobile contaminants, realistic values in 0 to 100
                'noise': 0,  # Scale amount of noise, realistic values in 0 to 2
                'labels': 'basic',  # The labelling scheme, must be 'basic' or 'detailed'
                'margin': 1.5 * 2.502,  # No vacancies within this distance of the image edge (in Angstrom)
                },
           'set_B':
               {'num_examples': 4000,
                'num_pixels': 48,
                'fov': 15,
                'contamination': 0,
                'noise': 0,
                'labels': 'detailed',
                'margin': 1.5 * 2.502,
                }
           }


def simulate_all(preset):
    shape = (preset['num_pixels'],) * 2
    contamination = preset['contamination']
    noise = preset['noise']
    extent = (preset['fov'],) * 2
    folder = os.path.join(os.path.abspath('..'), 'data', preset_key)

    Path(folder).mkdir(parents=True, exist_ok=True)

    for prefix in ('train', 'test'):

        if prefix == 'train':
            N = int(preset['num_examples'] * .9)
        else:
            N = preset['num_examples'] - int(preset['num_examples'] * .9)

        images = np.zeros((N,) + shape, dtype=np.float32)
        labels = np.zeros(N, dtype=np.int)

        for i in tqdm(range(N)):
            num_b_vacancies = np.random.poisson(.4)
            num_n_vacancies = np.random.poisson(.4)

            atoms = make_random_hbn_model(extent)

            add_vacancy(atoms, num_b_vacancies, 5, preset['margin'])
            add_vacancy(atoms, num_n_vacancies, 7, preset['margin'])

            probe = make_probe()
            image = simulate_2d_material(atoms, shape, probe, 1.6)

            add_contamination(image, contamination)
            add_noise(image, noise)

            if (num_b_vacancies + num_n_vacancies) == 0:
                label = 0
            elif (num_b_vacancies == 1) & (num_n_vacancies == 0) & (preset['labels'] == 'detailed'):
                label = 1
            elif (num_b_vacancies == 0) & (num_n_vacancies == 1) & (preset['labels'] == 'detailed'):
                label = 2
            else:
                if (preset['labels'] == 'detailed'):
                    label = 3
                else:
                    label = 1

            images[i] = ((image - image.mean()) / image.std()).astype(np.float32)
            labels[i] = label

    return images, labels


if __name__ == '__main__':

    # Choose a preset here
    preset_key = 'set_A'

    preset = presets[preset_key]

    shape = (preset['num_pixels'],) * 2
    contamination = preset['contamination']
    noise = preset['noise']
    extent = (preset['fov'],) * 2
    folder = os.path.join(os.path.abspath('..'), 'data', preset_key)

    Path(folder).mkdir(parents=True, exist_ok=True)

    for prefix in ('train', 'test'):

        if prefix == 'train':
            N = int(preset['num_examples'] * .9)
        else:
            N = preset['num_examples'] - int(preset['num_examples'] * .9)

        images = np.zeros((N,) + shape, dtype=np.float32)
        labels = np.zeros(N, dtype=np.int)

        for i in tqdm(range(N)):
            num_b_vacancies = np.random.poisson(.4)
            num_n_vacancies = np.random.poisson(.4)

            atoms = make_random_hbn_model(extent)

            add_vacancy(atoms, num_b_vacancies, 5, preset['margin'])
            add_vacancy(atoms, num_n_vacancies, 7, preset['margin'])

            probe = make_probe()
            image = simulate_2d_material(atoms, shape, probe, 1.6)

            add_contamination(image, contamination)
            add_noise(image, noise)

            if (num_b_vacancies + num_n_vacancies) == 0:
                label = 0
            elif (num_b_vacancies == 1) & (num_n_vacancies == 0) & (preset['labels'] == 'detailed'):
                label = 1
            elif (num_b_vacancies == 0) & (num_n_vacancies == 1) & (preset['labels'] == 'detailed'):
                label = 2
            else:
                if (preset['labels'] == 'detailed'):
                    label = 3
                else:
                    label = 1

            images[i] = ((image - image.mean()) / image.std()).astype(np.float32)
            labels[i] = label

        np.save(os.path.join(folder, '_'.join((prefix, 'images.npy'))), images)
        np.save(os.path.join(folder, '_'.join((prefix, 'labels.npy'))), labels)
