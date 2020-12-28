from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle


def is_cell_valid(atoms, tol=1e-12) -> bool:
    """
    Check whether the cell of an ASE atoms object can be converted to a structure that is usable by abTEM.

    Parameters
    ----------
    atoms : ASE atoms object
        The atoms that should be checked.
    tol : float
        Components of the lattice vectors below this value are considered to be zero.

    Returns
    -------
    bool
        If true, the atomic structure is usable by abTEM.
    """
    if np.abs(atoms.cell[0, 0] - np.linalg.norm(atoms.cell[0])) > tol:
        return False

    if np.abs(atoms.cell[1, 2]) > tol:
        return False

    if np.abs(atoms.cell[2, 2] - np.linalg.norm(atoms.cell[2])) > tol:
        return False

    return True


def check_is_cell_valid(atoms, tol=1e-12):
    if not is_cell_valid(atoms, tol=tol):
        raise RuntimeError('This cell cannot be made orthogonal using currently implemented methods.')


def standardize_cell(atoms, tol=1e-12):
    """
    Standardize the cell of an ASE atoms object. The atoms are rotated so one of the lattice vectors in the xy-plane
    aligns with the x-axis, then all of the lattice vectors are made positive.

    Parameters
    ----------
    atoms : ASE atoms object
        The atoms that should be standardized
    tol : float
        Components of the lattice vectors below this value are considered to be zero.

    Returns
    -------
    atoms : ASE atoms object
        The standardized atoms.
    """
    cell = np.array(atoms.cell)

    vertical_vector = np.where(np.all(np.abs(cell[:, :2]) < tol, axis=1))[0]

    if len(vertical_vector) != 1:
        raise RuntimeError('Invalid cell: no vertical lattice vector')

    cell[[vertical_vector[0], 2]] = cell[[2, vertical_vector[0]]]
    atoms.set_cell(cell)

    r = np.arctan2(atoms.cell[0, 1], atoms.cell[0, 0]) / np.pi * 180

    if r != 0.:
        atoms.rotate(-r, 'z', rotate_cell=True)

    check_is_cell_valid(atoms, tol)

    atoms.set_cell(np.abs(atoms.get_cell()))

    atoms.wrap()
    return atoms


def cut_rectangle(atoms, origin, extent, margin=0.):
    """
    Cuts out a cell starting at the origin to a given extent from a sufficiently repeated copy of atoms.

    Parameters
    ----------
    atoms : ASE atoms object
        This should correspond to a repeatable unit cell.
    origin : two float
        Origin of the new cell. Units of Angstrom.
    extent : two float
        xy-extent of the new cell. Units of Angstrom.
    margin : float
        Atoms within margin from the border of the new cell will be included. Units of Angstrom. Default is 0.

    Returns
    -------
    ASE atoms object
    """
    atoms = atoms.copy()
    cell = atoms.cell.copy()

    extent = (extent[0], extent[1], atoms.cell[2, 2],)
    atoms.positions[:, :2] -= np.array(origin)

    a = atoms.cell.scaled_positions(np.array((extent[0] + 2 * margin, 0, 0)))
    b = atoms.cell.scaled_positions(np.array((0, extent[1] + 2 * margin, 0)))
    cell_with_margin = np.dot(np.array([a[:2], b[:2]]), atoms.cell[:2, :2])

    scaled_corners_newcell = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    corners = np.dot(scaled_corners_newcell, cell_with_margin)
    scaled_corners = np.linalg.solve(atoms.cell[:2, :2].T, corners.T).T
    repetitions = np.ceil(scaled_corners.ptp(axis=0)).astype(np.int) + 1

    atoms = atoms.repeat(tuple(repetitions) + (1,))
    atoms.positions[:, :2] += np.dot(np.floor(scaled_corners.min(axis=0)), cell[:2, :2])

    margin_cell_translation = np.ceil(cell.scaled_positions(np.array([margin, margin, 0]))[:2])
    atoms.positions[:, :2] -= margin_cell_translation[0] * cell[0, :2]
    atoms.positions[:, :2] -= margin_cell_translation[1] * cell[1, :2]

    atoms.set_cell([extent[0], extent[1], cell[2, 2]])

    atoms = atoms[((atoms.positions[:, 0] >= -margin) &
                   (atoms.positions[:, 1] >= -margin) &
                   (atoms.positions[:, 0] < extent[0] + margin) &
                   (atoms.positions[:, 1] < extent[1] + margin))
    ]
    return atoms


def _plane2axes(plane):
    """Internal function for extracting axes from a plane."""
    axes = ()
    last_axis = [0, 1, 2]
    for axis in list(plane):
        if axis == 'x':
            axes += (0,)
            last_axis.remove(0)
        if axis == 'y':
            axes += (1,)
            last_axis.remove(1)
        if axis == 'z':
            axes += (2,)
            last_axis.remove(2)
    return axes + (last_axis[0],)


_cube = np.array([[[0, 0, 0], [0, 0, 1]],
                  [[0, 0, 0], [0, 1, 0]],
                  [[0, 0, 0], [1, 0, 0]],
                  [[0, 0, 1], [0, 1, 1]],
                  [[0, 0, 1], [1, 0, 1]],
                  [[0, 1, 0], [1, 1, 0]],
                  [[0, 1, 0], [0, 1, 1]],
                  [[1, 0, 0], [1, 1, 0]],
                  [[1, 0, 0], [1, 0, 1]],
                  [[0, 1, 1], [1, 1, 1]],
                  [[1, 0, 1], [1, 1, 1]],
                  [[1, 1, 0], [1, 1, 1]]])


def show_atoms(atoms, repeat=(1, 1), scans=None, plane='xy', ax=None, scale_atoms=.5, title=None, numbering=False):
    """
    Show atoms function

    Function to display atoms, especially in Jupyter notebooks.

    Parameters
    ----------
    atoms : ASE atoms object
        The atoms to be shown.
    repeat : two ints, optional
        Tiling of the image. Default is (1,1), ie. no tiling.
    scans : ndarray, optional
        List of scans to apply. Default is None.
    plane : str
        The projection plane.
    ax : axes object
        pyplot axes object.
    scale_atoms : float
        Scaling factor for the atom display sizes. Default is 0.5.
    title : str
        Title of the displayed image. Default is None.
    numbering : bool
        Option to set plot numbering. Default is False.
    """

    if ax is None:
        fig, ax = plt.subplots()

    axes = _plane2axes(plane)

    atoms = atoms.copy()
    cell = atoms.cell
    atoms *= repeat + (1,)

    for line in _cube:
        cell_lines = np.array([np.dot(line[0], cell), np.dot(line[1], cell)])
        ax.plot(cell_lines[:, axes[0]], cell_lines[:, axes[1]], 'k-')

    if len(atoms) > 0:
        positions = atoms.positions[:, axes[:2]]
        order = np.argsort(atoms.positions[:, axes[2]])
        positions = positions[order]

        colors = jmol_colors[atoms.numbers[order]]

        sizes = covalent_radii[atoms.numbers[order]] * scale_atoms

        circles = []
        for position, size in zip(positions, sizes):
            circles.append(Circle(position, size))

        coll = PatchCollection(circles, facecolors=colors, edgecolors='black')
        ax.add_collection(coll)

        ax.axis('equal')
        ax.set_xlabel(plane[0] + ' [Å]')
        ax.set_ylabel(plane[1] + ' [Å]')

        ax.set_title(title)

        if numbering:
            for i, (position, size) in enumerate(zip(positions, sizes)):
                ax.annotate('{}'.format(order[i]), xy=position, ha="center", va="center")

    if scans is not None:
        if not isinstance(scans, Iterable):
            scans = [scans]

        for scan in scans:
            scan.add_to_mpl_plot(ax)

    return ax
