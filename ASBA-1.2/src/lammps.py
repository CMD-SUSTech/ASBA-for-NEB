# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

"""
This module implements a core class LammpsData for generating/parsing
LAMMPS data file, and other bridging classes to build LammpsData from
molecules. This module also implements a subclass CombinedData for
merging LammpsData object.

Only point particle styles are supported for now (atom_style in angle,
atomic, bond, charge, full and molecular only). See the pages below for
more info.

    http://lammps.sandia.gov/doc/atom_style.html
    http://lammps.sandia.gov/doc/read_data.html

"""

import numpy as np
from pathlib import Path
from monty.json import MSONable

from lattice import Lattice 

__author__ = "Kiran Mathew, Zhi Deng, Tingzheng Hou"
__copyright__ = "Copyright 2018, The Materials Virtual Lab"
__version__ = "1.0"
__maintainer__ = "Zhi Deng"
__email__ = "z4deng@eng.ucsd.edu"
__date__ = "Aug 1, 2018"

MODULE_DIR = Path(__file__).resolve().parent

SECTION_KEYWORDS = {"atom": ["Atoms", "Velocities", "Masses",
                             "Ellipsoids", "Lines", "Triangles", "Bodies"],
                    "topology": ["Bonds", "Angles", "Dihedrals", "Impropers"],
                    "ff": ["Pair Coeffs", "PairIJ Coeffs", "Bond Coeffs",
                           "Angle Coeffs", "Dihedral Coeffs",
                           "Improper Coeffs"],
                    "class2": ["BondBond Coeffs", "BondAngle Coeffs",
                               "MiddleBondTorsion Coeffs",
                               "EndBondTorsion Coeffs", "AngleTorsion Coeffs",
                               "AngleAngleTorsion Coeffs",
                               "BondBond13 Coeffs", "AngleAngle Coeffs"]}

CLASS2_KEYWORDS = {"Angle Coeffs": ["BondBond Coeffs", "BondAngle Coeffs"],
                   "Dihedral Coeffs": ["MiddleBondTorsion Coeffs",
                                       "EndBondTorsion Coeffs",
                                       "AngleTorsion Coeffs",
                                       "AngleAngleTorsion Coeffs",
                                       "BondBond13 Coeffs"],
                   "Improper Coeffs": ["AngleAngle Coeffs"]}

SECTION_HEADERS = {"Masses": ["mass"],
                   "Velocities": ["vx", "vy", "vz"],
                   "Bonds": ["type", "atom1", "atom2"],
                   "Angles": ["type", "atom1", "atom2", "atom3"],
                   "Dihedrals": ["type", "atom1", "atom2", "atom3", "atom4"],
                   "Impropers": ["type", "atom1", "atom2", "atom3", "atom4"]}

ATOMS_HEADERS = {"angle": ["molecule-ID", "type", "x", "y", "z"],
                 "atomic": ["type", "x", "y", "z"],
                 "bond": ["molecule-ID", "type", "x", "y", "z"],
                 "charge": ["type", "q", "x", "y", "z"],
                 "full": ["molecule-ID", "type", "q", "x", "y", "z"],
                 "molecular": ["molecule-ID", "type", "x", "y", "z"]}


def clean_lines(string_list, remove_empty_lines=True):
    """
    Strips whitespace, carriage returns and empty lines from a list of strings.

    Args:
        string_list: List of strings
        remove_empty_lines: Set to True to skip lines which are empty after
            stripping.

    Returns:
        List of clean strings with no whitespaces.
    """

    for s in string_list:
        clean_s = s
        if '#' in s:
            ind = s.index('#')
            clean_s = s[:ind]
        clean_s = clean_s.strip()
        if (not remove_empty_lines) or clean_s != '':
            yield clean_s

class LammpsBox(MSONable):
    """
    Object for representing a simulation box in LAMMPS settings.
    """

    def __init__(self, bounds, tilt=None):
        """

        Args:
            bounds: A (3, 2) array/list of floats setting the
                boundaries of simulation box.
            tilt: A (3,) array/list of floats setting the tilt of
                simulation box. Default to None, i.e., use an
                orthogonal box.

        """
        bounds_arr = np.array(bounds)
        assert bounds_arr.shape == (3, 2), \
            "Expecting a (3, 2) array for bounds," \
            " got {}".format(bounds_arr.shape)
        self.bounds = bounds_arr.tolist()
        matrix = np.diag(bounds_arr[:, 1] - bounds_arr[:, 0])

        self.tilt = None
        if tilt is not None:
            tilt_arr = np.array(tilt)
            assert tilt_arr.shape == (3,), \
                "Expecting a (3,) array for box_tilt," \
                " got {}".format(tilt_arr.shape)
            self.tilt = tilt_arr.tolist()
            matrix[1, 0] = tilt_arr[0]
            matrix[2, 0] = tilt_arr[1]
            matrix[2, 1] = tilt_arr[2]
        self._matrix = matrix

    def __str__(self):
        return self.get_string()

    def __repr__(self):
        return self.get_string()

    @property
    def volume(self):
        """
        Volume of simulation box.

        """
        m = self._matrix
        return np.dot(np.cross(m[0], m[1]), m[2])

    def get_string(self, significant_figures=6):
        """
        Returns the string representation of simulation box in LAMMPS
        data file format.

        Args:
            significant_figures (int): No. of significant figures to
                output for box settings. Default to 6.

        Returns:
            String representation

        """
        ph = "{:.%df}" % significant_figures
        lines = []
        for bound, d in zip(self.bounds, "xyz"):
            fillers = bound + [d] * 2
            bound_format = " ".join([ph] * 2 + [" {}lo {}hi"])
            lines.append(bound_format.format(*fillers))
        if self.tilt:
            tilt_format = " ".join([ph] * 3 + [" xy xz yz"])
            lines.append(tilt_format.format(*self.tilt))
        return "\n".join(lines)

    def get_box_shift(self, i):
        """
        Calculates the coordinate shift due to PBC.

        Args:
            i: A (n, 3) integer array containing the labels for box
            images of n entries.

        Returns:
            Coorindate shift array with the same shape of i

        """
        return np.inner(i, self._matrix.T)

    def to_lattice(self):
        """
        Converts the simulation box to a more powerful Lattice backend.
        Note that Lattice is always periodic in 3D space while a
        simulation box is not necessarily periodic in all dimensions.

        Returns:
            Lattice

        """
        return Lattice(self._matrix)
