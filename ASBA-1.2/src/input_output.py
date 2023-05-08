# coding: utf-8
## Copyright (c) Pymatgen Development Team.
## Distributed under the terms of the MIT License.

"""
Read and output the structure files. See original files:
https://github.com/materialsproject/pymatgen/blob/master/pymatgen/core/structure.py
https://github.com/materialsproject/pymatgen/blob/master/pymatgen/core/sites.py
https://github.com/materialsproject/pymatgen/blob/master/pymatgen/util/io_utils.py
https://github.com/materialsproject/pymatgen/blob/master/pymatgen/io/vasp/inputs.py
"""

import os
import re
import pickle
import itertools
import logging
import collections
from abc import ABCMeta, abstractmethod
import warnings
from fnmatch import fnmatch
from typing import Dict, List, Tuple, Optional, Union, Iterator, Set, Sequence, Iterable
from pathlib import Path
import numpy as np
from numpy.linalg import det

from monty.io import zopen
from monty.json import MSONable

from lattice import Lattice


class SiteCollection(collections.abc.Sequence, metaclass=ABCMeta):
    """
    Basic SiteCollection. Essentially a sequence of Sites or PeriodicSites.
    This serves as a base class for Structure (a collection of PeriodicSites, 
    i.e., periodicity). Not meant to be instantiated directly.
    """
    @property
    def species_and_occu(self) -> List[str]:
        """
        List of species and occupancies at each site of the structure.
        """
        return [site.species for site in self]

    @property
    def num_sites(self) -> int:
        """
        Number of sites.
        """
        return len(self)

    @property
    def cart_coords(self) -> np.ndarray:
        """
        Returns an np.array of the Cartesian coordinates of sites in the
        structure.
        """
        return np.array([site.coords for site in self])

    @property
    def lattice(self) -> Lattice:
        """
        Lattice of the structure.
        """
        return self._lattice

    @property
    def sites(self):
        """
        Returns a tuple of sites.
        """
        return self._sites

    @property
    def frac_coords(self):
        """
        Fractional coordinates as a Nx3 numpy array.
        """
        return np.array([site.frac_coords for site in self._sites])

    def __getitem__(self, ind):
        return self.sites[ind]

    def __len__(self):
        return len(self.sites)


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


class Poscar(MSONable):
    """
    Object for representing the data in a POSCAR or CONTCAR file.
    Please note that this current implementation. Most attributes can be set
    directly.
    .. attribute:: structure
        Associated Structure.
    .. attribute:: comment
        Optional comment string.
    """
    def __init__(
        self,
        structure,
        comment: str = None,
        selective_dynamics=None
    ):
        structure = Structure.from_sites(structure)
        self.structure = structure
        self.comment = "ASBA" if comment is None else comment
    
    @property
    def site_symbols(self):
        """
        Sequence of symbols associated with the Poscar. Similar to 6th line in
        vasp 5+ POSCAR.
        """
        syms = [site.species for site in self.structure]
        return [a[0] for a in itertools.groupby(syms)]

    @property
    def natoms(self):
        """
        Sequence of number of sites of each type associated with the Poscar.
        Similar to 7th line in vasp 5+ POSCAR or the 6th line in vasp 4 POSCAR.
        """
        syms = [site.species for site in self.structure]
        return [len(tuple(a[1])) for a in itertools.groupby(syms)]

    @staticmethod
    def from_string(data):
        """
        Reads a Poscar from a string.
        Args:
            data (str): String containing Poscar data.
        Returns:
            Poscar object.
        """
        # "^\s*$" doesn't match lines with no whitespace
        chunks = re.split(r"\n\s*\n", data.rstrip(), flags=re.MULTILINE)
        try:
            if chunks[0] == "":
                chunks.pop(0)
                chunks[0] = "\n" + chunks[0]
        except IndexError:
            raise ValueError("Empty POSCAR")

        # Parse positions
        lines = tuple(clean_lines(chunks[0].split("\n"), False))
        comment = lines[0]
        scale = float(lines[1])
        lattice = np.array([[float(i) for i in line.split()] for line in lines[2:5]])
        if scale < 0:
            # In vasp, a negative scale factor is treated as a volume. We need
            # to translate this to a proper lattice vector scaling.
            vol = abs(det(lattice))
            lattice *= (-scale / vol) ** (1 / 3)
        else:
            lattice *= scale

        vasp5_symbols = False
        try:
            natoms = [int(i) for i in lines[5].split()]
            ipos = 6
        except ValueError:
            vasp5_symbols = True
            symbols = lines[5].split()
            nlines_symbols = 1
            for nlines_symbols in range(1, 11):
                try:
                    int(lines[5 + nlines_symbols].split()[0])
                    break
                except ValueError:
                    pass
            for iline_symbols in range(6, 5 + nlines_symbols):
                symbols.extend(lines[iline_symbols].split())
            natoms = []
            iline_natoms_start = 5 + nlines_symbols
            for iline_natoms in range(
                iline_natoms_start, iline_natoms_start + nlines_symbols
            ):
                natoms.extend([int(i) for i in lines[iline_natoms].split()])
            atomic_symbols = list()
            for i in range(len(natoms)):
                atomic_symbols.extend([symbols[i]] * natoms[i])
            ipos = 5 + 2 * nlines_symbols

        postype = lines[ipos].split()[0]

        sdynamics = False
        # Selecitive dynamics
        if postype[0] in "sS" or os.path.exists("SD_data.txt"):
            sdynamics = True
            ipos += 1
            postype = lines[ipos].split()[0]

        cart = postype[0] in "cCkK"
        nsites = sum(natoms)

        # read the atomic coordinates
        coords = []
        selective_dynamics = list() if sdynamics else None
        for i in range(nsites):
            toks = lines[ipos + 1 + i].split()
            crd_scale = scale if cart else 1
            coords.append([float(j) * crd_scale for j in toks[:3]])
            if sdynamics:
                if not os.path.exists("SD_data.txt"):
                       selective_dynamics.append([tok.upper()[0] == "T" for tok in toks[3:6]])
        if selective_dynamics:
            with open("SD_data.txt", "wb") as f:
                     pickle.dump(selective_dynamics, f)
        struct = Structure(
            lattice,
            atomic_symbols,
            coords,
            coords_are_cartesian=cart,
        )
        # print("SD:{}".format(selective_dynamics))
        return Poscar(
            struct,
            comment,
        )

    def get_string(self, direct=True, significant_figures=6):
        """
        Returns a string to be written as a POSCAR file. By default, site
        symbols are written, which means compatibility is for vasp >= 5.
        Args:
            direct (bool): Whether coordinates are output in direct or
                Cartesian. Defaults to True.
            significant_figures (int): No. of significant figures to
                output all quantities. Defaults to 6. Note that positions are
                output in fixed point, while velocities are output in
                scientific format.
        Returns:
            String representation of POSCAR.
        """
        latt = self.structure.lattice
        if np.linalg.det(latt.matrix) < 0:
            latt = Lattice(-latt.matrix)

        format_str = "{{:.{0}f}}".format(significant_figures)
        lines = [self.comment, "1.0"]
        for v in latt.matrix:
            lines.append(" ".join([format_str.format(c) for c in v]))

        lines.append(" ".join(self.site_symbols))
        lines.append(" ".join([str(x) for x in self.natoms]))
        # print("SD:{}".format(self.selective_dynamics))
        selective_dynamics = None
        if os.path.exists("SD_data.txt"):
               with open("SD_data.txt", "rb") as f:
                    selective_dynamics = pickle.load(f)
                    lines.append("Selective dynamics")
        lines.append("direct" if direct else "cartesian")

        for (i, site) in enumerate(self.structure):
            coords = site.frac_coords if direct else site.coords
            line = " ".join([format_str.format(c) for c in coords])
            if os.path.exists("SD_data.txt"):
                sd = ["T" if j else "F" for j in selective_dynamics[i]]
                line += " %s %s %s" % (sd[0], sd[1], sd[2])
            lines.append(line)

        return "\n".join(lines) + "\n"

    def __repr__(self):
        return self.get_string()

    def __str__(self):
        return self.get_string()

    def write_file(self, filename, **kwargs):
        """
        Writes POSCAR to a file. The supported kwargs are the same as those for
        the Poscar.get_string method and are passed through directly.
        """
        with zopen(filename, "wt") as f:
            f.write(self.get_string(**kwargs))


class PeriodicSite(MSONable):
    """
    Extension of generic Site object to periodic systems.
    PeriodicSite includes a lattice system.
    """
    def __init__(self,
                 species: Union[str],
                 coords: Union[Tuple, List, np.ndarray],
                 lattice: Lattice,
                 coords_are_cartesian: bool = False,
                 ):
        """
        Create a periodic site.
        :param species: Species on the site.
        :param coords: Cartesian coordinates of site.
        :param lattice: Lattice associated with the site.
        :param coords_are_cartesian: Set to True if you are providing
            Cartesian coordinates. Defaults to False.
        """
        if coords_are_cartesian:
            frac_coords = lattice.get_fractional_coords(coords)
        else:
            frac_coords = coords

        frac_coords = np.array(frac_coords)

        self._lattice = lattice
        self._frac_coords = frac_coords
        self._species = species
        self._coords = None

    @property
    def lattice(self):
        """
        Lattice associated with PeriodicSite
        """
        return self._lattice

    @property  # type: ignore
    def coords(self) -> np.ndarray:  # type: ignore
        """
        Cartesian coordinates
        """
        if self._coords is None:
            self._coords = self._lattice.get_cartesian_coords(self._frac_coords)
        return self._coords

    @property  # type: ignore
    def frac_coords(self) -> np.ndarray:
        """
        Fractional coordinates
        """
        return self._frac_coords

    @property
    def species(self) -> str:
        """
        :return: The species on the site.
        """
        return self._species  # type: ignore


class Structure(SiteCollection, Poscar, PeriodicSite, MSONable):
    """
    Mutable version of structure.
    """
    def __init__(self,
                 lattice: Union[List, np.ndarray, Lattice],
                 species: Sequence[Union[str]],
                 coords: Sequence[Sequence[float]],
                 coords_are_cartesian: bool = False,
                 ):
        """
        Create a periodic structure.
        Args:
            lattice: The lattice, either as a pymatgen.core.lattice.Lattice or
                simply as any 2D array. Each row should correspond to a lattice
                vector. E.g., [[10,0,0], [20,10,0], [0,0,30]] specifies a
                lattice with lattice vectors [10,0,0], [20,10,0] and [0,0,30].
            species: List of species on each site. Can take in flexible input.
            coords (Nx3 array): list of fractional/cartesian coordinates of
                each species.
            coords_are_cartesian (bool): Set to True if you are providing
                coordinates in Cartesian coordinates. Defaults to False.
        """
        if isinstance(lattice, Lattice):
            self._lattice = lattice
        else:
            self._lattice = Lattice(lattice)

        sites = []
        for i, sp in enumerate(species):
            sites.append(
                PeriodicSite(sp, coords[i], self._lattice,
                             coords_are_cartesian=coords_are_cartesian,
                             ))
        # print("SD:{}".format(selective_dynamics))
        self._sites = tuple(sites)

    @classmethod
    def from_sites(cls,
                   sites: List[PeriodicSite],
                   ):
        """
        Convenience constructor to make a Structure from a list of sites.
        Args:
            sites: Sequence of PeriodicSites. Sites must have the same
                lattice.
        Returns:
            (Structure) Note that missing properties are set as None.
        """
        lattice = sites[0].lattice
        for i, site in enumerate(sites):
            if site.lattice != lattice:
                raise ValueError("Sites must belong to the same lattice")
        return cls(lattice, [site.species for site in sites],
                   [site.frac_coords for site in sites],
                   )

    def get_distance(self, i, j, jimage=None):
        """
        Get distance between site i and j assuming periodic boundary
        conditions. If the index jimage of two sites atom j is not specified it
        selects the jimage nearest to the i atom and returns the distance and
        jimage indices in terms of lattice vector translations if the index
        jimage of atom j is specified it returns the distance between the i
        atom and the specified jimage atom.
        Args:
            i (int): Index of first site
            j (int): Index of second site
            jimage: Number of lattice translations in each lattice direction.
                Default is None for nearest image.
        Returns:
            distance
        """
        return self[i].distance(self[j], jimage)


    def interpolate(self, end_structure,
                    nimages: Union[int, Iterable] = 10,
                    interpolate_lattices: bool = False,
                    pbc: bool = True,
                    autosort_tol: float = 0):
        """
        Interpolate between this structure and end_structure. Useful for
        construction of NEB inputs.
        Args:
            end_structure (Structure): structure to interpolate between this
                structure and end.
            nimages (int,list): No. of interpolation images or a list of
                interpolation images. Defaults to 10 images.
            interpolate_lattices (bool): Whether to interpolate the lattices.
                Interpolates the lengths and angles (rather than the matrix)
                so orientation may be affected.
            pbc (bool): Whether to use periodic boundary conditions to find
                the shortest path between endpoints.
            autosort_tol (float): A distance tolerance in angstrom in
                which to automatically sort end_structure to match to the
                closest points in this particular structure. This is usually
                what you want in a NEB calculation. 0 implies no sorting.
                Otherwise, a 0.5 value usually works pretty well.
        Returns:
            List of interpolated structures. The starting and ending
            structures included as the first and last structures respectively.
            A total of (nimages + 1) structures are returned.
        """
        # Check length of structures
        if not (interpolate_lattices or self.lattice == end_structure.lattice):
            raise ValueError("Structures with different lattices!")

        if not isinstance(nimages, collections.abc.Iterable):
            images = np.arange(nimages + 1) / nimages
        else:
            images = nimages

        start_coords = np.array(self.frac_coords)
        end_coords = np.array(end_structure.frac_coords)
        if autosort_tol:
            dist_matrix = self.lattice.get_all_distances(start_coords,
                                                         end_coords)
            site_mappings = collections.defaultdict(list)  # type: Dict[int, List[int]]
            unmapped_start_ind = []
            for i, row in enumerate(dist_matrix):
                ind = np.where(row < autosort_tol)[0]
                if len(ind) == 1:
                    site_mappings[i].append(ind[0])
                else:
                    unmapped_start_ind.append(i)

            sorted_end_coords = np.zeros_like(end_coords)
            matched = []
            for i, j in site_mappings.items():
                sorted_end_coords[i] = end_coords[j[0]]
                matched.append(j[0])

            if len(unmapped_start_ind) == 1:
                i = unmapped_start_ind[0]
                j = list(set(range(len(start_coords))).difference(matched))[0]  # type: ignore
                sorted_end_coords[i] = end_coords[j]

            end_coords = sorted_end_coords

        vec = end_coords - start_coords
        if pbc:
            vec -= np.round(vec)
        sp = self.species_and_occu
        structs = []

        if interpolate_lattices:
            # interpolate lattice matrices using polar decomposition
            from scipy.linalg import polar
            # u is unitary (rotation), p is stretch
            u, p = polar(np.dot(end_structure.lattice.matrix.T,
                                np.linalg.inv(self.lattice.matrix.T)))
            lvec = p - np.identity(3)
            lstart = self.lattice.matrix.T

        for x in images:
            if interpolate_lattices:
                l_a = np.dot(np.identity(3) + x * lvec, lstart).T
                lat = Lattice(l_a)
            else:
                lat = self.lattice
            fcoords = start_coords + x * vec
            structs.append(self.__class__(lat, sp, fcoords))  # type: ignore
        return structs

    def to(self, filename=None, **kwargs):
        """
        Outputs the structure to a file or string.
        Args:
            filename (str): If provided, output will be written to a file. If
                fmt is not specified, the format is determined from the
                filename. Defaults is None, i.e. string output.
            **kwargs: Kwargs passthru to relevant methods. E.g., This allows
                the passing of parameters like symprec to the
                CifWriter.__init__ method for generation of symmetric cifs.
        Returns:
            (str) if filename is None. None otherwise.
        """
        filename = filename or ""
        fname = os.path.basename(filename)

        writer = Poscar(self, **kwargs)
        if filename:
            writer.write_file(filename)
            return None
        return writer.__str__()
    
    @classmethod
    def from_file(cls, filename):
        """
        Reads a structure from a file.
        Args:
            filename (str): The filename to read from.
        Returns:
            Structure.
        """
        filename = str(filename)
        fname = os.path.basename(filename)
        with zopen(filename, "rt") as f:
            contents = f.read()
        s = Poscar.from_string(contents).structure
        sites = cls.from_sites(s)
        return sites
