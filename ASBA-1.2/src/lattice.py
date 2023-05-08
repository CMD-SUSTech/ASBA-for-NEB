# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

"""
Define the classes relating to lattice. See original file:
https://github.com/materialsproject/pymatgen/blob/master/pymatgen/core/lattice.py
"""

from typing import List, Union, Tuple, Optional

import numpy as np
from numpy.linalg import inv
from numpy import pi, dot

from monty.json import MSONable

import coord_cython as cuc

__author__ = "Shyue Ping Ong, Michael Kocher"
__copyright__ = "Copyright 2011, The Materials Project"
__maintainer__ = "Shyue Ping Ong"
__email__ = "shyuep@gmail.com"


def pbc_shortest_vectors(lattice, fcoords1, fcoords2, mask=None,
                         return_d2=False):
    """
    Returns the shortest vectors between two lists of coordinates taking into
    account periodic boundary conditions and the lattice.

    Args:
        lattice: lattice to use
        fcoords1: First set of fractional coordinates. e.g., [0.5, 0.6, 0.7]
            or [[1.1, 1.2, 4.3], [0.5, 0.6, 0.7]]. It can be a single
            coord or any array of coords.
        fcoords2: Second set of fractional coordinates.
        mask (boolean array): Mask of matches that are not allowed.
            i.e. if mask[1,2] == True, then subset[1] cannot be matched
            to superset[2]
        return_d2 (boolean): whether to also return the squared distances

    Returns:
        array of displacement vectors from fcoords1 to fcoords2
        first index is fcoords1 index, second is fcoords2 index
    """
    return cuc.pbc_shortest_vectors(lattice, fcoords1, fcoords2, mask,
                                    return_d2)

Vector3Like = Union[List[float], np.ndarray]

class Lattice(MSONable):
    """
    A lattice object.  Essentially a matrix with conversion matrices. In
    general, it is assumed that length units are in Angstroms and angles are in
    degrees unless otherwise stated.
    """
    
    # Properties lazily generated for efficiency.

    def __init__(self, matrix: Union[List[float], List[List[float]], np.ndarray]):
        """
        Create a lattice from any sequence of 9 numbers. Note that the sequence
        is assumed to be read one row at a time. Each row represents one
        lattice vector.

        Args:
            matrix: Sequence of numbers in any form. Examples of acceptable
                input.
                i) An actual numpy array.
                ii) [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                iii) [1, 0, 0 , 0, 1, 0, 0, 0, 1]
                iv) (1, 0, 0, 0, 1, 0, 0, 0, 1)
                Each row should correspond to a lattice vector.
                E.g., [[10, 0, 0], [20, 10, 0], [0, 0, 30]] specifies a lattice
                with lattice vectors [10, 0, 0], [20, 10, 0] and [0, 0, 30].
        """
        m = np.array(matrix, dtype=np.float64).reshape((3, 3))
        m.setflags(write=False)
        self._matrix = m  # type: np.ndarray
        self._inv_matrix = None  # type: Optional[np.ndarray]
        self._diags = None
        self._lll_matrix_mappings = {}  # type: Dict[float, np.ndarray]
        self._lll_inverse = None

    @property
    def lengths(self) -> Tuple[float, float, float]:
        """
        :return: The lengths (a, b, c) of the lattice.
        """
        return tuple(np.sqrt(np.sum(self._matrix ** 2, axis=1)).tolist())  # type: ignore

    @property
    def angles(self) -> Tuple[float, float, float]:
        """
        Returns the angles (alpha, beta, gamma) of the lattice.
        """
        m = self._matrix
        lengths = self.lengths
        angles = np.zeros(3)
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            angles[i] = abs_cap(dot(m[j], m[k]) / (lengths[j] * lengths[k]))
        angles = np.arccos(angles) * 180.0 / pi
        return tuple(angles.tolist())  # type: ignore

    @property
    def is_orthogonal(self) -> bool:
        """
        :return: Whether all angles are 90 degrees.
        """
        return all([abs(a - 90) < 1e-5 for a in self.angles])

    def __format__(self, fmt_spec=""):
        """
        Support format printing. Supported formats are:

        1. "l" for a list format that can be easily copied and pasted, e.g.,
           ".3fl" prints something like
           "[[10.000, 0.000, 0.000], [0.000, 10.000, 0.000], [0.000, 0.000, 10.000]]"
        2. "p" for lattice parameters ".1fp" prints something like
           "{10.0, 10.0, 10.0, 90.0, 90.0, 90.0}"
        3. Default will simply print a 3x3 matrix form. E.g.,
           10.000 0.000 0.000
           0.000 10.000 0.000
           0.000 0.000 10.000
        """
        m = self._matrix.tolist()
        if fmt_spec.endswith("l"):
            fmt = "[[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]]"
            fmt_spec = fmt_spec[:-1]
        elif fmt_spec.endswith("p"):
            fmt = "{{{}, {}, {}, {}, {}, {}}}"
            fmt_spec = fmt_spec[:-1]
            m = (self.lengths, self.angles)
        else:
            fmt = "{} {} {}\n{} {} {}\n{} {} {}"
        return fmt.format(*[format(c, fmt_spec) for row in m for c in row])

    @property
    def matrix(self) -> np.ndarray:
        """Copy of matrix representing the Lattice"""
        return self._matrix

    @property
    def inv_matrix(self) -> np.ndarray:
        """
        Inverse of lattice matrix.
        """
        if self._inv_matrix is None:
            self._inv_matrix = inv(self._matrix)
            self._inv_matrix.setflags(write=False)
        return self._inv_matrix


    def get_cartesian_coords(self, fractional_coords: Vector3Like) -> np.ndarray:
        """
        Returns the cartesian coordinates given fractional coordinates.

        Args:
            fractional_coords (3x1 array): Fractional coords.

        Returns:
            Cartesian coordinates
        """
        return dot(fractional_coords, self._matrix)

    def get_fractional_coords(self, cart_coords: Vector3Like) -> np.ndarray:
        """
        Returns the fractional coordinates given cartesian coordinates.

        Args:
            cart_coords (3x1 array): Cartesian coords.

        Returns:
            Fractional coordinates.
        """
        return dot(cart_coords, self.inv_matrix)


    @property
    def a(self) -> float:
        """
        *a* lattice parameter.
        """
        return self.lengths[0]

    @property
    def b(self) -> float:
        """
        *b* lattice parameter.
        """
        return self.lengths[1]

    @property
    def c(self) -> float:
        """
        *c* lattice parameter.
        """
        return self.lengths[2]

    @property
    def abc(self) -> Tuple[float, float, float]:
        """
        Lengths of the lattice vectors, i.e. (a, b, c)
        """
        return self.lengths


    @property
    def lll_matrix(self) -> np.ndarray:
        """
        :return: The matrix for LLL reduction
        """
        if 0.75 not in self._lll_matrix_mappings:
            self._lll_matrix_mappings[0.75] = self._calculate_lll()
        return self._lll_matrix_mappings[0.75][0]

    @property
    def lll_mapping(self) -> np.ndarray:
        """
        :return: The mapping between the LLL reduced lattice and the original
            lattice.
        """
        if 0.75 not in self._lll_matrix_mappings:
            self._lll_matrix_mappings[0.75] = self._calculate_lll()
        return self._lll_matrix_mappings[0.75][1]

    @property
    def lll_inverse(self) -> np.ndarray:
        """
        :return: Inverse of self.lll_mapping.
        """
        return np.linalg.inv(self.lll_mapping)


    def __eq__(self, other):
        """
        A lattice is considered to be equal to another if the internal matrix
        representation satisfies np.allclose(matrix1, matrix2) to be True.
        """
        if other is None:
            return False
        # shortcut the np.allclose if the memory addresses are the same
        # (very common in Structure.from_sites)
        return self is other or np.allclose(self.matrix, other.matrix)



    def _calculate_lll(self, delta: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs a Lenstra-Lenstra-Lovasz lattice basis reduction to obtain a
        c-reduced basis. This method returns a basis which is as "good" as
        possible, with "good" defined by orthongonality of the lattice vectors.

        This basis is used for all the periodic boundary condition calculations.

        Args:
            delta (float): Reduction parameter. Default of 0.75 is usually
                fine.

        Returns:
            Reduced lattice matrix, mapping to get to that lattice.
        """
        # Transpose the lattice matrix first so that basis vectors are columns.
        # Makes life easier.
        a = self._matrix.copy().T

        b = np.zeros((3, 3))  # Vectors after the Gram-Schmidt process
        u = np.zeros((3, 3))  # Gram-Schmidt coeffieicnts
        m = np.zeros(3)  # These are the norm squared of each vec.

        b[:, 0] = a[:, 0]
        m[0] = dot(b[:, 0], b[:, 0])
        for i in range(1, 3):
            u[i, 0:i] = dot(a[:, i].T, b[:, 0:i]) / m[0:i]
            b[:, i] = a[:, i] - dot(b[:, 0:i], u[i, 0:i].T)
            m[i] = dot(b[:, i], b[:, i])

        k = 2

        mapping = np.identity(3, dtype=np.double)
        while k <= 3:
            # Size reduction.
            for i in range(k - 1, 0, -1):
                q = round(u[k - 1, i - 1])
                if q != 0:
                    # Reduce the k-th basis vector.
                    a[:, k - 1] = a[:, k - 1] - q * a[:, i - 1]
                    mapping[:, k - 1] = mapping[:, k - 1] - q * mapping[:, i - 1]
                    uu = list(u[i - 1, 0: (i - 1)])
                    uu.append(1)
                    # Update the GS coefficients.
                    u[k - 1, 0:i] = u[k - 1, 0:i] - q * np.array(uu)

            # Check the Lovasz condition.
            if dot(b[:, k - 1], b[:, k - 1]) >= (
                    delta - abs(u[k - 1, k - 2]) ** 2
            ) * dot(b[:, (k - 2)], b[:, (k - 2)]):
                # Increment k if the Lovasz condition holds.
                k += 1
            else:
                # If the Lovasz condition fails,
                # swap the k-th and (k-1)-th basis vector
                v = a[:, k - 1].copy()
                a[:, k - 1] = a[:, k - 2].copy()
                a[:, k - 2] = v

                v_m = mapping[:, k - 1].copy()
                mapping[:, k - 1] = mapping[:, k - 2].copy()
                mapping[:, k - 2] = v_m

                # Update the Gram-Schmidt coefficients
                for s in range(k - 1, k + 1):
                    u[s - 1, 0: (s - 1)] = (
                        dot(a[:, s - 1].T, b[:, 0: (s - 1)]) / m[0: (s - 1)]
                    )
                    b[:, s - 1] = a[:, s - 1] - dot(
                        b[:, 0: (s - 1)], u[s - 1, 0: (s - 1)].T
                    )
                    m[s - 1] = dot(b[:, s - 1], b[:, s - 1])

                if k > 2:
                    k -= 1
                else:
                    # We have to do p/q, so do lstsq(q.T, p.T).T instead.
                    p = dot(a[:, k:3].T, b[:, (k - 2): k])
                    q = np.diag(m[(k - 2): k])
                    result = np.linalg.lstsq(q.T, p.T, rcond=None)[0].T  # type: ignore
                    u[k:3, (k - 2): k] = result

        return a.T, mapping.T

    def get_lll_frac_coords(self, frac_coords: Vector3Like) -> np.ndarray:
        """
        Given fractional coordinates in the lattice basis, returns corresponding
        fractional coordinates in the lll basis.
        """
        return dot(frac_coords, self.lll_inverse)



    def get_all_distances(
            self,
            fcoords1: Union[Vector3Like, List[Vector3Like]],
            fcoords2: Union[Vector3Like, List[Vector3Like]],
    ) -> np.ndarray:
        """
        Returns the distances between two lists of coordinates taking into
        account periodic boundary conditions and the lattice. Note that this
        computes an MxN array of distances (i.e. the distance between each
        point in fcoords1 and every coordinate in fcoords2). This is
        different functionality from pbc_diff.

        Args:
            fcoords1: First set of fractional coordinates. e.g., [0.5, 0.6,
                0.7] or [[1.1, 1.2, 4.3], [0.5, 0.6, 0.7]]. It can be a single
                coord or any array of coords.
            fcoords2: Second set of fractional coordinates.

        Returns:
            2d array of cartesian distances. E.g the distance between
            fcoords1[i] and fcoords2[j] is distances[i,j]
        """
        v, d2 = pbc_shortest_vectors(self, fcoords1, fcoords2, return_d2=True)
        return np.sqrt(d2)


    def get_distance_and_image(
            self,
            frac_coords1: Vector3Like,
            frac_coords2: Vector3Like,
            jimage: Optional[Union[List[int], np.ndarray]] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Gets distance between two frac_coords assuming periodic boundary
        conditions. If the index jimage is not specified it selects the j
        image nearest to the i atom and returns the distance and jimage
        indices in terms of lattice vector translations. If the index jimage
        is specified it returns the distance between the frac_coords1 and
        the specified jimage of frac_coords2, and the given jimage is also
        returned.

        Args:
            frac_coords1 (3x1 array): Reference fcoords to get distance from.
            frac)coords2 (3x1 array): fcoords to get distance from.
            jimage (3x1 array): Specific periodic image in terms of
                lattice translations, e.g., [1,0,0] implies to take periodic
                image that is one a-lattice vector away. If jimage is None,
                the image that is nearest to the site is found.

        Returns:
            (distance, jimage): distance and periodic lattice translations
            of the other site for which the distance applies. This means that
            the distance between frac_coords1 and (jimage + frac_coords2) is
            equal to distance.
        """
        if jimage is None:
            v, d2 = pbc_shortest_vectors(
                self, frac_coords1, frac_coords2, return_d2=True
            )
            fc = self.get_fractional_coords(v[0][0]) + frac_coords1 - frac_coords2
            fc = np.array(np.round(fc), dtype=np.int)
            return np.sqrt(d2[0, 0]), fc

        jimage = np.array(jimage)
        mapped_vec = self.get_cartesian_coords(jimage + frac_coords2 - frac_coords1)
        return np.linalg.norm(mapped_vec), jimage


def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.

    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.

    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
