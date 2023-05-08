# coding: UTF-8
"""
Calculate Root-mean-square deviation (RMSD) between structure A and B.
""" 

import sys
import argparse
import numpy as np

from input_output import Structure

__author__ = "Hongsheng Cai, Guoyuan Liu, Guangfu Luo"
__copyright__ = "..."
__version__ = "1.2"
__maintainer__ = "Hongsheng Cai"
__email__ = "11611318@mail.sustech.edu.cn"
__date__ = "Sep 6, 2022"

def pbc_diff(fcoords1, fcoords2):
    """
    Returns the 'fractional distance' between two coordinates taking into
    account periodic boundary conditions.

    Args:
        fcoords1: First set of fractional coordinates. e.g., [0.5, 0.6,
            0.7] or [[1.1, 1.2, 4.3], [0.5, 0.6, 0.7]]. It can be a single
            coord or any array of coords.
        fcoords2: Second set of fractional coordinates.

    Returns:
        Fractional distance. Each coordinate must have the property that
        abs(a) <= 0.5. Examples:
        pbc_diff([0.1, 0.1, 0.1], [0.3, 0.5, 0.9]) = [-0.2, -0.4, 0.2]
        pbc_diff([0.9, 0.1, 1.01], [0.3, 0.5, 0.9]) = [-0.4, -0.4, 0.11]
    """
    fdist = np.subtract(fcoords1, fcoords2)
    return fdist - np.round(fdist)

def rmsd_pbc(file_path_1, file_path_2, original_def=True):
    """
    Calculate absolute root-mean-square diffence between two structures.
    No rotation nor recentering will be considered. Periodic boundary condition
    will be considered.
    """
    try:
        a = Structure.from_file(filename=file_path_1)
        b = Structure.from_file(filename=file_path_2)
    except Exception:
        sys.exit("File import failed.")

    # check if two structures are valid for compare
    natoms = check_validity(a, b)

    # get fractional coords of each structure
    # a_frac = [a[i].frac_coords for i in range(natoms)]
    # b_frac = [b[i].frac_coords for i in range(natoms)]
    a_frac = a.frac_coords
    b_frac = b.frac_coords

    # get frac_diff considering pbc (abs(diff) <= 0.5)
    frac_diff = pbc_diff(a_frac, b_frac)

    # convert to cartesian coords difference
    cart_diff = a.lattice.get_cartesian_coords(frac_diff)

    if original_def:
    # original definition of RMSD
        return np.sqrt(np.sum(cart_diff ** 2))
    else:
        # revised definition. The top 5 deviation is considered
        # aiming to better compare the difference of two similar structures
        return np.sum(np.sort(np.abs(cart_diff))[:5]) / 5


def check_validity(structure_a, structure_b):
    if len(structure_a) == len(structure_b):
        return len(structure_a)
    else:
        sys.exit("Invalid structures to compare.")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(
     description="Calculate root-mean-square-difference between two structures.")
    argparse.add_argument("structure_1", nargs=1)
    argparse.add_argument("structure_2", nargs=1)
    args = argparse.parse_args()

    file_a = args.structure_1[0]
    file_b = args.structure_2[0]
    print(rmsd_pbc(file_a, file_b))
