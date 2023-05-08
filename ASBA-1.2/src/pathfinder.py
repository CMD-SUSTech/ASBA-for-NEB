# coding: utf-8
## Copyright (c) Materials Virtual Lab.
## Distributed under the terms of the BSD License.

"""
Generate the initial guess of MEP structures using the ASBA method. 
This file was revised based on the following python file:
https://github.com/materialsvirtuallab/pymatgen-analysis-diffusion/blob/master/pymatgen/analysis/diffusion/neb/pathfinder.py
"""

import warnings
import os
from typing import List
import itertools
import re
import numpy as np
from tqdm import tqdm
from input_output import Structure, PeriodicSite
from lammps import LammpsBox



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


class Solver:
    """
    A solver using image dependent pair potential (IDPP) algo to get an
    improved initial NEB path. For more details about this algo, please
    refer to Smidstrup et al., J. Chem. Phys. 140, 214106 (2014).

    """
    def __init__(self, structures):
        """
        Initialization.

        Args:
            structures (list of pmg_structure) : Initial guess of the NEB path
                (including initial and final end-point structures).
        """

        latt = structures[0].lattice
        natoms = structures[0].num_sites
        nimages = len(structures) - 2

        # Initial guess of the path (in Cartesian coordinates)
        init_coords = []
        for ni, i in itertools.product(range(nimages + 2), range(natoms)):
            frac_coords = structures[ni][i].frac_coords
            init_coords.append(latt.get_cartesian_coords(frac_coords))
        
        # build element string list
        elements = []
        for i in structures[0]:
            elements.append(
                str(i.species)
            )
        
        self.init_coords = np.array(init_coords).reshape(nimages + 2, natoms, 3)
        self.structures = structures
        self.nimages = nimages
        self.natoms = natoms
        self.elements = elements
        self.lammps_dump_box = None

    @classmethod
    def from_endpoints(cls, endpoints, nimages=5, sort_tol=1.0, switch=True):
        """
        A class method that starts with end-point structures instead. The
        initial guess for the IDPP algo is then constructed using linear
        interpolation.

        Args:
            endpoints (list of Structure objects): The two end-point
                structures.
            nimages (int): Number of images between the two end-points.
            sort_tol (float): Distance tolerance (in Angstrom) used to match
                the atomic indices between start and end structures. Need to
                increase the value in some cases.
            switch (boolean): Whether to consider periodic boundary condition
        """
        try:
            images = endpoints[0].interpolate(
                endpoints[1], nimages=nimages + 1, autosort_tol=sort_tol, pbc=switch
            )
        except Exception as e:
            if "Unable to reliably match structures " in str(e):
                warnings.warn(
                    "Auto sorting is turned off because it is unable"
                    " to match the end-point structures!",
                    UserWarning,
                )
                images = endpoints[0].interpolate(
                    endpoints[1], nimages=nimages + 1, autosort_tol=0
                )
            else:
                raise e

        return Solver(images)

    @staticmethod
    def get_unit_vector(vec):
        return vec / np.sqrt(np.sum(vec ** 2))

    def _get_total_forces(self, x, true_forces, spring_const):
        """
        Calculate the total force on each image structure, which is equal to
        the spring force along the tangent + true force perpendicular to the
        tangent. Note that the spring force is the modified version in the
        literature (e.g. Henkelman et al., J. Chem. Phys. 113, 9901 (2000)).

        x [niamges, natoms, 3]: cartesian coords of the whole path.
        """
        total_forces = []
        natoms = np.shape(true_forces)[1]

        for ni in range(1, len(x) - 1):
            # add tolerance
            vec1 = (x[ni + 1] - x[ni]).flatten()
            vec2 = (x[ni] - x[ni - 1]).flatten()

            # Local tangent
            tangent = self.get_unit_vector(vec1) + self.get_unit_vector(vec2)
            tangent = self.get_unit_vector(tangent)

            # Spring force
            spring_force = (
                spring_const * (np.linalg.norm(vec1) - np.linalg.norm(vec2)) * tangent
            )
                        
            # Total force
            flat_ft = true_forces[ni - 1].copy().flatten()
            total_force = true_forces[ni - 1] + (
                spring_force - np.dot(flat_ft, tangent) * tangent
            ).reshape(natoms, 3)
            total_forces.append(total_force)

        return np.array(total_forces)

    def _Nonlinear_core_atoms(self, threshold):
        """
        Compare initial and final states of the given path and determine if an atom has
        large enough displacement.

        Args
        initial: fractional coords of initial state
        final: fractional coords of final state
        threshold: numerical threshold in cartesian coords
        """
        NLc = []
        initial = self.structures[0].frac_coords
        final = self.structures[-1].frac_coords
        frac_diff = pbc_diff(initial, final)
        disp = np.linalg.norm(
            self.structures[0].lattice.get_cartesian_coords(frac_diff), axis=1
        )
        for i in range(self.natoms):
            if disp[i] > threshold:
                NLc.append(i)
        print("\nAtoms with displacements >= 0.3 Angstrom between the initial and final states")
        for n in NLc:
            print("{}\t{}\t{}".format(n, self.elements[n], float('%.3f'%disp[n])))
        return NLc

    def ASBA_NEB(
        self,
        path,
        maxiter,
        dump_dir,
        ini_structure,
        fin_structure,
        bond_type,
        bond_type_with_pi_bond,
        pi_bond,
        given_atom_radius,
        dump_asba=True,
        con_co=1.4,
        dump_total=True,
        moving_atoms=None,
        step_size=0.05,
        max_disp=0.05,
        max_iter=200,
        ftol=1e-4,
        step_update_method=None,
        base_step=0.01,
        max_step=0.05,
        spring_const=5.0,
        k_att=5.0,
        k_rep=0.15,
        NLc=[],
        NLc_threshold=2.0,
        **kwargs,
    ):
        """
        Conduct ASBA process on given path with NEB.

        Args:
        path [ni+2, na, 3]: initial fractional coords path for ASBA
        maxiter (int): maximum iteration path (list of Structures): initial path for
        ASBA. The first and last structures correspond to the initial and final
        states.
        moving_atoms (list of int): index of atoms that are allowed to move
        during NEB. If None, then all atoms are allowed to move.
        """

        # generate atom radius
        self.bond_length = self._build_bond_length_list(ini_structure, fin_structure, bond_type,bond_type_with_pi_bond, given_atom_radius, pi_bond)
        
        # Construct cartesian coords for path
        # minimization is updated on path_coords not images

        # convert to cart_coords for all images
        latt = self.structures[0].lattice
        initial_images = path[1:-1]
        image_coords = []
        for ni, i in itertools.product(range(self.nimages), range(self.natoms)):
            # no getter for cart_coords attribute, so has to do this
            frac_coord_temp = initial_images[ni][i].frac_coords
            image_coords.append(latt.get_cartesian_coords(frac_coord_temp))
        image_coords = np.array(image_coords).reshape(self.nimages, self.natoms, 3)
        # add initial and final state to path_coords
        path_coords = [self.init_coords[0], *image_coords, self.init_coords[-1]]
        path_coords = np.array(path_coords)

        # if moving_atoms is [], then all atoms are allowed to move
        if moving_atoms is None:
            moving_atoms = list(range(len(self.structures[0])))

        # find nonlinear core atoms
        NLc += self._Nonlinear_core_atoms(NLc_threshold)
        if not NLc:
            NLc = range(self.natoms)
            warnings.warn("No core nonlinear atoms detected. All atoms are considered nonlinear core atoms.")

        # calculate force, update images and output structures with force
        max_forces = [float("inf")]
        initial_step = step_size
        # print("force tolerance:{}".format(ftol))
        total_forces_list = []
        for n in tqdm(range(maxiter)):
            if k_att == 0 and k_rep == 0:
                dump_asba, dump_total = False, False
                true_forces = np.zeros([self.nimages, self.natoms, 3])
            else:
                # for each iteration ASBA force is evaluated on latest image coords
                ASBA_forces = self._get_ASBA_forces_and_energy(
                    image_coords=path_coords[1:-1],
                    NLc=NLc,
                    k_att=k_att,
                    k_rep=k_rep,
                    **kwargs,
                )
                true_forces = ASBA_forces

            # path_coords includes initial and final states
            total_forces = self._get_total_forces(
                path_coords, true_forces, spring_const
            )
            total_forces_list.append(total_forces)
            # output dump file for each imaegs
            if dump_asba:
                self.dump_writer("dump_asba", dump_dir, path_coords, ASBA_forces, n)
            if dump_total:
                self.dump_writer("dump_total", dump_dir, path_coords, total_forces, n)

            # calculate displacement. disp_mat[ni][nn][3]
            disp_mat = step_size * total_forces[:, moving_atoms, :]
            # disp_mat = step_size * total_forces_list[n]
            disp_mat = np.where(
                np.abs(disp_mat) > max_disp, np.sign(disp_mat) * max_disp, disp_mat
            )
            path_coords[1:-1, moving_atoms] += disp_mat

            # calculate max force and store
            max_forces.append(np.abs(total_forces[:, moving_atoms, :]).max())
            # stop criteria
            if max_forces[-1] < ftol:
                print("\n>>> Total force converges!")
                break

            # change step size for better optimization
            if step_update_method == "decay" and max_forces[-1] < max_forces[-2]:
                step_size = initial_step * (1 / (1 + 0.01 * n))
    
        else:
            print("\n>>> ASBA-NEB: Maximum iteration number is reached without convergence!")
            print(">>> Max force at end of optimization: {}".format(max_forces[-1]))

        # apply PBC to all atoms of all images
        ASBA_path = [self.structures[0]]
        for ni in range(self.nimages):
            new_sites = []
            for site, cart_coords in zip(self.structures[ni + 1], path_coords[ni + 1]):
                new_site = PeriodicSite(
                    site.species,
                    coords=cart_coords,
                    lattice=site.lattice,
                    coords_are_cartesian=True,
                )
                new_sites.append(new_site)

            ASBA_path.append(Structure.from_sites(new_sites))
        # Also include end-point structure.
        ASBA_path.append(self.structures[-1])

        return ASBA_path

    def lammps_dump_str(self, coords, force_matrix, step):
        """
        Output cartesian coordinates and corresponding total forces in
        lammps dump format.

        Args:
        coords [nn, 3]: Cartesian coords of each atoms during optimization
        force_matrix [nn, 3]: force to output
        step: time step starts with 0
        Return:
        str: dump string representation for each time step.
        """
        # concatenation by join a list is significantly faster than += string
        dump = []
        # append header
        dump.append("ITEM: TIMESTEP\n")
        dump.append("{}\n".format(step))
        dump.append("ITEM: NUMBER OF ATOMS\n")
        dump.append("{}\n".format(self.natoms))

        # append box bounds in lammps dump format
        if self.lammps_dump_box is None:
            lmpbox = self.lattice_2_dump_lmpbox(self.structures[0].lattice)
            self.lammps_dump_box = self.get_box_str(lmpbox)
            dump.append(self.lammps_dump_box)
        else:
            dump.append(self.lammps_dump_box)

        # append coords and forces
        dump.append("ITEM: ATOMS id type x y z fx fy fz\n")
        for i in range(self.natoms):
            dump.append("{} {} ".format(i, self.elements[i]))
            for j in range(3):
                dump.append("{:.6f} ".format(coords[i][j]))
            for j in range(3):
                dump.append("{:.6f} ".format(force_matrix[i][j]))
            dump.append("\n")
        return "".join(dump)

    def lattice_2_dump_lmpbox(self, lattice, origin=(0, 0, 0)):
        """
        Converts a lattice object to LammpsBox,Adapted from
        pytmatgen.core.io.lammps.data.lattice_2_lmpbox. The original lmpbox is by lammps
        data format which is different from dump format in bounds definition. Note that
        this method will result in wrong lattice matrix so cannnot converted back into
        pymatgen lattice.

        Args:
            lattice (Lattice): Input lattice.
            origin: A (3,) array/list of floats setting lower bounds of
                simulation box. Default to (0, 0, 0).

        Returns:
            LammpsBox

        """
        a, b, c = lattice.abc
        xlo, ylo, zlo = origin
        xhi = a + xlo
        m = lattice.matrix
        xy = np.dot(m[1], m[0] / a)
        yhi = np.sqrt(b ** 2 - xy ** 2) + ylo
        xz = np.dot(m[2], m[0] / a)
        yz = (np.dot(m[1], m[2]) - xy * xz) / (yhi - ylo)
        zhi = np.sqrt(c ** 2 - xz ** 2 - yz ** 2) + zlo
        tilt = None if lattice.is_orthogonal else [xy, xz, yz]
        xlo_bound = xlo + min(0.0, xy, xz, xy + xz)
        xhi_bound = xhi + max(0.0, xy, xz, xy + xz)
        ylo_bound = ylo + min(0.0, yz)
        yhi_bound = yhi + max(0.0, yz)
        bounds = [[xlo_bound, xhi_bound], [ylo_bound, yhi_bound], [zlo, zhi]]
        return LammpsBox(bounds, tilt)

    def get_box_str(self, lmpbox: LammpsBox):
        is_orthogonal = lmpbox.tilt is None
        m = lmpbox.bounds.copy()
        out_str = []
        out_str.append(
            "ITEM: BOX BOUNDS pp pp pp\n"
            if is_orthogonal
            else "ITEM: BOX BOUNDS xy xz yz pp pp pp\n"
        )
        for i in range(len(m)):
            for j in m[i]:
                out_str.append("{:.6f} ".format(j))
            out_str.append("\n" if is_orthogonal else "{:.6f}\n".format(lmpbox.tilt[i]))
        return "".join(out_str)

    def dump_writer(self, filename, dump_dir, coords, forces, n):
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        for i in range(self.nimages):
            with open(
                os.path.join(dump_dir, "{}_{:02d}".format(filename, i + 1)), "a"
            ) as f:
                f.write(self.lammps_dump_str(coords[i + 1], forces[i], n))
        return True

    @staticmethod
    def _get_triangular_step(
        iteration: int, half_period=5, base_step=0.05, max_step=0.1
    ):
        """
        Given the inputs, calculates the step_size that should be applicable
        for this iteration using CLR technique from Anand Saha
        (http://teleported.in/posts/cyclic-learning-rate/).
        """
        cycle = np.floor(1 + iteration / (2 * half_period))
        x = np.abs(iteration / half_period - 2 * cycle + 1)
        step = base_step + (max_step - base_step) * np.maximum(0, (1 - x))
        return step

    @staticmethod
    def _decay(iteration: int, initial_step_size, decay):
        step_size = initial_step_size * 1 / (1 + decay * iteration)
        return step_size

    def _get_ASBA_forces_and_energy(
        self,
        image_coords,
        NLc,
        att_tol=0.2,
        k_rep=5.0,
        k_att=0.05,
        rep_tol=0.1,
        **kwargs
    ):
        """
        calculate forces and energies

        Args:
        image_coords ([ni,na,3]): current cart coords of each images
        NLc (list): list of indices of manually selected core nonlinear atoms
        att_tol (float): tolerance for attractive force 
        k_rep (float): spring constant for repulsive force
        k_att (float): spring constant for attractive force
        rep_tol (float): tolerance for repulsive force

        Returns:
            Clash_forces[ni, nn, 3]
        """
        # get lattice
        lattice = self.structures[0].lattice

        
        # get frac_coords of current image_coords
        frac_image_coords = []
        for ni in range(self.nimages):
            frac_image_coords.append(lattice.get_fractional_coords(image_coords[ni]))
        
        # calculate attractive forces
        # find bonded atom pairs 
        atom_pairs = self._find_bonded_atom_pairs(    
            frac_image_coords, image_coords, NLc, att_tol, rep_tol, **kwargs
        )
            
        attractive_forces = np.zeros((self.nimages, self.natoms, 3), dtype=np.float64)
        for case in atom_pairs[2]:
            ni, i, j, d = case
            coord_bonded = image_coords[ni][i]
            coord_pulling = image_coords[ni][j]

            # get direction (towards pulling atoms) considering PBC
            direction = self.get_direction_pbc(coord_pulling, coord_bonded)

            # get displacement and calculate force
            delta_d = d - self._get_attractive_length(i, j, att_tol)
            f = 1 * ( d ** -4 ) * k_att * ( delta_d ** 2 ) * direction

            # apply force on atoms
            # attractive_forces have no counter forces unless they are both in atom set
            attractive_forces[ni][i] += f
            attractive_forces[ni][j] += -f

        for case in atom_pairs[3]:
            ni, j, z, d = case
            coord_bonded = image_coords[ni][j]
            coord_pulling = image_coords[ni][z]

            # get direction (towards pulling atoms) considering PBC
            direction = self.get_direction_pbc(coord_pulling, coord_bonded)

            # get displacement and calculate force
            delta_d = d - self._get_attractive_length(j, z, att_tol)
            f = 1 * ( d ** -4 ) * k_att * ( delta_d ** 2 ) * direction

            # apply force on atoms
            # attractive_forces have no counter forces unless they are both in atom set
            attractive_forces[ni][j] += f

        # calculate repulsive forces
        repulsive_forces = np.zeros((self.nimages, self.natoms, 3), dtype=np.float64)
        for case in atom_pairs[0]:
            ni, i, j, d = case
            coord1 = image_coords[ni][i]
            coord2 = image_coords[ni][j]
            # direction pointing towards atom i
            direction = self.get_direction_pbc(coord1, coord2)
            delta_d = abs(d - self._get_repulsive_length(i, j, rep_tol))
            f = 1 * ( d ** -4 ) * k_rep * ( delta_d ** 2 ) * direction
            # force and counter force
            repulsive_forces[ni][i] += f
            repulsive_forces[ni][j] += -f
        
        for case in atom_pairs[1]:
            ni, j, z, d = case
            coord1 = image_coords[ni][j]
            coord2 = image_coords[ni][z]
            # direction pointing towards atom j
            direction = self.get_direction_pbc(coord1, coord2)
            delta_d = abs(d - self._get_repulsive_length(j, z, rep_tol))
            f = 1 * ( d ** -4 ) * k_rep * ( delta_d ** 2 ) * direction
            # force
            repulsive_forces[ni][j] += f

        ASBA_forces = repulsive_forces + attractive_forces
        return ASBA_forces


    def _find_bonded_atom_pairs(
        self,
        frac_coords,
        image_coords,
        NLc: List,
        att_tol=0.2,
        rep_tol=0,
    ):
        """
        Alls atoms are assumed to be connected. For any atoms, if its closest neighbor
        is further than the corresponding attractive_length, then all its neighbors will
        pull the atom. The neighbor atom pairs are searched within a radius using
        pymatgen.core.lattice.get_points_in_sphere() which will try to use cthyon code.

        Args:
            frac_coords ([ni,na,3]): fractional coords of current optimizing
                structure.
            NLc (list): list of indices of manually selected core nonlinear atoms

        Return:
            cases (2D list): [image number, atom index, BondedNeighbor 1,
             BondedNeighbor 2...]
        """
        lattice = self.structures[0].lattice 
        
        # add moving sites to nonlinear core atoms to allow manual selection
        NLc = set(NLc)
        
        # get frac_coords of current image_coords
        frac_image_coords = []
        for ni in range(self.nimages):
            frac_image_coords.append(lattice.get_fractional_coords(image_coords[ni]))
        
        # find nonlinear first neighboring atoms around nonlinear core atoms
        for ni in range(self.nimages):
            d = lattice.get_all_distances(frac_image_coords[ni], frac_image_coords[ni])
            NL1s = set()
            NL1 = []
            for n in NLc:
                for j in range(self.natoms):
                    if (
                        d[n][j] < self._get_interaction_radius(n,j)
                        and n != j
                    ):
                        NL1.append(j)
                        NL1s = NL1s | set(NL1)
                        NL1s = NL1s ^ NLc
        # for nonlinear core atoms and nonlinear first neighboring atoms, check if their neighbors need bonding
        atoms_set = NLc | NL1s
        atoms_set_list = list(atoms_set)

        rep_atom_pairs1 = []
        rep_atom_pairs2 = []
        att_atom_pairs1 = []
        att_atom_pairs2 = [] 
        
        for ni in range(self.nimages):
            d = lattice.get_all_distances(frac_image_coords[ni], frac_image_coords[ni])
            for i in atoms_set_list:
                for j in range(self.natoms): 
                    if ( d[i][j] < self._get_repulsive_length(i, j, rep_tol) and i != j ):
                        if [ni, j, i, d[j][i]] not in rep_atom_pairs1:
                            rep_atom_pairs1.append([ni, i, j, d[i][j]])
                            if j not in atoms_set_list:
                            # j is nonlinear second neighboring atom
                                for z in range(self.natoms):
                                    if ( d[j][z] < self._get_repulsive_length(j, z, rep_tol) and j != z ):
                                        if z not in atoms_set_list:
                                        # z is linear atom
                                            rep_atom_pairs2.append([ni, j, z, d[j][z]])
                                    elif ( d[j][z] < self._get_interaction_radius(j, z) and d[j][z] > self._get_attractive_length(j, z, att_tol) ):
                                        if z not in atoms_set_list:
                                        # z is linear atom
                                            att_atom_pairs2.append([ni, j, z, d[j][z]])
                    elif ( d[i][j] < self._get_interaction_radius(i, j) and d[i][j] > self._get_attractive_length(i, j, att_tol) ):
                        if [ni, j, i, d[j][i]] not in att_atom_pairs1:
                            att_atom_pairs1.append([ni, i, j, d[i][j]])
                            if j not in atoms_set_list:
                            # j is nonlinear second neighboring atom
                                for z in range(self.natoms):
                                    if ( d[j][z] < self._get_repulsive_length(j, z, rep_tol) and j != z ):
                                        if z not in atoms_set_list:
                                        # z is linear atom
                                            rep_atom_pairs2.append([ni, j, z, d[j][z]])
                                    elif ( d[j][z] < self._get_interaction_radius(j, z) and d[j][z] > self._get_attractive_length(j, z, att_tol) ):
                                        if z not in atoms_set_list:
                                        # z is linear atom
                                            att_atom_pairs2.append([ni, j, z, d[j][z]])   
        return rep_atom_pairs1, rep_atom_pairs2, att_atom_pairs1, att_atom_pairs2

    def get_direction_pbc(self, coords1, coords2):
        """
        return a unit vector pointing towards coord1
        coords1 and coords2: cartesian coordinates of a single atom
        """
        latt = self.structures[0].lattice
        frac_coords1 = latt.get_fractional_coords(coords1)
        frac_coords2 = latt.get_fractional_coords(coords2)
        coord_diff = pbc_diff(frac_coords1, frac_coords2)
        return self.get_unit_vector(latt.get_cartesian_coords(coord_diff))

    def _get_repulsive_length(self, atom_1, atom_2, rep_tol):
        element_1 = self.elements[atom_1]
        element_2 = self.elements[atom_2]
        if (element_1, element_2) in self.bond_length:
            d = (1 - rep_tol) * self.bond_length[(element_1, element_2)]
        else:
            d = (1 - rep_tol) * self.bond_length[(element_2, element_1)]
        return d

    def _get_attractive_length(self, atom_1, atom_2, att_tol):
        element_1 = self.elements[atom_1]
        element_2 = self.elements[atom_2]
        if (element_1, element_2) in self.bond_length:
            d = (1 + att_tol) * self.bond_length[(element_1, element_2)]
        else:
            d = (1 + att_tol) * self.bond_length[(element_2, element_1)]
        return d

    def _get_interaction_radius(self, atom_1, atom_2):
        element_1 = self.elements[atom_1]
        element_2 = self.elements[atom_2]
        if (element_1, element_2) in self.bond_length:
            d = (1 + 0.25) * self.bond_length[(element_1, element_2)]
        else:
            d = (1 + 0.25) * self.bond_length[(element_2, element_1)]
        return d

    def read_poscar(self, file_name):
        """
        param: file_name
        return: Cartesian coordinates
        lattice: scale, float
        basis: [x1, y1, z1], [x2, y2, z2], [x3, y3, z3]], float
        elements: [element1, element2, ...], str
        num_atoms: [number_of_element1, number_of_element2, ...], int
        coordinates: [[x1, y1, z1], [x2, y2, z2], ...], float
        """
        space = re.compile(r'\s+')
        with open(file_name) as input_file:
            content = input_file.readlines()

        basis = []
        for i in range(2, 5):
            line = space.split(content[i].strip())
            basis.append([float(line[0]), float(line[1]), float(line[2])])

        elements = space.split(content[5].strip())
        num_atoms = list(map(int, space.split(content[6].strip())))

        coordinates = []
        if "Selective dynamics" in content[7]:
            start = 9
        else:
            start = 8
        end = start + sum(num_atoms)
        for i in range(start, end):
            line = space.split(content[i].strip())
            coordinates.append([float(line[0]), float(line[1]), float(line[2])])


        return coordinates, elements, num_atoms

    def get_atoms_dict(self, elements, num_atoms, num_1=0):
        atoms_dict = {}
        i = 0
        while i <= len(elements) - 1:
            if i == 0:
                num = num_atoms[i]
                num_list = list(range(num))
                atoms_dict[elements[i]] = num_list
                i += 1
            else:
                num_1 += num_atoms[i - 1]
                num_2 = num_1 + num_atoms[i]
                num_list = list(range(num_1, num_2))
                atoms_dict[elements[i]] = num_list
                i += 1

        return atoms_dict

    def get_atoms_combination(self, bond_type):
        atoms_combination = []
        bonds = bond_type.split(',')
        for bond in bonds:
            i = bond.split('-')
            atoms_combination.append(i)

        return atoms_combination

    def get_average_bond_length(self, atoms_combination, atoms_dicts, coordinates_list_ini, coordinates_list_fin):
        bond_length_dict = {}
        for bond_type in atoms_combination:
            for a, b in itertools.combinations(bond_type, 2):
                if a == b:
                    length_list = []
                    for i, j in itertools.combinations(atoms_dicts[a], 2):
                        bond_length_ini = pbc_diff(coordinates_list_ini[i], coordinates_list_ini[j])
                        length_ini = np.linalg.norm(self.structures[0].lattice.get_cartesian_coords(bond_length_ini))
                        bond_length_fin = pbc_diff(coordinates_list_fin[i], coordinates_list_fin[j])
                        length_fin = np.linalg.norm(self.structures[0].lattice.get_cartesian_coords(bond_length_fin))
                        length_list.append(length_ini)
                        length_list.append(length_fin)
                        length_list.sort()
                    bond_length_min = length_list[0]
                    length_list_new = length_list[:]
                    for length in length_list:
                        if length > bond_length_min * 1.15 or length > bond_length_min + 0.3:
                            length_list_new.remove(length)
                    length_list = length_list_new
                    bond_length_dict[(a, b)] = np.mean(length_list)

                else:
                    length_list = []
                    for i in atoms_dicts[a]:
                        for j in atoms_dicts[b]:
                            bond_length_ini = pbc_diff(coordinates_list_ini[i], coordinates_list_ini[j])
                            length_ini = np.linalg.norm(self.structures[0].lattice.get_cartesian_coords(bond_length_ini))
                            bond_length_fin = pbc_diff(coordinates_list_fin[i], coordinates_list_fin[j])
                            length_fin = np.linalg.norm(self.structures[0].lattice.get_cartesian_coords(bond_length_fin))
                            length_list.append(length_ini)
                            length_list.append(length_fin)
                            length_list.sort()
                    bond_length_min = length_list[0]
                    length_list_new = length_list[:]
                    for length in length_list:
                        if length > bond_length_min * 1.15 or length > bond_length_min + 0.3:
                            length_list_new.remove(length)
                    length_list = length_list_new
                    bond_length_dict[(a, b)] = np.mean(length_list)

        return bond_length_dict

    def get_given_atom_radius(self, given_atom_radius):
        given_atoms_radius = {}
        atoms = []
        radius = []
        atom_radius = given_atom_radius.split(',')
        for r in atom_radius:
            i = r.split(':')
            j = 0
            while j < len(i):
                atoms.append(i[j])
                radius.append(i[j+1])
                j += 2
        for z in range(len(atoms)):
            given_atoms_radius[atoms[z]] = radius[z]

        return given_atoms_radius

    def get_atoms_radius(self, bond_length_dict, given_atoms_radius):
        element_list1 = []
        element_list2 = []
        if given_atoms_radius:
            for key in given_atoms_radius.keys():
                bond_length_dict[(key, key)] = 2 * float(given_atoms_radius.get(key))
        print("average bond length from initial and final states:")
        for key in bond_length_dict.keys():
            print("{}:{}".format(key, float('%.3f'%bond_length_dict[key])))
        for key in bond_length_dict.keys():
            element_list1.append(list(key))
            element_list2.extend(list(key))
            elements_list = list(set(element_list2))
        row_num = len(element_list1)
        column_num = len(elements_list)
        svd_array = np.zeros([row_num, column_num])
        for i in range(row_num):
            for j in range(column_num):
                if elements_list[j] in element_list1[i]:
                    svd_array[i][j] = element_list1[i].count(elements_list[j])
        bond_length_array = np.array(list(bond_length_dict.values()))
        u, sigma, vt = np.linalg.svd(svd_array)
        v = np.transpose(vt)
        w = np.zeros([row_num, column_num])
        for i in range(len(sigma)):
            w[i][i] = sigma[i]
        w_pinv = np.linalg.pinv(w)
        ut = np.transpose(u)
        atoms_radius = np.linalg.multi_dot([v, w_pinv, ut, bond_length_array])
        atoms_radius_dict = {}
        for i in elements_list:
            atoms_radius_dict[i] = atoms_radius[elements_list.index(i)]

        return atoms_radius_dict

    def _build_bond_length_list(self, ini_structure, fin_structure, bond_type, bond_type_with_pi_bond, given_atom_radius, pi_bond):
        """
        Build bond length list in the initial and final states.
        """
        data_ini = self.read_poscar(ini_structure)
        coords_ini = data_ini[0]
        elements = data_ini[-2]
        number_atoms = data_ini[-1]
        coord_array_ini = np.array(coords_ini)
        coordinates_list_ini = coord_array_ini.reshape(sum(number_atoms), 3)
        atoms_dicts = self.get_atoms_dict(elements, number_atoms)
        data_fin = self.read_poscar(fin_structure)
        coords_fin = data_fin[0]
        coord_array_fin = np.array(coords_fin)
        coordinates_list_fin = coord_array_fin.reshape(sum(number_atoms), 3)

        atoms_combination = self.get_atoms_combination(bond_type)
        average_bond_length = self.get_average_bond_length(atoms_combination, atoms_dicts, coordinates_list_ini,
                                                      coordinates_list_fin)
        if given_atom_radius == 'None':
            given_atoms_radius = None
        else:    
            given_atoms_radius = self.get_given_atom_radius(given_atom_radius)
        radii_table = self.get_atoms_radius(average_bond_length, given_atoms_radius)
        bond_length_list = {}

        radii_dict = {}
        for e in elements:
            if e in radii_table:
                radii_dict[e] = radii_table[e]
            else:
                radii_dict[e] = float(given_atoms_radius[e])
        print("")
        for k in radii_dict:
            print("{} radius: {}".format(k, float('%.3f'%radii_dict[k])))

        if bond_type_with_pi_bond != 'None':
            atoms_combination_pi_bond = self.get_atoms_combination(bond_type_with_pi_bond)
            for a, b in atoms_combination_pi_bond:
                print("bond type with pi bond: {}-{}, pi bond: {}".format(a, b, pi_bond))
                bond_length_list[(a, b)] = radii_dict[a] + radii_dict[b] + pi_bond
        for i, j in itertools.combinations_with_replacement(elements, 2):
            if (i, j) not in bond_length_list and (j, i) not in bond_length_list:
                bond_length_list[(i, j)] = radii_dict[i] + radii_dict[j]

        return bond_length_list
