# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

import warnings
import os
from typing import List
import itertools
import re
import numpy as np
from tqdm import tqdm
from pymatgen.core import Structure, PeriodicSite
from pymatgen.core.periodic_table import Element
# from pymatgen.core.lattice import get_points_in_sphere
from pymatgen.util.coord import pbc_diff
from pymatgen.io.lammps.data import LammpsBox

__author__ = "Iek-Heng Chu"
__version__ = "1.0"
__date__ = "March 14, 2017"

"""
Algorithms for NEB migration path analysis.
"""


# TODO: (1) ipython notebook example files, unittests


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
        target_dists = []

        # Initial guess of the path (in Cartesian coordinates) used in the IDPP
        # algo.
        init_coords = []

        # Construct the set of target distance matrices via linear interpola-
        # tion between those of end-point structures.
        for i in range(1, nimages + 1):
            # Interpolated distance matrices
            dist = structures[0].distance_matrix + i / (nimages + 1) * (
                structures[-1].distance_matrix - structures[0].distance_matrix
            )
            # linear interpolated distance
            target_dists.append(dist)  # with shape in [ni,na,na]
        target_dists = np.array(target_dists)

        # A set of weight functions. It is set as 1/d^4 for each image. Here,
        # we take d as the average of the target distance matrix and the actual
        # distance matrix.
        weights = np.zeros_like(target_dists, dtype=np.float64)
        for ni in range(nimages):
            avg_dist = (target_dists[ni] + structures[ni + 1].distance_matrix) / 2.0
            weights[ni] = 1.0 / (
                avg_dist ** 4 + np.eye(natoms, dtype=np.float64) * 1e-8
            )

        # Set of translational vector matrices (anti-symmetric) for the images.
        translations = np.zeros((nimages, natoms, natoms, 3), dtype=np.float64)
        for ni, i in itertools.product(range(nimages + 2), range(natoms)):
            frac_coords = structures[ni][i].frac_coords
            init_coords.append(latt.get_cartesian_coords(frac_coords))
            # ?consider periodic boundary condition?
            if ni not in [0, nimages + 1]:
                for j in range(i + 1, natoms):
                    img = latt.get_distance_and_image(
                        frac_coords, structures[ni][j].frac_coords
                    )[1]
                    translations[ni - 1, i, j] = latt.get_cartesian_coords(img)
                    translations[ni - 1, j, i] = -latt.get_cartesian_coords(img)

        # build element string list
        elements = []
        for i in structures[0]:
            elements.append(
                str(i.species.elements[0]) if i.species.is_element else None
            )

        self.init_coords = np.array(init_coords).reshape(nimages + 2, natoms, 3)
        self.translations = translations
        self.weights = weights
        self.structures = structures
        self.target_dists = target_dists
        self.nimages = nimages
        self.natoms = natoms
        self.elements = elements
        self.lammps_dump_box = None
        self.paramerters = {}

    def _screen(self, target_dists: np.array, coords, r=5):
        """
        Screen target_dsit assuming atoms behave like rigid balls.
        Nearest neighbors of each atoms will be recognized and will
        push atoms away if clash occurs; pull them over if there is
        too big a gap.

        Args:
            target_dists: linear interpolated distance for images
            between initial and final structures.
            coords: cartesian coordinates of all images including
            initial and final structures.
            r: neighbor range in angstrom

        Return:
            [target_dists] adjusted target_dists.
        """
        # DEBUG
        # print(">>> _screen")

        natoms = self.structures[0].num_sites
        # TODO dynamically setting r, max_bond_length and radius
        max_bond_length = 4.09
        radius = 1.44
        images = []
        # generate the image structures from coords
        for ni in range(1, self.nimages + 1):
            new_sites = []
            for site, cart_coords in zip(self.structures[ni], coords[ni]):
                new_site = PeriodicSite(
                    site.species,
                    coords=cart_coords,
                    lattice=site.lattice,
                    coords_are_cartesian=True,
                    properties=site.properties,
                )
                new_sites.append(new_site)
            images.append(Structure.from_sites(new_sites))
        # find nearest neighbors within r angstroms
        neighbors = []
        for ni in range(self.nimages):
            neighbors.append(images[ni].get_all_neighbors(r))
        neighbor_indices = []
        for ni in range(self.nimages):
            index_temp = []
            for na in range(natoms):
                temp = []
                for nn in range(len(neighbors[ni][na])):
                    temp.append(neighbors[ni][na][nn].index)
                index_temp.append(temp)
            neighbor_indices.append(index_temp)
        # DEBUG
        neighbor_26 = []
        for ni in range(self.nimages):
            neighbor_26.append(neighbor_indices[ni][25])

        # get distance matrix of each images
        dists = []
        for i in range(self.nimages):
            dists.append(images[i].distance_matrix)
        # # unit vector of all neighbors towrads center atom
        # # unit_vec[ni][na][3]
        # unit_vec = []
        # for ni in range(self.nimages):
        #     temp_vec = np.zeros([natoms, 3])
        #     for na in range(natoms):
        #         for nn in range(len(neighbors[ni][na])):
        #             temp_vec[na] += (images[ni].cart_coords[na]
        #                              - neighbors[ni][na][nn].site.coords)
        #         temp_vec[na] = temp_vec[na] / LA.norm(temp_vec[na])
        #     unit_vec.append(temp_vec)

        # adjust anomalies in neighbors
        # neighbors[ni][na][nn][PerodicSite] --> target_dist[ni][na][na]

        # for ni in range(self.nimages):
        #     anomalies_temp = []
        #     isTooFar = False
        #     isTooClose = False
        #     for na, nn in itertools.product(range(natoms),
        #                                     range(len(neighbors[ni][na]))):
        #         d_temp = neighbors[ni][na][ni].distance
        #         diff_bond = d_temp - max_bond_length
        #         diff_radius = 2 * radius - d_temp
        #         if (diff_radius > 0):
        #             isTooClose = True
        #         if (diff_bond > 0):
        #             isTooFar = True
        #         if (isTooClose):
        #             indices_temp.append(neighbors[ni][na][nn].index)
        #             vector_tmep.append()
        #         if (isTooFar):
        #             indices_temp.append(neighbors[ni][na][nn].index)
        #             vector_tmep.append()
        # images[ni].translate_sites(indices = indices_temp,
        #                            vector = vector_temp)
        # obtain new target_distance
        for ni in range(self.nimages):
            # for i, j in itertools.combinations(range(natoms), 2):
            for i in range(natoms):
                for j in range(i + 1, natoms):
                    # d_ij = target_dists[ni][i][j]
                    # if (d_ij < 2 * radius and d_ij > 0):
                    #     # if too close
                    #     target_dists[ni][i][j] = 2 * radius
                    #     target_dists[ni][j][i] = 2 * radius
                    if dists[ni][i][j] < 2 * radius:
                        # if too close
                        target_dists[ni][i][j] += 1
                        target_dists[ni][j][i] += 1
                    if (
                        j in neighbor_indices[ni][i]
                        and dists[ni][i][j] > max_bond_length
                    ):
                        # if too far
                        # this may push atoms even further
                        target_dists[ni][i][j] = max_bond_length
                        target_dists[ni][j][i] = max_bond_length
        return (target_dists, neighbor_26)


    @classmethod
    def from_endpoints(cls, endpoints, nimages=5, sort_tol=1.0):
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
            pi_bond (float): pi_bond thickness in angstrom. When used will
                add this value to all the elements other than carbon.
        """
        try:
            images = endpoints[0].interpolate(
                endpoints[1], nimages=nimages + 1, autosort_tol=sort_tol
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
        # print("true_forces:")
        # for i in true_forces:
        #     print("{}".format(i))
        # print("natoms: {}  np.shape(true_forces): {}".format(natoms, np.shape(true_forces)))
        

        for ni in range(1, len(x) - 1):
            # TODO add tolerance
            vec1 = (x[ni + 1] - x[ni]).flatten()
            vec2 = (x[ni] - x[ni - 1]).flatten()

            # Local tangent
            tangent = self.get_unit_vector(vec1) + self.get_unit_vector(vec2)
            tangent = self.get_unit_vector(tangent)
            # print("tangent: {}".format(tangent))


            # Spring force
            spring_force = (
                spring_const * (np.linalg.norm(vec1) - np.linalg.norm(vec2)) * tangent
            )
            # print("spring_forces:{}".format(spring_force))
            # for i in spring_force:
            #     print("{}".format(i))
                        
            # Total force
            flat_ft = true_forces[ni - 1].copy().flatten()
            total_force = true_forces[ni - 1] + (
                spring_force - np.dot(flat_ft, tangent) * tangent
            ).reshape(natoms, 3)
            total_forces.append(total_force)
            # print("total_forces:")
            # for i in total_forces:
            #     print("{}".format(i))

        return np.array(total_forces)


    def _NEB_atoms(self, threshold):
        """
        Compare initial and final states of the given path and determine if an atom has
        large enough displacement.

        Args
        initial: fractional coords of initial state
        final: fractional coords of final state
        threshold: numerical threshold in cartesian coords
        """
        NEB_atoms = []
        initial = self.structures[0].frac_coords
        final = self.structures[-1].frac_coords
        frac_diff = pbc_diff(initial, final)
        disp = np.linalg.norm(
            self.structures[0].lattice.get_cartesian_coords(frac_diff), axis=1
        )
        for i in range(self.natoms):
            if disp[i] > threshold:
                NEB_atoms.append(i)
        # debug
        # print("\nNEB atoms list\nIndex | Element | Displacement")
        print("NEB atoms list:")
        for n in NEB_atoms:
            print("-------------")
            print("{}\t{}\t{}".format(n, self.elements[n], disp[n]))
        return NEB_atoms

    def clash_removal_NEB(
        self,
        path,
        maxiter,
        dump_dir,
        ini_structure,
        fin_structure,
        bond_type,
        dump_cr=True,
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
        k_bonded=5.0,
        k_steric=0.15,
        NEB_atoms=[],
        NEB_threshold=2.0,
        pi_bond=0.0,
        **kwargs,
    ):
        """
        Conduct clash removal process on given path with NEB.

        Args:
        path [ni+2, na, 3]: initial fractional coords path for clash removal
        maxiter (int): maximum iteration path (list of Structures): initial path for
        clash removal. The first and last structures correspond to the initial and final
        states.
        moving_atoms (list of int): index of atoms that are allowed to move
        during NEB. If None, then all atoms are allowed to move.
        """

        # generate atom radius
        self.radii_list = self._build_radii_list(ini_structure, fin_structure, bond_type, pi_bond)
        
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

        # find NEB atoms
        NEB_atoms += self._NEB_atoms(NEB_threshold)
        # NEB_atoms += self._NEB_atoms(0.3)
        # print("**********")
        # for i in NEB_atoms:
        #     print("*******NEB_atoms, threshold:{}  {}".format(i, NEB_threshold))
        if not NEB_atoms:
            NEB_atoms = range(self.natoms)
            warnings.warn("No NEB atoms detected. All atoms are considered NEB atoms.")
        
        # initialize for IDPP force calculation
        # old_funcs = np.zeros((self.nimages,), dtype=np.float64)

        # calculate force, update images and output structures with force
        max_forces = [float("inf")]
        initial_step = step_size
        # print("force tolerance:{}".format(ftol))
        for n in tqdm(range(maxiter)):
            
            # true force = IDPP force + CR force
            # total force = true force (perpendicular to tangent)
            #             + spring force (along tangent)
            if k_bonded == 0 and k_steric == 0:
                dump_cr, dump_total = False, False
                true_forces = np.zeros([self.nimages, self.natoms, 3])
            else:
                # for each iteration clash force is evaluated on latest image coords
                clash_forces = self._get_clash_forces_and_energy(
                    image_coords=path_coords[1:-1],
                    NEB_atoms=NEB_atoms,
                    k_bonded=k_bonded,
                    k_steric=k_steric,
                    **kwargs,
                )
                # print("iter:{}".format(n))
                true_forces = clash_forces

            # path_coords includes initial and final states
            total_forces = self._get_total_forces(
                path_coords, true_forces, spring_const
            )
            # print("total_force: {}".format(total_forces))
            # output dump file for each imaegs
            if dump_cr:
                self.dump_writer("dump_cr", dump_dir, path_coords, clash_forces, n)
            if dump_total:
                self.dump_writer("dump_total", dump_dir, path_coords, total_forces, n)

            # calculate displacement. disp_mat[ni][nn][3]
            disp_mat = step_size * total_forces[:, moving_atoms, :]
            # print("moving_atoms:")
            # for n in moving_atoms:
            #     print("{}".format(n))
            disp_mat = np.where(
                np.abs(disp_mat) > max_disp, np.sign(disp_mat) * max_disp, disp_mat
            )
            # update images_coords
            path_coords[1:-1, moving_atoms] += disp_mat

            # DEBUG
            # print(">>>step: {} >>>\n total: {}\ndisp: {}".format(n, total_forces, disp_mat))

            # calculate max force and store
            max_forces.append(np.abs(total_forces[:, moving_atoms, :]).max())
            # print("max force: {}".format(max_forces))
            # stop criteria
            if max_forces[-1] < ftol:
                print("\n>>> Total force converges! >>>")
                break

            # change step size for better optimization
            if step_update_method == "decay" and max_forces[-1] < max_forces[-2]:
                step_size = initial_step * (1 / (1 + 0.01 * n))
    
        else:
            print("\n>>> max force at end of optimization: {} >>>".format(max_forces[-1]))
            warnings.warn(
                "CR-NEB: Maximum iteration number is reached without convergence!",
                UserWarning,
            )

        # apply PBC to all atoms of all images
        clash_removed_path = [self.structures[0]]
        for ni in range(self.nimages):
            new_sites = []
            for site, cart_coords in zip(self.structures[ni + 1], path_coords[ni + 1]):
                new_site = PeriodicSite(
                    site.species,
                    coords=cart_coords,
                    lattice=site.lattice,
                    coords_are_cartesian=True,
                    properties=site.properties,
                )
                new_sites.append(new_site)

            clash_removed_path.append(Structure.from_sites(new_sites))
        # Also include end-point structure.
        clash_removed_path.append(self.structures[-1])

        return clash_removed_path

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

    def _get_clash_forces_and_energy(
        self,
        image_coords,
        NEB_atoms,
        max_bond_tol=0.2,
        # original_distances=None,
        k_steric=5.0,
        k_bonded=0.05,
        repul_tol=0.1,
        steric_tol=1e-8,
        elastic_limit=0.2,
        r_threshold:float = 3.0,
        **kwargs
    ):
        """
        calculate forces and energies

        Args:
        image_coords ([ni,na,3]): current cart coords of each images
        k_steric (float): spring constant for steric hinderance
        steric_shreshold (float): atoms with internulcear distance smaller than
            this value will be subject to repulsive force.
        steric_tol (float): atoms too close together will be regarded as same atoms

        Returns:
            Clash_forces[ni, nn, 3]
        """
        # debug
        # print(k_steric, k_bonded, repul_tol, max_bond_tol)

        # get lattice
        lattice = self.structures[0].lattice

        
        # get frac_coords of current image_coords
        frac_image_coords = []
        for ni in range(self.nimages):
            frac_image_coords.append(lattice.get_fractional_coords(image_coords[ni]))
        
        # calculate attractive forces
        # find bonded neighbors
        bonded_neighbors = []
        
        for i in NEB_atoms:
            bonded_neighbor = lattice.get_points_in_sphere(
                frac_image_coords[ni], image_coords[ni][i], r_threshold, zip_results=True,
            )
            for _, d, index, _ in bonded_neighbor:
                # check if the neighbor is far enough
                # print("atom neighbor:")
                # print("atom: {} neighbor: {} distance: {}".format(i, index, d))
                if (
                    d > self._get_max_bond_length(i, index, max_bond_tol)
                and d < self._get_max_force_length(i, index)
                ):
                    bonded_neighbors.append(BondedNeighbor(i, index, d, ni))
                    # print max_bond_length
                    # print("max_bond_length:")
                    # print("{}\t{}\t{}".format(i,index,self._get_max_bond_length(i,index,max_bond_tol)))
        # print bonded_neighbor
        # print("bonded_neighbors:")
        # for n in bonded_neighbors:
        #     print("{}\t{}\t{}\t{}".format(n.image+1,n.index,n.n_index,n.nn_distance))
        # debug

        
        attractive_forces = np.zeros((self.nimages, self.natoms, 3), dtype=np.float64)
        for nei in bonded_neighbors:
            # each element in bonded_neighbors is a BondedNeighbor object
            ni = nei.image
            i = nei.index
            n_i = nei.n_index
            d = nei.nn_distance
            coord_bonded = image_coords[ni][i]
            coord_pulling = image_coords[ni][n_i]

            # get direction (towards pulling atoms) considering PBC
            direction = self.get_direction_pbc(coord_pulling, coord_bonded)

            # get displacement and calculate force
            delta_d = d - self._get_max_bond_length(i, n_i, max_bond_tol)
            f = 1 * (d ** -4) * (k_bonded * delta_d ** 2) * direction

            # apply force on atoms
            # attractive_forces have no counter forces unless they are both in atom set
            attractive_forces[ni][i] += f
            # DEBUG monitor attractive forces
            """
            print("max_bond_length:")
            print("{}\t{}\t{}".format(i,n_i,self._get_max_bond_length(i,n_i,max_bond_tol)))
            
            print(     
                  "image {} on atom {} attractive force: {} from {} distance: {}".format(
                      ni+1, i, attractive_forces[ni][i], nei.n_index, nei.nn_distance
                  )
            )
            """

        # find steric hindered atoms
        atom_pairs = self._find_steric_hindered_atom(    
            frac_image_coords, image_coords, NEB_atoms, repul_tol, **kwargs
        )
        # print sterci_hindered
        # print("steric_hindered:")
        # for n in steric_hindered:
           #  print("{}".format(n))
        

        # calculate repulsive forces
        repulsive_forces = np.zeros((self.nimages, self.natoms, 3), dtype=np.float64)
        for case in atom_pairs[0]:
            ni, i, j, d = case
            coord1 = image_coords[ni][i]
            coord2 = image_coords[ni][j]
            # direction pointing towards atom i
            direction = self.get_direction_pbc(coord1, coord2)
            delta_d = abs(d - self._get_steric_threshold(i, j, repul_tol))
            f = 1 * (d ** -4) * (k_steric * delta_d ** 2) * direction
            # force and counter force
            repulsive_forces[ni][i] += f
            repulsive_forces[ni][j] += -f
            # print information
            
            """ 
            print("steric_hindered:")
            print("{}\t{}\t{}".format(i,j,self._get_steric_threshold(i,j,repul_tol)))
            
            print(
                  "image {} on atom {} repulsive force: {} from {} distance: {}".format(
                      ni+1, i , repulsive_forces[ni][i], j, d
                  )
            )
            print(
                  "image {} on atom {} repulsive force: {} from {} distance: {}".format(
                      ni+1, j , repulsive_forces[ni][j], i, d
                  )
            )
            """
        
        for case in atom_pairs[1]:
            ni, j, z, d = case
            coord1 = image_coords[ni][j]
            coord2 = image_coords[ni][z]
            # direction pointing towards atom j
            direction = self.get_direction_pbc(coord1, coord2)
            delta_d = abs(d - self._get_steric_threshold(j, z, repul_tol))
            f = 1 * (d ** -4) * (k_steric * delta_d ** 2) * direction
            # force
            repulsive_forces[ni][j] += f
            # print information
        
        # monitor repulsive forces
        # scalar_rpl_force = np.linalg.norm(repulsive_forces, axis=2)
        # max_rpl_force_index = np.argmax(scalar_rpl_force, axis=1)
        # max_rpl_force = np.amax(scalar_rpl_force, axis=1)


        clash_forces = repulsive_forces + attractive_forces
        return clash_forces


    def _find_steric_hindered_atom(
        self,
        frac_coords,
        image_coords,
        NEB_atoms: List,
        repul_tol=0,
        r_threshold: float = 3.0,
        numerical_tol: float = 1e-8,
        steric_tol=1e-8
    ):
        """
        Alls atoms are assumed to be connected. For any atoms, if its closest neighbor
        is further than the corresponding max_bond_length, then all its neighbors will
        pull the atom. The neighbor atom pairs are searched within a radius using
        pymatgen.core.lattice.get_points_in_sphere() which will try to use cthyon code.

        Args:
            frac_coords ([ni,na,3]): fractional coords of current optimizing
                structure.
            moving_sites: list of indices of manually selected NEB atoms
            r (float): radius of the search range.

        Return:
            cases (2D list): [image number, atom index, BondedNeighbor 1,
             BondedNeighbor 2...]
        """
        lattice = self.structures[0].lattice 
        
        # add moving sites to NEB atoms to allow manual selection
        NEB_atoms = set(NEB_atoms)
        
        cases = []
        cases_neighbor = []

        for ni in range(self.nimages):
            clash_atoms = set()
            # find clash atoms around NEB atoms
            for n in NEB_atoms:
                # get_points_in_sphere return
                # [fcoord, dist, indices, supercell_image]
                _, _, temp_atoms, _ = lattice.get_points_in_sphere(
                        frac_coords[ni],
                        image_coords[ni][n],
                        r_threshold,
                        zip_results=False,
                )
                clash_atoms = clash_atoms | set(temp_atoms)
                # delete NEB atoms in clash atoms set
                clash_atoms = clash_atoms ^ NEB_atoms
                # print("image:{}  NEB atom:{}".format(ni+1, n))
                # print("clash atom list:{}".format(clash_atoms))
            # print("image:{}".format(ni+1))
            # print("NEB atoms:{} clash atoms:{}".format(NEB_atoms, clash_atoms))
        # for NEB atoms and clash atoms, check if their neighbors need bonding
        atoms_set = NEB_atoms | clash_atoms
        # print("NEB atoms, clash atoms, atoms set: {}/{}/{}".format(NEB_atoms, clash_atoms, atoms_set))
        atoms_set_list = list(atoms_set)
        atom_pairs = []
        
        
        # get frac_coords of current image_coords
        frac_image_coords = []
        for ni in range(self.nimages):
            frac_image_coords.append(lattice.get_fractional_coords(image_coords[ni]))
        

        for ni in range(self.nimages):
            #     # get frac_coord diff considering pbc
            #     diff = pbc_diff(frac_image_coords[ni][i], frac_image_coords[ni][j])
            #     # convert back to cart coords
            #     dist = np.linalg.norm(lattice.get_cartesian_coords(diff))
            #     if (
            #         dist < self._get_steric_threshold(i, j, repul_tol)
            #         and dist > steric_tol
            #     ):
            #         steric_hindered.append([ni, i, j, dist])
            # calculate cartesian dist of each atom pairs
            
            for i in atoms_set_list:
                steric_hindered = lattice.get_points_in_sphere(
                    frac_coords[ni], image_coords[ni][i], r_threshold, zip_results=True
                )
                for _, d, j, _ in steric_hindered:
                    # print("atom_pair:")
                    # print("{}\t{}\t{}".format(i,j,d))
                    if (
                        d < self._get_steric_threshold(i, j, repul_tol) and d > steric_tol
                    ):
                        if [j, i] not in atom_pairs:
                            atom_pairs.append([i,j])
                            cases.append([ni, i, j, d])
                            # print steric_threshold
                            # print("steric_threshold:")
                            # print("{}\t{}\t{}\t{}".format(i,j,self._get_steric_threshold(i,j,repul_tol),d))
                            if j not in atoms_set_list:
                                second_neighbor = lattice.get_points_in_sphere(
                                        frac_coords[ni], image_coords[ni][j], r_threshold, zip_results=True
                                )
                                for _, d, z, _ in second_neighbor:
                                    if (
                                        d < self._get_steric_threshold(j, z, repul_tol) and d > steric_tol
                                    ):
                                        if z not in atoms_set_list:
                                            cases_neighbor.append([ni, j, z, d])
            """
            
            d = lattice.get_all_distances(
                    frac_image_coords[ni], frac_image_coords[ni]
            )
            for i, j in itertools.combinations(atoms_set_list, 2):
                # print("atom-pairs:")
                # print("{}\t{}".format(i,j))
                if (
                    d[i][j] < self._get_steric_threshold(i,j,repul_tol)
                    and d[i][j] > steric_tol
                ):
                    cases.append([ni, i, j, d[i][j]])
                    # print steric_threshold
                    # print("steric_threshold:")
                    # print("{}\t{}\t{}".format(i,j,self._get_steric_threshold(i,j,repul_tol)))
            """ 
        # print steric_hindered
        # print("steric_hindered:")
        # for ni, i, j, d in cases:
        #     print("{}\t{}\t{}\t{}".format(ni+1,i,j,d))
        
        return cases, cases_neighbor

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

    def _get_max_force_length(self, atom_1, atom_2):
        d = self.radii_list[atom_1] + self.radii_list[atom_2] + 0.5
        return d

    def _get_steric_threshold(self, atom_1, atom_2, tol):
        d = self.radii_list[atom_1] + self.radii_list[atom_2] - tol
        return d

    def _get_max_bond_length(self, atom1, atom2, max_bond_tol):
        d = self.radii_list[atom1] + self.radii_list[atom2] + max_bond_tol
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

        for i in range(0, len(coordinates)):
            x = coordinates[i][0] * basis[0][0] + coordinates[i][1] * basis[1][0] + coordinates[i][2] * basis[2][0]
            y = coordinates[i][0] * basis[0][1] + coordinates[i][1] * basis[1][1] + coordinates[i][2] * basis[2][1]
            z = coordinates[i][0] * basis[0][2] + coordinates[i][1] * basis[1][2] + coordinates[i][2] * basis[2][2]
            coordinates[i][0] = x
            coordinates[i][1] = y
            coordinates[i][2] = z

        return coordinates, elements, num_atoms

    def get_atoms_dict(self, elements, num_atoms, num_1=0):
        atoms_dict = {}
        i = 0
        while i <= len(elements) - 1:
            # print("{}\t{}\t{}".format(elements[i],i,num_atoms[i]))
            if i == 0:
                num = num_atoms[i]
                num_list = list(range(num))
                # print(num)
                # print(list(range(0, 47, 1)))
                atoms_dict[elements[i]] = num_list
                i += 1
                # print(elements_dict)
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
        print(bonds)
        for bond in bonds:
            i = bond.split('-')
            # print(i)
            atoms_combination.append(i)

        return atoms_combination

    def get_average_bond_length(self, atoms_combination, atoms_dicts, coordinates_list_ini, coordinates_list_fin):
        bond_length_dict = {}
        for bond_type in atoms_combination:
            for a, b in itertools.combinations(bond_type, 2):
                if a == b:
                    # print("atom:{}".format(a))
                    length_list = []
                    # print("{}\t{}\t{}".format(a, b, coords_list))
                    for i, j in itertools.combinations(atoms_dicts[a], 2):
                        bond_length_ini = coordinates_list_ini[i] - coordinates_list_ini[j]
                        length_ini = np.linalg.norm(bond_length_ini)
                        bond_length_fin = coordinates_list_fin[i] - coordinates_list_fin[j]
                        length_fin = np.linalg.norm(bond_length_fin)
                        # print("{}\t{}\t{}".format(i, j, length))
                        # print("{}\t{}\t{}\t{}\t{}\t{}".format(min_bond_length, max_bond_length, a, b, i, j))
                        length_list.append(length_ini)
                        length_list.append(length_fin)
                        length_list.sort()
                        # length_mean = np.mean(length_list)
                        # print(length_list)
                        # print("{}\t{}\t length:{}".format(a, b, length_list))
                    bond_length_min = length_list[0]
                    # print("{}".format(length_list[0]))
                    length_list_new = length_list[:]
                    for length in length_list:
                        if length > bond_length_min * 1.15 or length > bond_length_min + 0.3:
                            length_list_new.remove(length)
                    length_list = length_list_new
                    # print(length_list)
                    bond_length_dict[(a, b)] = np.mean(length_list)

                else:
                    length_list = []
                    for i in atoms_dicts[a]:
                        for j in atoms_dicts[b]:
                            bond_length_ini = coordinates_list_ini[i] - coordinates_list_ini[j]
                            length_ini = np.linalg.norm(bond_length_ini)
                            bond_length_fin = coordinates_list_fin[i] - coordinates_list_fin[j]
                            length_fin = np.linalg.norm(bond_length_fin)
                            # print("{}\t{}\t{}".format(i, j, length))
                            length_list.append(length_ini)
                            length_list.append(length_fin)
                            length_list.sort()
                            # length_mean = np.mean(length_list)
                            # print(length_list)
                            # print("{}\t{}\t length:{}".format(a, b, length_list))
                    bond_length_min = length_list[0]
                    # print("{}".format(length_list[0]))
                    length_list_new = length_list[:]
                    for length in length_list:
                        if length > bond_length_min * 1.15 or length > bond_length_min + 0.3:
                            length_list_new.remove(length)
                    length_list = length_list_new
                    # print(length_list)
                    bond_length_dict[(a, b)] = np.mean(length_list)

        return bond_length_dict

    def get_atoms_radius(self, bond_length_dict):
        # elements_list = []
        element_list1 = []
        element_list2 = []
        for key in bond_length_dict.keys():
            element_list1.append(list(key))
            element_list2.extend(list(key))
            elements_list = list(set(element_list2))
        # print(elements_list)
        row_num = len(element_list1)
        column_num = len(elements_list)
        # print(row_num)
        # print(column_num)
        svd_array = np.zeros([row_num, column_num])
        # print(svd_array)
        for i in range(row_num):
            for j in range(column_num):
                if elements_list[j] in element_list1[i]:
                    svd_array[i][j] = element_list1[i].count(elements_list[j])
        # print(svd_array)
        bond_length_array = np.array(list(bond_length_dict.values()))
        u, sigma, vt = np.linalg.svd(svd_array)
        v = np.transpose(vt)
        # print(U)
        w = np.zeros([row_num, column_num])
        for i in range(len(sigma)):
            w[i][i] = sigma[i]
        w_pinv = np.linalg.pinv(w)
        # print(e)
        ut = np.transpose(u)
        # print(VT)
        atoms_radius = np.linalg.multi_dot([v, w_pinv, ut, bond_length_array])
        atoms_radius_dict = {}
        for i in elements_list:
            atoms_radius_dict[i] = atoms_radius[elements_list.index(i)]
        # print(atoms_radius_dict)

        return atoms_radius_dict

    def _build_radii_list(self, ini_structure, fin_structure, bond_type, pi_bond=0.0):
        """
        Build radii list of each atoms in the structure. If steric_threshold or
        max_bond_length is not manually set, radii_list should be used to
        automaticallty generate those parameters according to atomic radius.
        """
        # radii_table will overide radii of pymatgen element atomic_radius.
        # The value for Alkali metals (Li, Na, K) is measured on models of ions
        # adsorped on graphene, therefore it has accounted for the pi_bond radius.
        # Other metals are measured on corresponding unit cells of Material Studio.
        # data_ini = read_vasp(input("Please input the initial CONTCAR: "))
        data_ini = self.read_poscar(ini_structure)
        coords_ini = data_ini[0]
        elements = data_ini[-2]
        number_atoms = data_ini[-1]
        coord_array_ini = np.array(coords_ini)
        coordinates_list_ini = coord_array_ini.reshape(sum(number_atoms), 3)
        atoms_dicts = self.get_atoms_dict(elements, number_atoms)
        # data_fin = read_vasp(input("Please input the final CONTCAR: "))
        data_fin = self.read_poscar(fin_structure)
        coords_fin = data_fin[0]
        coord_array_fin = np.array(coords_fin)
        coordinates_list_fin = coord_array_fin.reshape(sum(number_atoms), 3)

        atoms_combination = self.get_atoms_combination(bond_type)
        # print(atoms_combination)
        average_bond_length = self.get_average_bond_length(atoms_combination, atoms_dicts, coordinates_list_ini,
                                                      coordinates_list_fin)
        # print(average_bond_length)
        radii_table = self.get_atoms_radius(average_bond_length)
        # print(radii_table)
        radii = []
        output_list = {}
        for e in self.elements:
            if e in radii_table:
                radii.append(radii_table[e] + pi_bond)
            else:
                radii.append(Element(e).atomic_radius + pi_bond)
            if e not in output_list:
                output_list[e] = radii[-1]
        # print(radii)
        # show which radius is being used
        for k in output_list:
            print("element:{} radius:{}".format(k, output_list[k]))
        return radii

    # def _get_radius(self, atom_index, pi_bond):
    #     structure = self.structures[0]

    #     if structure[atom_index].species.is_element:
    #         elmnt = structure[atom_index].species.elements[0]
    #         # TODO ionic radii list should be added
    #         if elmnt == Element("Li"):
    #             r = 0.90 + pi_bond
    #             # r = 1.34 + pi_bond
    #         elif set() == Element("Na"):
    #             r = 1.16 + pi_bond
    #         elif elmnt == Element("K"):
    #             r = 1.52 + pi_bond
    #         elif elmnt == Element("C"):
    #             r = elmnt.atomic_radius
    #         else:
    #             r = elmnt.atomic_radius + pi_bond
    #     else:
    #         raise ValueError("sites in structures should be elements not compositions")
    #     print("element:{} radius:{}".format(elmnt, r))
    #     return r


class BondedNeighbor:
    """
    container for bonded neighbors used during calculation of bonded force
    """

    def __init__(self, index, n_index, nn_distance, image):
        """
        index: index of the atom in question
        n_index: the index of bonded neighbor to the atom in question
        nn_distance: the distance between two atoms
        image: the number of image
        """
        self.index = index
        self.n_index = n_index
        self.nn_distance = nn_distance
        self.image = image

    def __repr__(self):
        return (
            "atom index: {} bonded neighbor index:{} " " distance: {} in image {}\n"
        ).format(self.index, self.n_index, self.nn_distance, self.image)
