# Copyright 2024 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..processors.processor import Processor
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import cKDTree as KDTree
from ..graph_utils import make_residue_graph
from itertools import product
from vermouth.file_writer import deferred_open
from collections import defaultdict
from vermouth import __version__ as VERSION

# BOND TYPE
# Types of contacts:
# HB -- 1 -- hydrogen-bond
# PH -- 2 -- hydrophobic
# AR -- 3 -- aromatic - contacts between aromatic rings
# IB -- 4 -- ionic bridge - contacts created by two atoms with different charges
# DC -- 5 -- destabilizing contact - contacts which are in general repulsive
# OT -- 6 -- denotes negligible other contacts.
# 1-HB,2-PH,3-AR,4-IP,5-DC,6-OT
BOND_TYPE = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 5, 5, 6, 6, 6, 1, 1],
                      [0, 1, 5, 1, 5, 5, 6, 6, 6, 1, 5],
                      [0, 1, 1, 5, 5, 5, 6, 6, 6, 5, 1],
                      [0, 5, 5, 5, 2, 2, 6, 6, 6, 5, 5],
                      [0, 5, 5, 5, 2, 3, 6, 6, 6, 5, 5],
                      [0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                      [0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                      [0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                      [0, 1, 1, 5, 5, 5, 6, 6, 6, 5, 4],
                      [0, 1, 5, 1, 5, 5, 6, 6, 6, 4, 5]])

PROTEIN_MAP = {
    "ALA": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0}
    },
    "ARG": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.88, 'atype': 4},
        'CD':       {'vrad': 1.88, 'atype': 7},
        'NE':       {'vrad': 1.64, 'atype': 3},
        'CZ':       {'vrad': 1.61, 'atype': 6},
        'NH1':      {'vrad': 1.64, 'atype': 3},
        'NH2':      {'vrad': 1.64, 'atype': 3},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0}
    },
    "ASN": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.61, 'atype': 6},
        'OD1':      {'vrad': 1.42, 'atype': 2},
        'ND2':      {'vrad': 1.64, 'atype': 3},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "ASP": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.61, 'atype': 6},
        'OD1':      {'vrad': 1.46, 'atype': 2},
        'OD2':      {'vrad': 1.42, 'atype': 2},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "CYM": {},
    "CYX": {},
    "CYS": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'SG':       {'vrad': 1.77, 'atype': 6},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "GLN": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.88, 'atype': 4},
        'CD':       {'vrad': 1.61, 'atype': 6},
        'OE1':      {'vrad': 1.42, 'atype': 2},
        'NE2':      {'vrad': 1.64, 'atype': 3},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "GLU": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.88, 'atype': 4},
        'CD':       {'vrad': 1.61, 'atype': 6},
        'OE1':      {'vrad': 1.46, 'atype': 2},
        'OE2':      {'vrad': 1.42, 'atype': 2},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "GLY": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':        {'vrad': 1.88, 'atype': 6},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "HIE": {},
    "HIP": {},
    "HIS": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.61, 'atype': 5},
        'ND1':      {'vrad': 1.64, 'atype': 1},
        'CD2':      {'vrad': 1.76, 'atype': 5},
        'CE1':      {'vrad': 1.76, 'atype': 5},
        'NE2':      {'vrad': 1.64, 'atype': 1},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "ILE": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG1':      {'vrad': 1.88, 'atype': 4},
        'CG2':      {'vrad': 1.88, 'atype': 4},
        'CD':       {'vrad': 1.88, 'atype': 4},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "LEU": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.88, 'atype': 4},
        'CD1':      {'vrad': 1.88, 'atype': 4},
        'CD2':      {'vrad': 1.88, 'atype': 4},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "LYS": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.88, 'atype': 4},
        'CD':       {'vrad': 1.88, 'atype': 4},
        'CE':       {'vrad': 1.88, 'atype': 7},
        'NZ':       {'vrad': 1.64, 'atype': 3},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "MET": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.88, 'atype': 4},
        'SD':       {'vrad': 1.77, 'atype': 8},
        'CE':       {'vrad': 1.88, 'atype': 4},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "PHE": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.88, 'atype': 5},
        'CD1':      {'vrad': 1.61, 'atype': 5},
        'CD2':      {'vrad': 1.76, 'atype': 5},
        'CE1':      {'vrad': 1.76, 'atype': 5},
        'CE2':      {'vrad': 1.76, 'atype': 5},
        'CZ':       {'vrad': 1.76, 'atype': 5},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "PRO": {
        'N':        {'vrad': 1.64, 'atype': 6},
        'CA':       {'vrad': 1.88, 'atype': 4},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.88, 'atype': 4},
        'CD':       {'vrad': 1.88, 'atype': 4},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "SER": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 6},
        'OG':       {'vrad': 1.46, 'atype': 1},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "THR": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 6},
        'OG1':      {'vrad': 1.46, 'atype': 1},
        'CG2':      {'vrad': 1.88, 'atype': 4},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "TRP": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.61, 'atype': 5},
        'CD1':      {'vrad': 1.76, 'atype': 5},
        'CD2':      {'vrad': 1.61, 'atype': 5},
        'NE1':      {'vrad': 1.64, 'atype': 3},
        'CE2':      {'vrad': 1.61, 'atype': 5},
        'CE3':      {'vrad': 1.76, 'atype': 5},
        'CZ2':      {'vrad': 1.76, 'atype': 5},
        'CZ3':      {'vrad': 1.76, 'atype': 5},
        'CH2':      {'vrad': 1.76, 'atype': 5},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "TYR": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG':       {'vrad': 1.61, 'atype': 5},
        'CD1':      {'vrad': 1.76, 'atype': 5},
        'CD2':      {'vrad': 1.76, 'atype': 5},
        'CE1':      {'vrad': 1.76, 'atype': 5},
        'CE2':      {'vrad': 1.76, 'atype': 5},
        'CZ':       {'vrad': 1.61, 'atype': 5},
        'OH':       {'vrad': 1.46, 'atype': 1},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    },
    "VAL": {
        'N':        {'vrad': 1.64, 'atype': 3},
        'CA':       {'vrad': 1.88, 'atype': 7},
        'C':        {'vrad': 1.61, 'atype': 6},
        'O':        {'vrad': 1.42, 'atype': 2},
        'CB':       {'vrad': 1.88, 'atype': 4},
        'CG1':      {'vrad': 1.88, 'atype': 4},
        'CG2':      {'vrad': 1.88, 'atype': 4},
        'OXT':      {'vrad': 1.42, 'atype': 2},
        'default':  {'vrad': 0.00, 'atype': 0},
    }
}


def _get_vdw_radius(resname, atomname):
    """
    get the vdw radius of an atom indexed internally within a serially numbered residue
    """
    try:
        res_vdw = PROTEIN_MAP[resname]
    except KeyError:
        return 0.00

    try:
        atom_vdw = res_vdw[atomname]
    except KeyError:
        atom_vdw = res_vdw['default']
    return atom_vdw['vrad']


def _get_atype(resname, atomname):
    """
    get the vdw radius of an atom indexed internally within a serially numbered residue
    """
    try:
        res_vdw = PROTEIN_MAP[resname]
    except KeyError:
        return 0

    try:
        atom_vdw = res_vdw[atomname]
    except KeyError:
        atom_vdw = res_vdw['default']

    return atom_vdw['atype']


def _make_surface(position, fiba, fibb, vrad):
    """
    Generate points on a sphere using Fibonacci points

    position: np.array
        shape (3,) array of an atomic position to build a sphere around
    fiba: int. n-1 fibonacci number to build number of points on sphere
    fibb: int. n fibonacci number to build number of points on sphere
    vrad: float. VdW radius of the input atom to build a sphere around.

    position: centre of sphere
    """

    x, y, z = position

    k = np.arange(fibb)
    phi_aux = (np.arange(1, fibb+1) * fiba) % fibb
    phi_aux[phi_aux == 0] = fibb
    theta = np.arccos(1.0 - 2.0 * k / fibb)
    phi = 2.0 * np.pi * phi_aux / fibb
    surface_x = x + vrad * np.sin(theta) * np.cos(phi)
    surface_y = y + vrad * np.sin(theta) * np.sin(phi)
    surface_z = z + vrad * np.cos(theta)
    surface = np.stack((surface_x, surface_y, surface_z), axis=-1)

    return surface


def atom2res(arrin, nresidues, atom_map, norm=False):
    '''
    take an array with atom level data and sum the entries over within the residue
    '''

    out = np.zeros((nresidues, nresidues))
    for res_idx, res_jdx in product(atom_map.keys(), atom_map.keys()):
        atom_idxs = atom_map[res_idx]
        atom_jdxs = atom_map[res_jdx][:, np.newaxis]
        value = arrin[atom_idxs, atom_jdxs].sum()
        out[res_idx, res_jdx] = value

    if norm:
        out[out > 0] = 1

    return out


def _contact_info(molecule):
    """
    get the atom attributes that we need to calculate the contacts
    """

    G = make_residue_graph(molecule)

    resids = []
    chains = []
    resnames = []
    positions_all = []
    ca_pos = []
    vdw_list = []
    atypes = []
    res_serial = []
    res_idx = []
    for residue in G.nodes:
        # we only need these for writing at the end
        resnames.append(G.nodes[residue]['resname'])
        resids.append(G.nodes[residue]['resid'])
        chains.append(G.nodes[residue]['chain'])
        res_idx.append(G.nodes[residue]['_res_serial'])
        subgraph = G.nodes[residue]['graph']

        for atom in sorted(subgraph.nodes):
            position = subgraph.nodes[atom].get('position', [np.nan]*3)
            if np.isfinite(position).all():
                res_serial.append(subgraph.nodes[atom]['_res_serial'])

                positions_all.append(subgraph.nodes[atom]['position'] * 10)

                vdw_list.append(_get_vdw_radius(subgraph.nodes[atom]['resname'],
                                                subgraph.nodes[atom]['atomname']))
                atypes.append(_get_atype(subgraph.nodes[atom]['resname'],
                                         subgraph.nodes[atom]['atomname']))

                if subgraph.nodes[atom]['atomname'] == 'CA':
                    ca_pos.append(subgraph.nodes[atom]['position'])


    vdw_list = np.array(vdw_list)
    atypes = np.array(atypes)
    coords = np.stack(positions_all)
    res_serial = np.array(res_serial)

    resids = np.array(resids)
    chains = np.array(chains)
    resnames = np.array(resnames)
    res_idx = np.array(res_idx)

    # 2) find the number of residues that we have
    nresidues = len(G)

    return vdw_list, atypes, coords, res_serial, resids, chains, resnames, res_idx, ca_pos, nresidues, G

def _calculate_overlap(coords_tree, vdw_list, natoms, vdw_max, alpha=1.24):
    """
    Find enlarged (OV) overlap contacts

    coords_tree: KDTree
        KDTree of the input coordinates
    vdw_list: list
        list of vdw radii of the input coordinates
    natoms: int
        number of atoms in the molecule
    vdw_max: float
        maximum possible vdw radius of atoms
    alpha: float
        Enlargement factor for attraction effects
    """
    over = np.zeros((natoms, natoms))
    over_sdm = coords_tree.sparse_distance_matrix(coords_tree, 2 * vdw_max * alpha)
    for (idx, jdx), distance_between in over_sdm.items():
        if idx != jdx:
            if distance_between < (vdw_list[idx] + vdw_list[jdx]) * alpha:
                over[idx, jdx] = 1
    return over

def _calculate_csu(coords, vdw_list, fiba, fibb, natoms, coords_tree, vdw_max, water_radius=2.80):
    """
    Calculate contacts of structural units (CSU)

    coords: Nx3 numpy array
        coordinates of atoms in the molecule
    vdw_list: list
        vdw radii of the atoms in the molecule
    fiba, fibb: int
        n-1th and nth fibonacci numbers from which to generate points on a sphere around the input coordinate
    natoms: int
        number of atoms in the molecule
    coords_tree: KDTree
        KDTree of the input coordinates
    vdw_max: float
        maximum possible vdw radius of atoms
    water_radius: float
        radius of water molecule in A

    Returns:
    hit_results: natoms x fibb np.array
        each i,j entry is the index of the atom in coords which is the closest atom to atom i at index j of the
        fibonacci sphere

    """

    #setup arrays to keep track
    hit_results = np.full((natoms, fibb), -1)
    dists_counter = np.full((natoms, fibb), np.inf)

    # sparse matrix with a cutoff at the maximum possible distance for a contact
    surface_sdm = coords_tree.sparse_distance_matrix(coords_tree, (2 * vdw_max) + water_radius)
    # n.b. this loop works because sparse_distance_matrix is sorted by (idx, jdx) pairs
    for (idx, jdx), distance_between in surface_sdm.items():
        # don't take atoms which are identical
        if idx == jdx:
            continue

        # check that the distance between them is shorter than the vdw sum and the water radius
        if distance_between >= (vdw_list[idx] + vdw_list[jdx] + water_radius):
            continue

        # Generate the fibonacci sphere for this point and make a KDTree from it
        base_tree = KDTree(_make_surface(coords[idx], fiba, fibb, vdw_list[idx]+water_radius))

        # find points on the base point sphere which are within the vdw cutoff of the target point's coordinate
        res = np.array(base_tree.query_ball_point(coords[jdx], vdw_list[jdx] + water_radius))

        # if we have any results
        if len(res) > 0:
            # find where the distance between the two points is smaller than the current recorded distance
            # at the points which are within the cutoff
            to_fill = np.where(distance_between < dists_counter[idx][res])[0]

            # record the new distances and indices of the points
            dists_counter[idx][res[to_fill]] = distance_between
            hit_results[idx][res[to_fill]] = jdx

    return hit_results


def _contact_types(hit_results, natoms, atypes):
    """
    From CSU contacts, establish contact types from atomtypes

    hit_results: NxM ndarray
        array for N atoms in molecule for M fibonnaci points on each atom.
        Each i,j entry is the index of the atom which is the closest contact to i
    natoms: int
        number of atoms in the molecule
    atypes: array
        list of the atomtypes of each atom in the molecule
    """

    contactcounter_1 = np.zeros((natoms, natoms))
    stabilisercounter_1 = np.zeros((natoms, natoms))
    destabilisercounter_1 = np.zeros((natoms, natoms))

    for i, j in enumerate(hit_results):
        for k in j:
            if k >= 0:
                at1 = atypes[i]
                at2 = atypes[k]
                if (at1 > 0) and (at2 > 0):
                    contactcounter_1[i, k] += 1
                    btype = BOND_TYPE[at1, at2]
                    if btype <= 4:
                        stabilisercounter_1[i, k] += 1
                    if btype == 5:
                        destabilisercounter_1[i, k] += 1

    return contactcounter_1, stabilisercounter_1, destabilisercounter_1

def make_atom_map(res_serial):

    atom_map = defaultdict(list)
    for atom_idx, res_idx in enumerate(res_serial):
        atom_map[res_idx].append(atom_idx)
    for key, value in atom_map.items():
        atom_map[key] = np.array(value)

    return atom_map

def _calculate_contacts(vdw_list, atypes, coords, res_serial, nresidues):
    """
    run the contact calculation functions

    vdw_list: np.array
        list of the vdw radii of the atoms in the system
    atypes: np.array
        list of the atom types in the system to determine the nature of contacts
    coords: nx3 array
        coordinates of all the atoms in the system
    res_serial: np.array
        list of the serial residue number of each atom in the system
    nresidues: int
        number of residues in the system
    """

    # some initial definitions of variables that we need
    fib = 14
    fiba, fibb = 0, 1
    for _ in range(fib):
        fiba, fibb = fibb, fiba + fibb

    natoms = len(coords)

    vdw_max = max(item['vrad'] for atoms in PROTEIN_MAP.values() for item in atoms.values())

    # make the KDTree of the input coordinates
    coords_tree = KDTree(coords)

    # calculate the OV contacts of the molecule
    over = _calculate_overlap(coords_tree, vdw_list, natoms, vdw_max, alpha=1.24)

    # Calculate the CSU contacts of the molecule
    hit_results = _calculate_csu(coords, vdw_list, fiba, fibb, natoms, coords_tree, vdw_max, water_radius=2.80)

    # find the types of contacts we have
    contactcounter_1, stabilisercounter_1, destabilisercounter_1 = _contact_types(hit_results, natoms, atypes)

    atom_map = make_atom_map(res_serial)

    # transform the resolution between atoms and residues
    overlapcounter_2 = atom2res(over, nresidues, atom_map, norm=True)
    contactcounter_2 = atom2res(contactcounter_1, nresidues, atom_map)
    stabilisercounter_2 = atom2res(stabilisercounter_1, nresidues, atom_map)
    destabilisercounter_2 = atom2res(destabilisercounter_1, nresidues, atom_map)

    return overlapcounter_2, contactcounter_2, stabilisercounter_2, destabilisercounter_2


def _get_contacts(nresidues, overlaps, contacts, stabilisers, destabilisers, res_idx, G):
    '''
    Generate contacts list from the contact arrays calculated

    nresidues: int
        number of residues in the molecule
    overlaps: ndarray
        nresidues x nresidues array of OV contacts in the molecule
    contacts: ndarray
        nresidues x nresidues array of CSU contacts in the molecule
    stabilisers: ndarray
        nresidues x nresidues array of CSU stabilising contacts in the molecule
    destabilisers: ndarray
        nresidues x nresidues array of CSU destabilising contacts in the molecule
    res_idx: list
        list of serial residue ids for each of the residues
    G: nx.Graph
        residue based graph of the molecule
    '''
    contacts_list = []
    all_contacts = []
    for i1, i2 in product(np.arange(nresidues), np.arange(nresidues)):
        if i1 == i2:
            continue
        over = overlaps[i1, i2]
        cont = contacts[i1, i2]
        stab = stabilisers[i1, i2]
        dest = destabilisers[i1, i2]
        rcsu = (stab - dest) > 0

        if (over > 0 or cont > 0):
            a = np.where(res_idx == i1)[0][0]
            b = np.where(res_idx == i2)[0][0]
            all_contacts.append([i1+1, i2+1, a, b, over, cont, stab, rcsu])
            if over == 1 or (over == 0 and rcsu):
                # this is a OV or rCSU contact we take it
                contacts_list.append((int(G.nodes[a]['resid']), G.nodes[a]['chain'],
                                      int(G.nodes[b]['resid']), G.nodes[b]['chain']))

    return contacts_list, all_contacts


def _write_contacts(fout, all_contacts, ca_pos, G):
    '''
    write the contacts calculated to file
    fout: str
        path to write file to
    all_contacts: list
        list of lists of every contact found
    ca_pos: list
        list of (3,) arrays with the position of the CA atom of each residue
    G: nx.Graph
        residue graph of the input molecule
    '''

    header = [f"Go contact map calculated with vermouth {VERSION}\n\n"]

    header.append("Residue-Residue Contacts\n"
                  "\n"
                  "ID       - atom identification\n"
                  "I1,I2    - serial residue id\n"
                  "AA       - 3-letter code of aminoacid\n"
                  "C        - chain\n"
                  "I(PDB)   - residue number in PDB file\n"
                  "DCA      - distance between CA\n"
                  "CMs      - OV , CSU , oCSU , rCSU\n"
                  "           (CSU does not take into account chemical properties of atoms)\n"
                  "rCSU     - net contact from rCSU\n"
                  "Count    - number of contacts between residues\n"
                  "\n"
                  "      ID    I1  AA  C I(PDB)     I2  AA  C I(PDB)        DCA       CMs    rCSU   Count \n"
                  "=======================================================================================\n")

    msgs = []
    count = 0
    for contact in all_contacts:
        count += 1
        msg = (f"R {int(count):6d} "
               f"{int(contact[0]):5d}  {G.nodes[contact[2]]['resname']:3s} "
               f"{G.nodes[contact[2]]['chain']:1s} {int(G.nodes[contact[2]]['resid']):4d}    "
               f"{int(contact[1]):5d}  {G.nodes[contact[3]]['resname']:3s} "
               f"{G.nodes[contact[3]]['chain']:1s} {int(G.nodes[contact[3]]['resid']):4d}    "
               f"{euclidean(ca_pos[contact[2]], ca_pos[contact[3]])*10:9.4f}     "
               f"{int(contact[4]):1d} {1 if contact[5] != 0 else 0} "
               f"{1 if contact[6] != 0 else 0} {1 if contact[7] else 0}"
               f"{int(contact[7]): 6d}  {int(contact[5]): 6d}\n")
        msgs.append(msg)
    message_out = ''.join(msgs)
    with deferred_open(fout, "w") as f:
        f.write(''.join(header))
        f.write(message_out)


"""
Read RCSU Go model contact maps.
"""


def read_go_map(system, file_path):
    """
    Read a RCSU contact map from the c code as published in
    doi:10.5281/zenodo.3817447. The format requires all
    contacts to have 18 columns and the first column to be
    a capital R.

    Parameters
    ----------
    system: vermouth.system.System
        The system to process. Is modified in-place.
    file_path: :class:`pathlib.Path`
        path to the contact map file

    Returns
    -------
    list(tuple)
        contact as chain id, res id, chain id, res id
    """
    with open(file_path, "r", encoding='UTF-8') as _file:
        contacts = []
        for line in _file:
            tokens = line.strip().split()
            if len(tokens) == 0:
                continue

            if tokens[0] == "R" and len(tokens) >= 18:
                # this is a bad place to filter but follows
                # the old script
                if tokens[11] == "1" or (tokens[11] == "0" and tokens[14] == "1"):
                    # this is a OV or rCSU contact we take it
                    contacts.append((int(tokens[5]), tokens[4], int(tokens[9]), tokens[8]))

        if len(contacts) == 0:
            raise IOError("You contact map is empty. Are you sure it has the right formatting?")

    system.go_params["go_map"].append(contacts)


def do_contacts(molecule, write_file):
    '''
    master function to calculate Go contacts

    molecule: vermouth.Molecule
        molecule to calculate contacts for
    write_file: bool
        write the file of the contacts out
    '''
    vdw_list, atypes, coords, res_serial, resids, chains, resnames, res_idx, ca_pos, nresidues, mol_graph = _contact_info(
        molecule)

    overlaps, contacts, stabilisers, destabilisers = _calculate_contacts(vdw_list,
                                                                        atypes,
                                                                        coords,
                                                                        res_serial,
                                                                        nresidues)

    contacts, all_contacts = _get_contacts(nresidues,
                            overlaps, contacts,
                            stabilisers,
                            destabilisers,
                            res_idx,
                            mol_graph)

    if isinstance(write_file, str):
        _write_contacts(write_file, all_contacts, ca_pos, mol_graph)

    return contacts


class GenerateContactMap(Processor):
    """
    Processor to generate the contact rCSU contact map for a protein from an atomistic structure
    """
    def __init__(self, write_file):
        self.write_file = write_file

    def run_molecule(self, molecule):
        """
        Process `system`.

        Parameters
        ----------
        system: vermouth.system.System
            The system to process. Is modified in-place.
        """
        return do_contacts(molecule, self.write_file)

    def run_system(self, system):
        for molecule in system.molecules:
            contacts = self.run_molecule(molecule)
            system.go_params["go_map"].append(contacts)
