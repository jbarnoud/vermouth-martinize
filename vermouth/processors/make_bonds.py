#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2018 University of Groningen
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
"""
Provides a processor that can add edges to a graph based on geometric criteria.
"""

from collections import defaultdict
import networkx as nx
import numpy as np

from .. import KDTree
from ..molecule import Molecule
from .processor import Processor
from ..utils import format_atom_string

from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))


# Van der Waals radii from A. Bondi, J. Phys. Chem., 68, 441-452, 1964.
# https://doi.org/10.1021/j100785a001
# For hydrogen, we use R.S. Rowland & R. Taylor, J.Phys.Chem., 100, 7384-7391, 1996.
# https://doi.org/10.1021/jp953141
VDW_RADII = {  # in nm
    'H': 0.120,
    'He': 0.140,
    'C': 0.170,
    'N': 0.155,
    'O': 0.152,
    'F': 0.147,
    'Ne': 0.154,
    'Si': 0.210,
    'P': 0.180,
    'S': 0.180,
    'Cl': 0.175,
    'Ar': 0.188,
    'As': 0.185,
    'Se': 1.90,
    'Br': 0.185,
    'Kr': 0.202,
    'Te': 0.206,
    'I': 0.198,
    'Xe': 0.216,
}
#VALENCES = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'S': 6}


def _residue_key_func(node):
    """
    Creates a residue "key" for a node. Keys should be identical only for nodes
    that are part of the same residue.
    """
    attrs = 'mol_idx', 'chain', 'resid', 'resname'
    return tuple(node.get(attr) for attr in attrs)


def _collect_residues(graph):
    """
    Creates groups of indices per residue.
    Returns a dict of {<key>: set(indices)}. <key> comes from _residue_key_func.
    """
    residues = defaultdict(set)
    for node_idx in graph:
        key = _residue_key_func(graph.nodes[node_idx])
        residues[key].add(node_idx)
    return dict(residues)


def _bonds_from_distance(graph, nodes=None, non_edges=None, fudge=1.0):
    """Add edges to `graph` between `nodes` based on distance.

    Adds edges to `graph` between nodes in `nodes`, but will never add an edge
    that is in `non_edges`. Edges are added based on a simple distance
    criterion. The criterion can be adjusted using `fudge`. Nodes need to have
    an element attribute that is in VDW_RADII in order to be eligible.

    Parameters
    ----------
    graph: networkx.Graph
    nodes: collections.abc.Collection[collections.abc.Hashable]
    non_edges: collections.abc.Container[frozenset[collections.abc.Hashable, collections.abc.Hashable]]
    fudge: float
    """
    if not nodes:
        nodes = graph.nodes
    if not non_edges:
        non_edges = set()

    # We filter out the nodes for which we do not know the radius. Indeed, we
    # consider these nodes cannot make bonds. The filtering is done before we
    # enter the KDTree; we only provide to the KDTree the position of the nodes
    # that could make a bond. `idx_to_nodenum` make the link between the
    # indices in the `positions` array, and the node keys in the `system`
    # graph.
    idx_to_nodenum = {
        idx: n
        for idx, n in enumerate(
            subn
            for subn in graph
            if subn in nodes and graph.nodes[subn].get('element') in VDW_RADII
        )
    }
    max_dist = max(VDW_RADII.values()) * fudge

    positions = np.array([
        graph.nodes[node]['position']
        for node in idx_to_nodenum
    ], dtype=float)

    # Pylint ignore because positions is a numpy array.
    if len(positions):  # pylint: disable=len-as-condition
        positions = np.atleast_2d(positions)
        tree = KDTree(positions)
        pairs = tree.sparse_distance_matrix(tree, max_dist * fudge)
    else:
        pairs = {}

    for (idx1, idx2), dist in pairs.items():
        if idx1 >= idx2:
            continue
        node_idx1 = idx_to_nodenum[idx1]
        node_idx2 = idx_to_nodenum[idx2]
        atom1 = graph.nodes[node_idx1]
        atom2 = graph.nodes[node_idx2]
        element1 = atom1['element']
        element2 = atom2['element']

        if frozenset((node_idx1, node_idx2)) in non_edges:
            continue

        bond_distance = 0.5 * (VDW_RADII[element1] + VDW_RADII[element2])
        if dist <= bond_distance * fudge and not graph.has_edge(node_idx1, node_idx2):
            LOGGER.debug("Guessed bond between {} and {} based on distance.",
                         format_atom_string(atom1),
                         format_atom_string(atom2))

            graph.add_edge(node_idx1, node_idx2, distance=dist)


def _bonds_from_names(graph, resname, nodes, force_field):
    """Add edges between `nodes` in `graph` based on atom names.

    Adds edges to `graph`, assuming the nodes in `nodes` constitute a residue
    with residue name `resname`, which can be found among the `force_field`
    blocks. Edges will be added as they are in the reference Block. In addition,
    all non-edges in the Block will be generated and returned.

    Parameters
    ----------
    graph: networkx.Graph
    resname: str
    nodes: collections.abc.Iterable[collections.abc.Hashable]
        Should be node keys in `graph`
    force_field: vermouth.forcefield.ForceField
        Force field in which to look for the block with name `resname`

    Raises
    ------
    KeyError
        If `resname` is not one of the blocks known to `force_field`; or when
        a residue contains duplicate atom names.

    Returns
    -------
    Set[Frozenset[collections.abc.Hashable, collections.abc.Hashable]]
        All non-edges found in the block, with node keys from `graph`.
    """
    block = force_field.blocks.get(resname)
    if not block:
        raise KeyError("Residue {} is not known to force field {}"
                       "".format(resname, force_field.name))

    mol_name_to_idx = defaultdict(set)
    for graph_idx in nodes:
        if 'atomname' in graph.nodes[graph_idx]:
            mol_name_to_idx[graph.nodes[graph_idx]['atomname']].add(graph_idx)
    mol_name_to_idx = dict(mol_name_to_idx)
    for name, graph_idxs in mol_name_to_idx.items():
        if len(graph_idxs) > 1:
            raise KeyError("Residue has multiple atoms with atom name {}"
                           "".format(name))
        mol_name_to_idx[name] = mol_name_to_idx[name].pop()

    for block_idx, block_jdx in block.edges:
        block_idx_name = block.nodes[block_idx]['atomname']
        block_jdx_name = block.nodes[block_jdx]['atomname']
        if block_idx_name in mol_name_to_idx and block_jdx_name in mol_name_to_idx:
            graph_idx = mol_name_to_idx[block_idx_name]
            graph_jdx = mol_name_to_idx[block_jdx_name]
            pos1 = np.array(graph.nodes[graph_idx].get('position', np.full(3, np.nan)))
            pos2 = np.array(graph.nodes[graph_jdx].get('position', np.full(3, np.nan)))
            dist = np.sqrt(np.sum((pos1 - pos2)**2))
            graph.add_edge(graph_idx, graph_jdx, distance=dist)

    non_edges = set()
    for block_idx, block_jdx in nx.non_edges(block):
        block_idx_name = block.nodes[block_idx]['atomname']
        block_jdx_name = block.nodes[block_jdx]['atomname']
        if block_idx_name in mol_name_to_idx and block_jdx_name in mol_name_to_idx:
            non_edges.add(frozenset((mol_name_to_idx[block_idx_name],
                                     mol_name_to_idx[block_jdx_name])))
    return non_edges


def make_bonds(molecule, allow_name=True, allow_dist=True, fudge=1.0):
    """Creates bonds within a molecule.

    First, edges will be created based on residue and atom names. Second, edges
    will be created based on a distance criterion. Nodes in system must have
    `position` and `element` attributes. The possible distance between nodes is
    determined by values in `VDW_RADII`. Edges within residues will only be
    guessed between atoms that are not known in the reference Block.

    Notes
    -----
    Edges for residues for which no block can be found will be added based on
        the distance criterion. A warning will be issued if this is the case.

    Elements that are not in `VDW_RADII` do not make bonds based on distances.

    Parameters
    ----------
    molecule: :class:`~vermouth.molecule.Molecule`
        The molecule in which to add edges.
    fudge: :class:`~numbers.Number`
        Scale the allowed distance by this factor.

    Returns
    -------
    :class:`~vermouth.molecule.Molecule`
        Molecule in which edges have been added based on atom names
        and possibly distance. Molecules can be disconnected within
        residues.
    """

    non_edges = set()

    residue_groups = _collect_residues(molecule)

    for ((mol_idx, chain, resid, resname), idxs) in residue_groups.items():
        if not allow_name:
            break
        try:
            # Try adding bonds within the residue based on atom names
            non_edges.update(_bonds_from_names(molecule, resname, idxs, force_field))
        except KeyError as error:
            # ... if that doesn't work, fall back to distance
            message = "Can't add bonds based on atom names for residue {}-{}{} because {}."
            if allow_dist:
                _bonds_from_distance(molecule, idxs, fudge=fudge)
                message += " Falling back to distance criteria."
            LOGGER.warning(message,
                           chain, resname, resid, error, force_field.name)
    # And finally, add edges based on distance, but ignore any edges that would
    # otherwise have been added by name. So only edges between residues will be
    # added, and edges involving atoms which are not known to the blocks (PTMs,
    # termini, ...)
    if allow_dist:
        _bonds_from_distance(molecule, non_edges=non_edges, fudge=fudge)

    return molecule


class MakeBonds(Processor):
    def __init__(self, allow_name=True, allow_dist=True, fudge=1):
        self.allow_name = allow_name
        self.allow_dist = allow_dist
        self.fudge = fudge

    def run_system(self, system):
        if not system.molecules:
            # No molecules means nothing to do.
            return

        # Separate molecules should remain separate molecules, even if they have
        # poorly chosen chain/resname/resid combinations. So add a mol_idx attribute
        for mol_idx, molecule in enumerate(system.molecules):
            nx.set_node_attributes(molecule, mol_idx, 'mol_idx')
        # If using multiple processors, run the system normally
        if self.nproc and self.nproc > 1:
            super().run_system(system)
        # If not using multiple processors, compose molecules into one
        else:
            system = nx.disjoint_union_all(system.molecules)
            super().run_system(system)
            # Split the system into connected components. We do want to keep residues
            # together, so make a residue graph (1 node per residue) first, and use that
            # to find the connected components.
            residue_groups = _collect_residues(system)
            residue_graph = nx.quotient_graph(system, residue_groups.values())
            molecules = []
            for node_idxs in nx.connected_components(residue_graph):
                node_idxs = set().union(*node_idxs)
                mol = Molecule(system.subgraph(node_idxs))
                molecules.append(mol)
            system.molecules = molecules

        # Restore the force field in each molecule. Setting the force field
        # at the system level propagates it to all the molecules.
        system.force_field = system.force_field
        LOGGER.info('{} molecules after guessing bonds', len(system.molecules))

    def run_molecule(self, molecule):
        return make_bonds(molecule,
                          allow_name=self.allow_name,
                          allow_dist=self.allow_dist,
                          fudge=self.fudge)