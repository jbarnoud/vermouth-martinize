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
Provides a processor that can perform a resolution transformation on a
molecule.
"""
from collections import defaultdict
from copy import deepcopy
from itertools import product, combinations

import networkx as nx

from ..molecule import Molecule, attributes_match
from .processor import Processor
from ..utils import are_all_equal, format_atom_string
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))


def build_graph_mapping_collection(from_ff, to_ff, mappings):
    """
    Function that produces a collection of :class:`vermouth.map_parser.Mapping`
    objects.
    Hereby deprecated.

    Parameters
    ----------
    from_ff: vermouth.forcefield.ForceField
        Origin force field.
    to_ff: vermouth.forcefield.ForceField
        Destination force field.
    mappings: dict[str, dict[str, vermouth.map_parser.Mapping]]
        All known mappings

    Returns
    -------
    collections.abc.Iterable
        A collection of mappings that map from `from_ff` to `to_ff`.
    """
    return mappings[from_ff.name][to_ff.name].values()


def edge_matcher(graph1, graph2, node11, node12, node21, node22):
    """
    Checks whether the resids for node11 and node12 in graph1 are the same, and
    whether that's also true for node21 and node22 in graph2.

    Parameters
    ----------
    graph1: networkx.Graph
    graph2: networkx.Graph
    node11: collections.abc.Hashable
        A node key in `graph1`.
    node12: collections.abc.Hashable
        A node key in `graph1`.
    node21: collections.abc.Hashable
        A node key in `graph2`.
    node22: collections.abc.Hashable
        A node key in `graph2`.

    Returns
    -------
    bool
    """
    node11 = graph1.nodes[node11]
    node12 = graph1.nodes[node12]
    node21 = graph2.nodes[node21]
    node22 = graph2.nodes[node22]
    return (node11.get('resid') == node12.get('resid')) ==\
           (node21.get('resid') == node22.get('resid'))


def node_matcher(node1, node2):
    """
    Checks whether nodes should be considered equal for isomorphism. Takes all
    attributes in `node2` into account, except for the attributes "atype",
    "charge", "charge_group", "resid", "replace", and "_old_atomname".

    Parameters
    ----------
    node1: dict
    node2: dict

    Returns
    -------
    bool
    """
    return attributes_match(node1, node2,
                            ignore_keys=('atype', 'charge', 'charge_group', 'mass',
                                         'resid', 'replace', '_old_atomname'))


def _old_atomname_match(node1, node2):
    """
    Adds a _name attribute to copies of the nodes, and feeds it all to
    :func:`node_matcher`
    """
    name1 = node1.get('_old_atomname', node1['atomname'])
    name2 = node2.get('_old_atomname', node2['atomname'])
    node1 = node1.copy()
    node2 = node2.copy()
    node1['_name'] = name1
    node2['_name'] = name2
    if 'order' in node2 and 'order' not in node1:
        node1['order'] = node2['order']
    del node1['atomname']
    del node2['atomname']
    return node_matcher(node1, node2)


def node_should_exist(modification, node_idx):
    """
    Returns True if the node with index `node_idx` in `modification` should
    already exist in the parent molecule.

    Parameters
    ----------
    modification: networkx.Graph
    node_idx: collections.abc.Hashable
        The key of a node in `modification`.

    Returns
    -------
    bool
        True iff the node `node_idx` in `modification` should already exist in
        the parent molecule.
    """
    return not modification.nodes[node_idx].get('PTM_atom', False)


def ptm_resname_match(mol_node, map_node):
    """
    As :func:`node_matcher`, except that empty resname and false PTM_atom
    attributes from `node2` are removed.
    """
    if 'resname' in map_node and not map_node['resname']:
        map_node = map_node.copy()
        del map_node['resname']
    if 'PTM_atom' in map_node and not map_node['PTM_atom']:
        map_node = map_node.copy()
        del map_node['PTM_atom']
    if 'modifications' in mol_node:
        map_node = map_node.copy()
        matching_mod = all(map_mod in mol_node.get('modifications', [])
                           for map_mod in map_node.pop('modifications', []))
    else:
        matching_mod = True
    is_equal = _old_atomname_match(mol_node, map_node)
    return is_equal and matching_mod


def cover(to_cover, options):
    """
    Implements a recursive backtracking algorithm to cover all elements of
    `to_cover` with the elements from `options` that have the lowest index.
    In this context "to cover" means that all items in an element of `options`
    must be in `to_cover`. Elements in `to_cover` can only be covered *once*.

    Parameters
    ----------
    to_cover: collections.abc.MutableSet
        The items that should be covered.
    options: collections.abc.Sequence[collections.abc.MutableSet]
        The elements that can be used to cover `to_cover`. All items in an
        element of `options` must be present in `to_cover` to qualify.

    Returns
    -------
    None or list
        None if no covering can be found, or the list of items from `options`
        with the lowest indices that exactly covers `to_cover`.
    """
    if not to_cover:
        return []
    for idx, option in enumerate(options):
        if all(item in to_cover for item in option):
            left_to_cover = to_cover.copy()
            for item in option:
                # Only remove the leftmost item. PS. we know for sure all items
                # in option are in left_to_cover at least once.
                left_to_cover.remove(item)
            found = cover(left_to_cover, options[idx:])
            if found is not None:
                return [option] + found
    return None


def get_mod_mappings(mappings):
    """
    Returns a dict of all known modification mappings.

    Parameters
    ----------
    mappings: collections.abc.Iterable[vermouth.map_parser.Mapping]
        All known mappings.

    Returns
    -------
    dict[tuple[str], vermouth.map_parser.Mapping]
        All mappings that describe a modification mapping.
    """
    out = {}
    for mapping in mappings:
        if mapping.type == 'modification':
            out[mapping.names] = mapping
    return out


def modification_matches(molecule, mappings):
    """
    Returns a minimal combination of modification mappings and where they
    should be applied that describes all modifications in `molecule`.

    Parameters
    ----------
    molecule: networkx.Graph
        The molecule whose modifications should be treated. Modifications are
        described by the 'modifications' node attribute.
    mappings: collections.abc.Iterable[vermouth.map_parser.Mapping]
        All known mappings.

    Returns
    -------
    list[tuple[dict, vermouth.molecule.Link, dict]]
        A list with the following items:
            Dict describing the correspondence of node keys in `molecule` to
                node keys in the modification.

            The modification.

            Dict with all reference atoms, mapping modification nodes to
                nodes in `molecule`.
    """
    modified_nodes = set()  # This will contain whole residues.
    for idx, node in molecule.nodes.items():
        if node.get('modifications', []):
            modified_nodes.add(idx)
    ptm_subgraph = molecule.subgraph(modified_nodes)
    grouped = nx.connected_components(ptm_subgraph)
    found_ptm_groups = []
    # For every modification group we would like a set with the names of the
    # involved modifications, so we can use that to figure out which mod
    # mappings should be used.
    for group in grouped:
        modifications = {
            mod.name for mol_idx in group for mod in molecule.nodes[mol_idx].get('modifications', [])
        }
        found_ptm_groups.append(modifications)
    needed_mod_mappings = set()
    known_mod_mappings = get_mod_mappings(mappings)
    for group in found_ptm_groups:
        # known_mod_mappings is a dict[tuple[str], Mapping]. We want to know
        # the minimal combination of those needed to cover all the PTMs found
        # in group. The cheapest solution is covering the names of the PTMs in
        # group with keys from known_mod_mappings. An improvement would be to
        # do the graph covering again.
        # TODO?
        covered_by = cover(list(group),
                           sorted(known_mod_mappings, key=len, reverse=True))
        if covered_by is None:
            LOGGER.warning("Can't find modification mappings for the "
                           "modifications {}. The following modification "
                           "mappings are known: {}",
                           list(group), sorted(known_mod_mappings.keys()),
                           type='unmapped-atom')
            continue
        needed_mod_mappings.update(covered_by)
    matches = []
    # Sort on the tuple[str] type names of the mappings so that mappings that
    # define most modifications at the same time get processed first
    for mod_name in sorted(needed_mod_mappings, key=len, reverse=True):
        mod_mapping = known_mod_mappings[mod_name]
        for mol_to_mod, modification, references in mod_mapping.map(molecule, node_match=ptm_resname_match):
            matches.append((mol_to_mod, modification, references))
            if not set(mol_to_mod) <= modified_nodes:
                # TODO: better message
                LOGGER.warning('Overlapping modification mappings', type='inconsistent-data')
            modified_nodes -= set(mol_to_mod)
    return matches


def apply_block_mapping(match, molecule, graph_out, mol_to_out, out_to_mol):
    """
    Performs a mapping operation for a "block". `match` is a tuple of 3
    elements that describes what nodes in `molecule` should correspond to
    a :class:`vermouth.molecule.Block` that should be added to `graph_out`, and
    any atoms that should be used a references.
    Add the required :class:`vermouth.molecule.Block` to `graph_out`, and
    updates `mol_to_out` and `out_to_mol` *in-place*.

    Parameters
    ----------
    match
    molecule: networkx.Graph
        The original molecule
    graph_out: vermouth.molecule.Molecule
        The newly created graph that describes `molecule` at a different
        resolution.
    mol_to_out: dict[collections.abc.Hashable, dict[collections.abc.Hashable, float]]
        A dict mapping nodes in `molecule` to nodes in `graph_out` with the
        associated weights.
    out_to_mol: dict[collections.abc.Hashable, dict[collections.abc.Hashable, float]]
        A dict mapping nodes in `graph_out` to nodes in `molecule` with the
        associated weights.

    Returns
    -------
    set
        A set of all overlapping nodes that were already mapped before.
    set
        A set of none-to-one mappings. I.e. nodes that were created without
        nodes mapping to them.
    dict
        A dict of reference atoms, mapping `graph_out` nodes to nodes in
        `molecule`.
    """
    mol_to_block, blocks_to, references = match
    if graph_out.nrexcl is None:
        graph_out.nrexcl = blocks_to.nrexcl
    try:
        # merge_molecule will return a dict mapping the node keys of the
        # added block to the ones in graph_out
        # FIXME: Issue #154 lives here.
        block_to_out = graph_out.merge_molecule(blocks_to)
    except ValueError:
        # This probably means the nrexcl of the block is different from the
        # others. This means the user messed up their data. Or there are
        # different forcefields in the same forcefield folder...
        LOGGER.exception('Residue(s) {} is not compatible with the others',
                         set(nx.get_node_attributes(blocks_to, 'resname').values()),
                         type='inconsistent-data')
        raise
    # overlap does not have to be a dict, since the values in block_to_out are
    # guaranteed to be unique in graph_out. So we can look them up in
    # mol_to_out
    overlap = set(mol_to_out.keys()) & set(mol_to_block.keys())
    for mol_idx in mol_to_block:
        for block_idx, weight in mol_to_block[mol_idx].items():
            out_idx = block_to_out[block_idx]
            mol_to_out[mol_idx][out_idx] = weight
            out_to_mol[out_idx][mol_idx] = weight

    none_to_one_mappings = set()
    mapped_block_idxs = {block_idx
                         for mol_idx in mol_to_block
                         for block_idx in mol_to_block[mol_idx]}
    for spawned in set(blocks_to.nodes) - mapped_block_idxs:
        # These nodes come from "nowhere", so, let's pretend they come from
        # all nodes in the block. This helps with setting attributes such
        # as 'chain'
        # "None to one" mapping - this is fine. This happens with e.g.
        # charge dummies.
        spawned = block_to_out[spawned]
        none_to_one_mappings.add(spawned)
        for mol_idx in mol_to_block:
            mol_to_out[mol_idx][spawned] = 0
            out_to_mol[spawned][mol_idx] = 0

    new_references = {block_to_out[mod_idx]: mol_idx for mod_idx, mol_idx in references.items()}
    return overlap, none_to_one_mappings, new_references


def apply_mod_mapping(match, molecule, graph_out, mol_to_out, out_to_mol):
    """
    Performs the mapping operation for a modification.

    Parameters
    ----------
    match
    molecule: networkx.Graph
        The original molecule
    graph_out: vermouth.molecule.Molecule
        The newly created graph that describes `molecule` at a different
        resolution.
    mol_to_out: dict[collections.abc.Hashable, dict[collections.abc.Hashable, float]]
        A dict mapping nodes in `molecule` to nodes in `graph_out` with the
        associated weights.
    out_to_mol: dict[collections.abc.Hashable, dict[collections.abc.Hashable, float]]
        A dict mapping nodes in `graph_out` to nodes in `molecule` with the
        associated weights.

    Returns
    -------
    dict[str, dict[tuple, vermouth.molecule.Link]]
        A dict of all modifications that have been applied by this modification
        mapping operations. Maps interaction type to involved atoms to the
        modification responsible.
    dict
        A dict of reference atoms, mapping `graph_out` nodes to nodes in
        `molecule`.
    """
    mol_to_mod, modification, references = match
    LOGGER.info('Applying modification mapping {}', modification.name, type='general')
    graph_out.citations.update(modification.citations)

    mod_to_mol = defaultdict(dict)
    for mol_idx, mod_idxs in mol_to_mod.items():
        for mod_idx in mod_idxs:
            mod_to_mol[mod_idx][mol_idx] = mol_to_mod[mol_idx][mod_idx]
    mod_to_mol = dict(mod_to_mol)
    mod_to_out = {}
    # Some nodes of modification will already exist. The question is
    # which, and which index they have in graph_out.
    # We'll store these anchors for now
    anchors = set()
    for mod_idx in modification:
        if not node_should_exist(modification, mod_idx):
            # Node does not exist yet.
            if not graph_out.nodes:
                out_idx = 0
            else:
                out_idx = max(graph_out) + 1
            mod_to_out[mod_idx] = out_idx
            graph_out.add_node(out_idx, **modification.nodes[mod_idx])
            graph_out.nodes[out_idx].update(modification.nodes[mod_idx].get('replace', {}))
        else:
            # Node should already exist
            # We need to find the out_index of this node. Since the
            # node already exists, there is at least one mol_idx in
            # mol_to_out that refers to the correct out_idx. What we do
            # is try to find those mol indices by looking at
            # mod_to_mol.
            # Find the other mol nodes that map to this bead according
            # to the mod mapping...
            mol_idxs = mod_to_mol[mod_idx]
            # ...and make the node with the correct attributes.
            out_idxs = set()
            for mol_idx in mol_idxs:
                out_idxs.update(mol_to_out.get(mol_idx, {}))
            for out_idx in out_idxs:
                out_node = graph_out.nodes[out_idx]
                if modification.nodes[mod_idx]['atomname'] == out_node['atomname']:
                    break
            else:  # No break, so no matching node found
                raise ValueError("No node found in molecule with "
                                 "atomname {}".format(modification.nodes[mod_idx]['atomname']))
            # Undefined loop variable is guarded against by the else-raise above
            anchors.add(mod_idx)
            mod_to_out[mod_idx] = out_idx  # pylint: disable=undefined-loop-variable
            graph_out.nodes[out_idx].update(modification.nodes[mod_idx].get('replace', {})) # pylint: disable=undefined-loop-variable
        graph_out.nodes[out_idx]['modifications'] = graph_out.nodes[out_idx].get('modifications', [])
        if modification not in graph_out.nodes[out_idx]['modifications']:
            graph_out.nodes[out_idx]['modifications'].append(modification)

    # FIXME Jank here to ensure the charge_group attributes look reasonable
    charge_group_start = max(graph_out.nodes[mod_to_out[idx]].get('charge_group', 1) for idx in anchors) if anchors else 1
    for charge_group, mod_idx in enumerate(modification, charge_group_start):
        out_idx = mod_to_out[mod_idx]
        if 'charge_group' not in graph_out.nodes[out_idx]:
            graph_out.nodes[out_idx]['charge_group'] = charge_group

    for mol_idx in mol_to_mod:
        for mod_idx, weight in mol_to_mod[mol_idx].items():
            out_idx = mod_to_out[mod_idx]
            if mol_idx not in mol_to_out:
                mol_to_out[mol_idx] = {}
            mol_to_out[mol_idx][out_idx] = weight
            if out_idx not in out_to_mol:
                out_to_mol[out_idx] = {}
            out_to_mol[out_idx][mol_idx] = weight

    for mod_idx, mod_jdx in modification.edges:
        out_idx = mod_to_out[mod_idx]
        out_jdx = mod_to_out[mod_jdx]
        if not graph_out.has_edge(out_idx, out_jdx):
            graph_out.add_edge(out_idx, out_jdx)

    new_references = {mod_to_out[mod_idx]: mol_idx for mod_idx, mol_idx in references.items()}

    # Apply interactions
    applied_interactions = defaultdict(lambda: defaultdict(list))
    for interaction_type, interactions in modification.interactions.items():
        for interaction in interactions:
            atoms = [mod_to_out[mod_idx] for mod_idx in interaction.atoms]
            assert len(atoms) == len(interaction.atoms)
            interaction = interaction._replace(atoms=atoms)
            applied_interactions[interaction_type][tuple(atoms)].append(modification)
            graph_out.add_or_replace_interaction(interaction_type, **interaction._asdict())

    # Some jank needed here, since modification node indices are integers
    mod_atom_name_to_out = {}
    for mod_idx in modification.nodes:
        name = modification.nodes[mod_idx]['atomname']
        mod_atom_name_to_out[name] = mod_to_out[mod_idx]
    for loglevel, entries in modification.log_entries.items():
        for entry in entries:
            graph_out.log_entries[loglevel][entry] += [mod_atom_name_to_out]
    return dict(applied_interactions), new_references


def attrs_from_node(node, attrs):
    """
    Helper function that applies a "replace" operations on the node if
    required, and then returns a dict of the attributes listed in `attrs`.

    Parameters
    ----------
    node: dict
    attrs: collections.abc.Container
        Attributes that should be in the output.

    Returns
    -------
    dict
    """
    if 'replace' in node:
        node = node.copy()
        node.update(node['replace'])
    return {attr: val for attr, val in node.items() if attr in attrs}


def do_mapping(molecule, mappings, to_ff, attribute_keep=(), attribute_must=(), attribute_stash=()):
    """
    Creates a new :class:`~vermouth.molecule.Molecule` in force field `to_ff`
    from `molecule`, based on `mappings`. It does this by doing a subgraph
    isomorphism of all blocks in `mappings` and `molecule`. Will issue warnings
    if there's atoms not contributing to the new molecule, or if there's
    overlapping blocks.
    Node attributes in the new molecule will come from the blocks constructing
    it, except for those in `attribute_keep`, which lists the attributes that
    will be kept from `molecule`.

    Parameters
    ----------
    molecule: :class:`~vermouth.molecule.Molecule`
        The molecule to transform.
    mappings: dict[str, dict[str, dict[str, tuple]]]
        ``{ff_name: {ff_name: {block_name: (mapping, weights, extra)}}}``
        A collection of mappings, as returned by e.g.
        :func:`~vermouth.map_input.read_mapping_directory`.
    to_ff: :class:`~vermouth.forcefield.ForceField`
        The force field to transform to.
    attribute_keep: :class:`~collections.abc.Iterable`
        The attributes that will always be transferred from `molecule` to the
        produced graph.
    attribute_must: :class:`~collections.abc.Iterable`
        The attributes that the nodes in the output graph *must* have. If
        they're not provided by the mappings/blocks they're taken from
        `molecule`.
    attribute_stash: tuple[str]
        The attributes that will always be transferred from the input molecule
        to the produced graph, but prefixed with _old_. Thus, they are new
        attributes and are not conflicting with already defined attributes.


    Returns
    -------
    :class:`~vermouth.molecule.Molecule`
        A new molecule, created by transforming `molecule` to `to_ff` according
        to `mappings`.
    """
    attribute_keep = tuple(attribute_keep)
    attribute_must = tuple(attribute_must)
    attribute_stash = tuple(attribute_stash)
    # Transferring the meta maybe should be a copy, or a deep copy...
    # If it breaks we look at this line.
    graph_out = Molecule(force_field=to_ff, meta=molecule.meta)
    graph_out.citations.update(molecule.citations)
    mappings = build_graph_mapping_collection(molecule.force_field, to_ff, mappings)
    block_matches = []
    for mapping in mappings:
        if mapping.type == 'block':
            block_matches.extend(mapping.map(molecule, node_match=_old_atomname_match,
                                             edge_match=edge_matcher))
    mod_matches = modification_matches(molecule, mappings)

    # Sort by lowest node key per residue. We need to do this, since
    # merge_molecule creates new resid's in order.
    block_sort_key = lambda x: min(x[0].keys())
    # Sort modifications by the lowest mapped index when all touched atoms are
    # newly created PTM atoms that do not exist yet. Otherwise, take the
    # highest index. If we don't do this we run the risk that a PTM mapped to
    # a node with a low index also changes a node of a higher residue, causing
    # all sorts of havok.
    mod_sort_key = lambda x: (max(x[0].keys())
                              if any(node_should_exist(x[1], idx)
                                     for idx in
                                     {mod_idx for mol_idx in x[0]
                                      for mod_idx in x[0][mol_idx]})
                              else
                              min(x[0].keys()))
    block_matches = sorted(block_matches, key=block_sort_key, reverse=True)
    mod_matches = sorted(mod_matches, key=mod_sort_key, reverse=True)

    # There are a few separate mapping cases to be considered:
    # One to one mapping - e.g. AA to AA, the simplest case
    # Many to one mapping - e.g. AA to CG without sharing atoms between beads
    # Many to many mapping - e.g. AA to CG *with* sharing atoms between beads
    # These three cases are covered by the normal operation, the following are
    # caught with some additional logic
    # None to one - whole block taken as origin, with weights 0
    # One to none - unmapped atoms (produces a warning)
    # One to many - e.g. CG to AA. This mostly works, but we don't know how to
    #               make sure the "many" should be connected together. Gives a
    #               warning if it's disconnected.

    mol_to_out = defaultdict(dict)
    out_to_mol = defaultdict(dict)
    overlapping_mappings = set()
    none_to_one_mappings = set()
    modified_interactions = {}
    all_references = {}
    all_matches = []
    while block_matches or mod_matches:
        # Take the match with the lowest atom id, and prefer blocks over
        # modifications
        if (not block_matches or
                (mod_matches and
                 mod_sort_key(mod_matches[-1]) < block_sort_key(block_matches[-1]))):
            match = mod_matches.pop(-1)
            applied_interactions, refs = apply_mod_mapping(match,
                                                           molecule, graph_out,
                                                           mol_to_out, out_to_mol)
            modified_interactions.update(applied_interactions)
        else:
            match = block_matches.pop(-1)
            overlap, none_to_one, refs = apply_block_mapping(match,
                                                             molecule, graph_out,
                                                             mol_to_out, out_to_mol)
            overlapping_mappings.update(overlap)
            none_to_one_mappings.update(none_to_one)
        all_matches.append(match)
        all_references.update(refs)

    # At this point, we should have created graph_out at the desired
    # resolution, *and* have the associated correspondence in mol_to_out and
    # out_to_mol.

    # Set node attributes based on what the original atoms are.
    to_remove = set()
    for out_idx in out_to_mol:
        mol_idxs = out_to_mol[out_idx].keys()

        # Time to collect the attributes to set. Start by grabbing all the
        # attributes we already have.
        node_attrs = deepcopy(graph_out.nodes[out_idx])

        # In all cases, keep track of what bead comes from where
        subgraph = molecule.subgraph(mol_idxs)
        node_attrs['graph'] = subgraph
        weights = out_to_mol[out_idx]
        node_attrs['mapping_weights'] = weights

        # Now we find attribute values from molecule.
        if out_idx in all_references:
            ref_idx = all_references[out_idx]
            mol_attrs = attrs_from_node(molecule.nodes[ref_idx], attribute_keep + attribute_stash)
            for attr, val in mol_attrs.items():
                # This is if/if rather than if/elif on purpose. It could be that
                # an attribute needs to be both stashed and kept
                if attr in attribute_stash:
                    node_attrs["_old_" + attr] = val
                if attr in attribute_keep or attr not in graph_out.nodes[out_idx]:
                    node_attrs[attr] = val
        else:
            attrs = defaultdict(list)
            attrs_not_sane = []
            for mol_idx in mol_idxs:
                mol_attrs = attrs_from_node(molecule.nodes[mol_idx], attribute_keep + attribute_stash)
                for attr, val in mol_attrs.items():
                    attrs[attr].append(val)

            for attr, vals in attrs.items():
                if not are_all_equal(vals):
                    attrs_not_sane.append(attr)
                # This is if/if rather than if/elif on purpose. It could be that
                # an attribute needs to be both stashed and kept
                if attr in attribute_stash:
                    node_attrs["_old_" + attr] = vals[0]
                if attr in attribute_keep or attr not in graph_out.nodes[out_idx]:
                    node_attrs[attr] = vals[0]

            if attrs_not_sane:
                LOGGER.warning('The attributes {} for atom {} are going to'
                               ' be garbage because the attributes of the'
                               ' constructing atoms are different.',
                               attrs_not_sane,
                               format_atom_string(graph_out.nodes[out_idx]),
                               type='inconsistent-data')
        graph_out.nodes[out_idx].update(node_attrs)
        if graph_out.nodes[out_idx].get('atomname', '') is None:
            to_remove.add(out_idx)

    # We need to add edges between residues. Within residues comes from the
    # blocks.
    for match1, match2 in combinations(all_matches, 2):
        match1 = match1[0]
        match2 = match2[0]
        edges = molecule.edges_between(match1.keys(), match2.keys())
        # TODO: Backmapping needs love here
        for mol_idx, mol_jdx in edges:
            # Subtract none_to_one_mappings, since those should not be made to
            # connect to things automatically.
            out_idxs = mol_to_out[mol_idx].keys() - none_to_one_mappings
            out_jdxs = mol_to_out[mol_jdx].keys() - none_to_one_mappings
            for out_idx, out_jdx in product(out_idxs, out_jdxs):
                if out_idx != out_jdx:
                    graph_out.add_edge(out_idx, out_jdx)

    ############################
    # Sanity check the results #
    ############################

    # "Many to one" mapping - overlapping blocks means dubious node properties
    if overlapping_mappings:
        LOGGER.warning('These atoms are covered by multiple blocks. This is a '
                       'bad idea: {}. This probably means the following output'
                       ' particles are wrong: {}.',
                       {format_atom_string(molecule.nodes[mol_idx])
                        for mol_idx in overlapping_mappings},
                       {format_atom_string(graph_out.nodes[out_idx], atomid='')
                        for mol_idx in overlapping_mappings
                        for out_idx in mol_to_out[mol_idx]},
                       type='inconsistent-data')

    # "One to many" mapping - not necessarily a problem, unless it leads to
    # missing edges
    for mol_idx in mol_to_out:
        # Subtract the none to one mapped nodes, since those don't contribute
        # and make false positives.
        out_idxs = mol_to_out[mol_idx].keys() - none_to_one_mappings
        if len(out_idxs) > 1 and not nx.is_connected(graph_out.subgraph(out_idxs)):
            # In this case there's a single input particle mapping to multiple
            # output particles. This probably means there's bonds missing
            LOGGER.warning('The input particle {} maps to multiple output '
                           'particles: {}, which are disconnected. There are '
                           'probably edges missing.',
                           format_atom_string(molecule.nodes[mol_idx]),
                           {format_atom_string(graph_out.nodes[out_idx], atomid='')
                            for out_idx in out_idxs},
                           type='inconsistent-data')

    # "One to none" mapping - this means your mapping files are incomplete
    uncovered_atoms = set(molecule.nodes.keys()) - set(mol_to_out.keys())
    if uncovered_atoms:
        uncovered_hydrogens = {idx for idx in uncovered_atoms
                               if molecule.nodes[idx].get('element', '') == 'H'}
        if uncovered_hydrogens:
            # Maybe this should be info?
            LOGGER.debug('These hydrogen atoms are not covered by a mapping.'
                         ' This is not the best idea. {}',
                         [format_atom_string(molecule.nodes[idx])
                          for idx in uncovered_hydrogens],
                         type='unmapped-atom'
                         )
        other_uncovered = uncovered_atoms - uncovered_hydrogens
        if other_uncovered:
            LOGGER.warning("These atoms are not covered by a mapping. Either"
                           " your mappings don't describe all atoms (bad idea),"
                           " or, there's no mapping available for all residues."
                           " {}",
                           [format_atom_string(molecule.nodes[idx])
                            for idx in other_uncovered],
                           type='unmapped-atom')

    # "None to one" mapping - Strictly speaking this cannot happen, but it could
    # be that output particles map to only particles that were not present in
    # the input structure. This probably causes massive issues downstream.
    # For now, flag a sane warning, and at the same time transfer missing "must"
    # attributes from nearby atoms.
    for node in graph_out:
        for attr in attribute_must:
            if attr not in graph_out.nodes[node]:
                # We can skip `node`, since if it had the attribute required we
                # wouldn't be in this mess to begin with.
                # Set a depth_limit to keep the runtime within bounds.
                close_nodes = (v for (u, v) in nx.bfs_edges(graph_out, source=node, depth_limit=3))
                for template_node in close_nodes:
                    if attr in graph_out.nodes[template_node]:
                        graph_out.nodes[node][attr] = deepcopy(graph_out.nodes[template_node][attr])
                        LOGGER.warning("Atom {} did not have a {}, so we took "
                                       "it from atom {} instead.",
                                       format_atom_string(graph_out.nodes[node]),
                                       attr,
                                       format_atom_string(graph_out.nodes[template_node]),
                                       type="inconsistent-data")
                        break
                else:  # no break, attribute not found
                    LOGGER.warning("Atom {} does not have a {}, and we couldn't "
                                   "find a nearby atom that does.",
                                   format_atom_string(graph_out.nodes[node]),
                                   attr, type="inconsistent-data")

    for interaction_type in modified_interactions:
        for atoms, modifications in modified_interactions[interaction_type].items():
            if len(modifications) != 1:
                # TODO: better message
                LOGGER.warning('Interaction set by multiple modification '
                               'mappings', type='inconsistent-data')

    graph_out.remove_nodes_from(to_remove)
    return graph_out


class DoMapping(Processor):
    """
    Processor for performing a resolution transformation from one force field to
    another.

    This processor will create new Molecules by stitching together Blocks from
    the target force field, as dictated by the available mappings.
    Fragments/atoms/residues/modifications for which no mapping is available
    will not be represented in the resulting molecule.

    The resulting molecules will have intra-block edges and interactions as
    specified in the blocks from the target force field. Inter-block edges will
    be added based on the connectivity of the original molecule, but no
    interactions will be added for those.

    Attributes
    ----------
    mappings: dict[str, dict[str, dict[str, tuple]]]
        ``{ff_name: {ff_name: {block_name: (mapping, weights, extra)}}}``
        A collection of mappings, as returned by e.g.
        :func:`~vermouth.map_input.read_mapping_directory`.
    to_ff: vermouth.forcefield.ForceField
        The force field to map to.
    delete_unknown: bool
        Not currently used
    attribute_keep: tuple[str]
        The attributes that will always be transferred from the input molecule
        to the produced graph.
    attribute_must: tuple[str]
        The attributes that the nodes in the output graph *must* have. If
        they're not provided by the mappings/blocks they're taken from
        the original molecule.
    attribute_stash: tuple[str]
        The attributes that will always be transferred from the input molecule
        to the produced graph, but prefixed with _old_.Thus they are new attributes
        and are not conflicting with already defined attributes.

    See Also
    --------
    :func:`do_mapping`
    """
    def __init__(self, mappings, to_ff, delete_unknown=False, attribute_keep=(),
                 attribute_must=(), attribute_stash=()):
        self.mappings = mappings
        self.to_ff = to_ff
        self.delete_unknown = delete_unknown
        self.attribute_keep = tuple(attribute_keep)
        self.attribute_must = tuple(attribute_must)
        self.attribute_stash = tuple(attribute_stash)
        super().__init__()

    def run_molecule(self, molecule):
        return do_mapping(
            molecule,
            mappings=self.mappings,
            to_ff=self.to_ff,
            attribute_keep=self.attribute_keep,
            attribute_must=self.attribute_must,
            attribute_stash=self.attribute_stash
        )

    def run_system(self, system):
        mols = []
        for molecule in system.molecules:
            try:
                new_molecule = self.run_molecule(molecule)
            except KeyError as err:
                if not self.delete_unknown:
                    raise err
                else:
                    raise
                    # TODO: raise a loud warning here
            else:
                if new_molecule:
                    mols.append(new_molecule)
        system.molecules = mols
        system.force_field = self.to_ff
