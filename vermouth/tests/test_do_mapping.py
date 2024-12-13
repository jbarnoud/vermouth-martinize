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
Tests for the DoMapping processor
"""

from collections import defaultdict
import pprint

import networkx as nx
import pytest

from vermouth.processors.do_mapping import do_mapping, cover, modification_matches, apply_mod_mapping
import vermouth.forcefield
from vermouth.molecule import Molecule, Block, Link, Interaction
from vermouth.map_parser import Mapping
from vermouth.tests.helper_functions import equal_graphs

# Pylint does *not* like pytest fixtures
# pylint: disable=redefined-outer-name


FF_MARTINI = vermouth.forcefield.get_native_force_field(name='martini22')
FF_UNIVERSAL = vermouth.forcefield.get_native_force_field(name='charmm')

AA_MOL = Molecule(force_field=FF_UNIVERSAL)
AA_MOL.add_nodes_from((
    (0, {'resid': 1, 'resname': 'IPO', 'atomname': 'C1', 'chain': 'A', 'element': 'C'}),
    (1, {'resid': 1, 'resname': 'IPO', 'atomname': 'C2', 'chain': 'A', 'element': 'C'}),
    (2, {'resid': 1, 'resname': 'IPO', 'atomname': 'C3', 'chain': 'A', 'element': 'C'}),
    (3, {'resid': 2, 'resname': 'IPO', 'atomname': 'C1', 'chain': 'A', 'element': 'C'}),
    (4, {'resid': 2, 'resname': 'IPO', 'atomname': 'C2', 'chain': 'A', 'element': 'C'}),
    (5, {'resid': 2, 'resname': 'IPO', 'atomname': 'C3', 'chain': 'A', 'element': 'C'}),
    (6, {'resid': 3, 'resname': 'IPO', 'atomname': 'C1', 'chain': 'A', 'element': 'C'}),
    (7, {'resid': 3, 'resname': 'IPO', 'atomname': 'C2', 'chain': 'A', 'element': 'C'}),
    (8, {'resid': 3, 'resname': 'IPO', 'atomname': 'C3', 'chain': 'A', 'element': 'C'}),
))
AA_MOL.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (4, 6), (6, 7), (7, 8)])
block_aa = Block(force_field=FF_UNIVERSAL)
block_aa.name = 'IPO'
block_aa.add_nodes_from((
    ('C1', {'resid': 1, 'resname': 'IPO', 'atomname': 'C1'}),
    ('C2', {'resid': 1, 'resname': 'IPO', 'atomname': 'C2'}),
    ('C3', {'resid': 1, 'resname': 'IPO', 'atomname': 'C3'}),
))
block_aa.add_edges_from([('C1', 'C2'), ('C2', 'C3')])

block_cg = Block(force_field=FF_MARTINI)
block_cg.name = 'IPO'
block_cg.add_nodes_from((('B1', {'resid': 1, 'resname': 'IPO', 'atomname': 'B1'}),))

FF_MARTINI.blocks['IPO'] = block_cg
FF_UNIVERSAL.blocks['IPO'] = block_aa


block_aa = Block(force_field=FF_UNIVERSAL)
block_aa.name = 'IPO_large'
block_aa.add_nodes_from((
    ('C1', {'resid': 1, 'resname': 'IPO', 'atomname': 'C1'}),
    ('C2', {'resid': 1, 'resname': 'IPO', 'atomname': 'C2'}),
    ('C3', {'resid': 1, 'resname': 'IPO', 'atomname': 'C3'}),
    ('C4', {'resid': 2, 'resname': 'IPO', 'atomname': 'C1'}),
    ('C5', {'resid': 2, 'resname': 'IPO', 'atomname': 'C2'}),
    ('C6', {'resid': 2, 'resname': 'IPO', 'atomname': 'C3'}),
    ('C7', {'resid': 3, 'resname': 'IPO', 'atomname': 'C1'}),
    ('C8', {'resid': 3, 'resname': 'IPO', 'atomname': 'C2'}),
    ('C9', {'resid': 3, 'resname': 'IPO', 'atomname': 'C3'}),
))
block_aa.add_edges_from([('C1', 'C2'), ('C2', 'C3'), ('C2', 'C4'),
                         ('C4', 'C5'), ('C5', 'C6'), ('C5', 'C7'),
                         ('C7', 'C8'), ('C8', 'C9'),
                         ])

block_cg = Block(force_field=FF_MARTINI)
block_cg.name = 'IPO_large'
block_cg.add_nodes_from((('B1', {'resid': 1, 'resname': 'IPO_large', 'atomname': 'B1'}),
                         ('B2', {'resid': 2, 'resname': 'IPO_large', 'atomname': 'B1'}),
                         ('B3', {'resid': 3, 'resname': 'IPO_large', 'atomname': 'B1'}),))
block_cg.add_edges_from([('B1', 'B2'), ('B2', 'B3')])

FF_MARTINI.blocks['IPO_large'] = block_cg
FF_UNIVERSAL.blocks['IPO_large'] = block_aa


def test_no_residue_crossing():
    """
    Make sure we don't cross residue boundaries
    """
    mapping = {'C1': {'B1': 1}, 'C2': {'B1': 1}, 'C3': {'B1': 1}}
    extra = ()
    mappings = {'charmm': {'martini22': {'IPO': Mapping(FF_UNIVERSAL.blocks['IPO'],
                                                           FF_MARTINI.blocks['IPO'],
                                                           mapping=mapping,
                                                           references={},
                                                           ff_from=FF_UNIVERSAL,
                                                           ff_to=FF_MARTINI,
                                                           names=('IPO',),
                                                           extra=extra)}}}

    cg = do_mapping(AA_MOL, mappings, FF_MARTINI, attribute_keep=['chain'])

    expected = Molecule(force_field=FF_MARTINI)
    expected.add_nodes_from((
        (0, {'resid': 1, 'resname': 'IPO', 'atomname': 'B1', 'chain': 'A', 'charge_group': 1}),
        (1, {'resid': 2, 'resname': 'IPO', 'atomname': 'B1', 'chain': 'A', 'charge_group': 2}),
        (2, {'resid': 3, 'resname': 'IPO', 'atomname': 'B1', 'chain': 'A', 'charge_group': 3}),
    ))
    expected.add_edges_from(([0, 1], [1, 2]))

    print(cg.nodes(data=True))
    print(cg.edges())
    print('-'*80)
    print(expected.nodes(data=True))
    print(expected.edges())
    assert equal_graphs(cg, expected)


def test_residue_crossing():
    '''
    Make sure we do cross residue boundaries and can rename residues
    '''
    mapping = {'C1': {'B1': 1}, 'C2': {'B1': 1}, 'C3': {'B1': 1},
               'C4': {'B2': 1}, 'C5': {'B2': 1}, 'C6': {'B2': 1},
               'C7': {'B3': 1}, 'C8': {'B3': 1}, 'C9': {'B3': 1},
               }
    extra = ()
    mappings = {
        'charmm': {
            'martini22': {
                'IPO_large': Mapping(FF_UNIVERSAL.blocks['IPO_large'],
                                     FF_MARTINI.blocks['IPO_large'],
                                     mapping=mapping,
                                     references={},
                                     ff_from=FF_UNIVERSAL,
                                     ff_to=FF_MARTINI,
                                     names=('IPO', 'IPO', 'IPO'),
                                     extra=extra)
            }
        }
    }

    cg = do_mapping(AA_MOL, mappings, FF_MARTINI, attribute_keep=('chain',))

    expected = Molecule()
    expected.add_nodes_from((
        (0, {'resid': 1, 'resname': 'IPO_large', 'atomname': 'B1', 'chain': 'A', 'charge_group': 1}),
        (1, {'resid': 2, 'resname': 'IPO_large', 'atomname': 'B1', 'chain': 'A', 'charge_group': 1}),
        (2, {'resid': 3, 'resname': 'IPO_large', 'atomname': 'B1', 'chain': 'A', 'charge_group': 1}),
    ))
    expected.add_edges_from(([0, 1], [1, 2]))

    print(cg.nodes(data=True))
    print(cg.edges())
    print('-'*80)
    print(expected.nodes(data=True))
    print(expected.edges())
    assert equal_graphs(cg, expected)


def _map_weights(mapping):
    """
    Get the weights associated with a mapping
    """
    inv_map = defaultdict(list)
    for from_, tos in mapping.items():
        for to in tos:
            inv_map[to].append(from_)
    weights = {}
    for bd, atoms in inv_map.items():
        weights[bd] = {atom: 1 for atom in atoms}
    return weights


def test_peptide():
    """
    Test multiple cg beads in residue
    """
    gly = {'C': {'BB': 1}, 'N': {'BB': 1}, 'O': {'BB': 1}, 'CA': {'BB': 1}}
    ile = {'C': {'BB': 1}, 'N': {'BB': 1}, 'O': {'BB': 1}, 'CA': {'BB': 1},
           'CB': {'SC1': 1}, 'CG1': {'SC1': 1}, 'CG2': {'SC1': 1}, 'CD': {'SC1': 1}}
    leu = {'C': {'BB': 1}, 'N': {'BB': 1}, 'O': {'BB': 1}, 'CA': {'BB': 1},
           'CB': {'SC1': 1}, 'CG': {'SC1': 1}, 'CD1': {'SC1': 1}, 'CD2': {'SC1': 1}}
    extra = ()

    mappings = {'charmm': {'martini22': {}}}
    mappings['charmm']['martini22']['GLY'] = Mapping(FF_UNIVERSAL.blocks['GLY'],
                                                        FF_MARTINI.blocks['GLY'],
                                                        mapping=gly,
                                                        references={},
                                                        ff_from=FF_UNIVERSAL,
                                                        ff_to=FF_MARTINI,
                                                        names=('GLY',),
                                                        extra=extra)
    mappings['charmm']['martini22']['ILE'] = Mapping(FF_UNIVERSAL.blocks['ILE'],
                                                        FF_MARTINI.blocks['ILE'],
                                                        mapping=ile,
                                                        references={},
                                                        ff_from=FF_UNIVERSAL,
                                                        ff_to=FF_MARTINI,
                                                        names=('ILE',),
                                                        extra=extra)
    mappings['charmm']['martini22']['LEU'] = Mapping(FF_UNIVERSAL.blocks['LEU'],
                                                        FF_MARTINI.blocks['LEU'],
                                                        mapping=leu,
                                                        references={},
                                                        ff_from=FF_UNIVERSAL,
                                                        ff_to=FF_MARTINI,
                                                        names=('LEU',),
                                                        extra=extra)

    peptide = Molecule(force_field=FF_UNIVERSAL)
    aa = FF_UNIVERSAL.blocks['GLY'].to_molecule()
    peptide.merge_molecule(aa)

    aa = FF_UNIVERSAL.blocks['ILE'].to_molecule()
    peptide.merge_molecule(aa)

    aa = FF_UNIVERSAL.blocks['LEU'].to_molecule()
    peptide.merge_molecule(aa)

    peptide.add_edge(list(peptide.find_atoms(atomname='N', resid=1))[0],
                     list(peptide.find_atoms(atomname='C', resid=2))[0])
    peptide.add_edge(list(peptide.find_atoms(atomname='N', resid=2))[0],
                     list(peptide.find_atoms(atomname='C', resid=3))[0])

    for node in peptide:
        peptide.nodes[node]['atomid'] = node + 1
        peptide.nodes[node]['chain'] = ''
        # create a gap in resid pattern
        if peptide.nodes[node]['resname'] == 'ILE':
            peptide.nodes[node]['resid'] = 5

    # make sure the old resids with the gap are stashed
    cg = do_mapping(peptide, mappings, FF_MARTINI, attribute_keep=('chain',),
                    attribute_stash=('resid',))

    expected = Molecule(force_field=FF_MARTINI)
    expected.add_nodes_from({1: {'atomname': 'BB',
                                 'atype': 'P5',
                                 'chain': '',
                                 'charge': 0.0,
                                 'charge_group': 1,
                                 'resid': 1,
                                 '_old_resid': 1,
                                 'resname': 'GLY'},
                             2: {'atomname': 'BB',
                                 'atype': 'P5',
                                 'chain': '',
                                 'charge': 0.0,
                                 'charge_group': 2,
                                 'resid': 2,
                                 '_old_resid': 5,
                                 'resname': 'ILE'},
                             3: {'atomname': 'SC1',
                                 'atype': 'AC1',
                                 'chain': '',
                                 'charge': 0.0,
                                 'charge_group': 3,
                                 'resid': 2,
                                 '_old_resid': 5,
                                 'resname': 'ILE'},
                             4: {'atomname': 'BB',
                                 'atype': 'P5',
                                 'chain': '',
                                 'charge': 0.0,
                                 'charge_group': 4,
                                 'resid': 3,
                                 '_old_resid': 3,
                                 'resname': 'LEU'},
                             5: {'atomname': 'SC1',
                                 'atype': 'AC1',
                                 'chain': '',
                                 'charge': 0.0,
                                 'charge_group': 5,
                                 'resid': 3,
                                 '_old_resid': 3,
                                 'resname': 'LEU'}}.items()
                            )
    expected.add_edges_from([(1, 2), (2, 3), (2, 4), (4, 5)])

    for node in expected:
        expected.nodes[node]['atomid'] = node + 1
    print(cg.nodes(data=True))
    print(cg.edges())
    print('-'*80)
    print(expected.nodes(data=True))
    print(expected.edges())

    assert equal_graphs(cg, expected)


@pytest.mark.parametrize('to_cover, options, expected', (
    ([], [], []),
    ([], [[1, 2], [3, 4]], []),
    ([1, 2, 3], [[1, 2, 3]], [[1, 2, 3]]),
    ([1, 2, 3], [[1], [2], [3]], [[1], [2], [3]]),
    ([1, 2, 3], [[1, 2, 3], [4]], [[1, 2, 3]]),
    ([1, 2, 3], [[1], [2], [3], [4]], [[1], [2], [3]]),
    ([1, 2, 3], [[4], [1, 2, 3]], [[1, 2, 3]]),
    ([1, 2, 3], [[4], [1], [2], [3], [5]], [[1], [2], [3]]),
    ([1, 2, 3], [[1, 3], [1], [2], [3], [4]], [[1, 3], [2]]),
    ([1, 2, 3], [[1, 3, 4], [1], [2], [3], [4]], [[1], [2], [3]]),
    ([1, 2, 3], [[1, 3, 4], [1], [2], [3], [1, 2, 3]], [[1], [2], [3]]),
    ([1, 2, 3], [], None),
    ([1, 2, 3], [[1, 2]], None),
    ([1, 2, 3], [[1, 2], [2, 3]], None),
    ([1, 2, 3], [[1, 2], [3, 4]], None),
    ([1, 2, 3], [[1, 2], [4]], None),
))
def test_cover(to_cover, options, expected):
    """
    Test the cover function
    """
    output = cover(to_cover, options)
    assert output == expected


@pytest.fixture
def modifications():
    """
    Provides modifications
    """
    mods = {}
    mod_a = Link(force_field=FF_UNIVERSAL, name='mA')
    mod_a.add_node('mA', atomname='mA', PTM_atom=True, modifications=[mod_a])
    mods['mA'] = mod_a

    mod_c = Link(force_field=FF_UNIVERSAL, name='mC')
    mod_c.add_node('mC', atomname='mC', PTM_atom=True, modifications=[mod_c])
    mods['mC'] = mod_c

    mod_d = Link(force_field=FF_UNIVERSAL, name='mD')
    mod_d.add_node('mD', atomname='mD', PTM_atom=True, modifications=[mod_d])
    mods['mD'] = mod_d

    mod_fg = Link(force_field=FF_UNIVERSAL, name='mFG')
    mod_fg.add_node('mF', atomname='mF', PTM_atom=True, modifications=[mod_fg])
    mod_fg.add_node('mG', atomname='mG', PTM_atom=True, modifications=[mod_fg])
    mod_fg.add_edge('mF', 'mG')
    mod_fg.add_interaction('bond', ['mF', 'mG'], (3, 4))
    mods['mFG'] = mod_fg

    mod_i = Link(force_field=FF_UNIVERSAL, name='mI')
    mod_i.add_node('mI', atomname='mI', PTM_atom=True, modifications=[mod_i])
    mods['mI'] = mod_i
    mod_i2 = Link(force_field=FF_UNIVERSAL, name='mI2')
    mod_i2.add_node('mI2', atomname='mI2', PTM_atom=True, modifications=[mod_i2])
    mods['mI2'] = mod_i2

    mod_j = Link(force_field=FF_UNIVERSAL, name='mJ')
    mod_j.add_node('mJ', atomname='mJ', PTM_atom=True, modifications=[mod_j])
    mods['mJ'] = mod_j
    mod_j2 = Link(force_field=FF_UNIVERSAL, name='mJ2')
    mod_j2.add_node('mJ2', atomname='mJ2', PTM_atom=True, modifications=[mod_j2])
    mods['mJ2'] = mod_j2
    mod = Link(name=('mJ', 'mJ2'), force_field=FF_UNIVERSAL)
    mod.add_nodes_from((['mJ', {'atomname': 'mJ', 'PTM_atom': True, 'modifications': [mods['mJ']]}],
                        ['mJ2', {'atomname': 'mJ2', 'PTM_atom': True, 'modifications': [mods['mJ2']]}],
                        ['J', {'atomname': 'J', 'PTM_atom': False}]))
    mod.add_edges_from([('J', 'mJ'), ('J', 'mJ2')])
    mods[('mJ', 'mJ2')] = mod
    return mods


@pytest.fixture
def modified_molecule(modifications):
    """
    Provides a molecule with modifications
    """
    mol = Molecule(force_field=FF_UNIVERSAL)
    mol.add_nodes_from(enumerate((
        # Lone PTM
        {'atomname': 'A', 'resid': 1, 'modifications': [modifications['mA']]},
        {'atomname': 'mA', 'resid': 1, 'PTM_atom': True, 'modifications': [modifications['mA']]},
        {'atomname': 'B', 'resid': 2},  # Spacer
        # Two PTMs on neighbouring residues
        {'atomname': 'C', 'resid': 3, 'modifications': [modifications['mC']]},
        {'atomname': 'mC', 'resid': 3, 'PTM_atom': True, 'modifications': [modifications['mC']]},
        {'atomname': 'D', 'resid': 4, 'modifications': [modifications['mD']]},
        {'atomname': 'mD', 'resid': 4, 'PTM_atom': True, 'modifications': [modifications['mD']]},
        {'atomname': 'E', 'resid': 5},  # Spacer
        # Bridging PTMs
        {'atomname': 'F', 'resid': 6, 'modifications': [modifications['mFG']]},
        {'atomname': 'mF', 'resid': 6, 'PTM_atom': True, 'modifications': [modifications['mFG']]},
        {'atomname': 'G', 'resid': 7, 'modifications': [modifications['mFG']]},
        {'atomname': 'mG', 'resid': 7, 'PTM_atom': True, 'modifications': [modifications['mFG']]},
        {'atomname': 'H', 'resid': 8},  # Spacer
        # Two PTMs for one residue
        {'atomname': 'I', 'resid': 9, 'modifications': [modifications['mI'], modifications['mI2']]},
        {'atomname': 'mI', 'resid': 9, 'PTM_atom': True, 'modifications': [modifications['mI']]},
        {'atomname': 'mI2', 'resid': 9, 'PTM_atom': True, 'modifications': [modifications['mI2']]},
        # Two PTMs for one residue, but a single mod mapping
        {'atomname': 'J', 'resid': 10, 'modifications': [modifications['mJ'], modifications['mJ2']]},
        {'atomname': 'mJ', 'resid': 10, 'PTM_atom': True, 'modifications': [modifications['mJ']]},
        {'atomname': 'mJ2', 'resid': 10, 'PTM_atom': True, 'modifications': [modifications['mJ2']]}
    )))
    mol.add_edges_from((
        (0, 1), (0, 2),  # A
        (2, 3),  # B
        (3, 4), (3, 5),  # C
        (5, 6), (5, 7),  # D
        (7, 8),  # E
        (8, 9), (8, 10),  # F
        (9, 11),  # Bridge between mF and mG
        (10, 11), (10, 12),  # G
        (12, 13),  # H
        (13, 14), (13, 15), (13, 16),  # I
        (16, 17), (16, 18)  # J
    ))
    return mol


def test_mod_matches(modified_molecule, modifications):
    """
    Test modification matches
    """
    mappings = []
    for name in 'ABCDEFGHIJ':
        block = Block(force_field=FF_UNIVERSAL, name=name)
        block.add_node(name, atomname=name)
        mappings.append(Mapping(block, block, mapping={name: {name: 1}},
                                references={}, ff_from=FF_UNIVERSAL,
                                ff_to=FF_UNIVERSAL, names=(name,)))
    for mod in modifications.values():
        name = (mod.name,) if isinstance(mod.name, str) else mod.name
        mappings.append(Mapping(mod, mod, mapping={idx: {idx: 1} for idx in mod},
                                references={}, ff_from=FF_UNIVERSAL,
                                ff_to=FF_UNIVERSAL, names=name, type='modification'))

    print([m.names for m in mappings])
    found = list(modification_matches(modified_molecule, mappings))
    expected = [
        ({1: {'mA': 1}}, modifications['mA'], {}),
        ({4: {'mC': 1}}, modifications['mC'], {}),
        ({6: {'mD': 1}}, modifications['mD'], {}),
        ({9: {'mF': 1}, 11: {'mG': 1}}, modifications['mFG'], {}),
        ({14: {'mI': 1}}, modifications['mI'], {}),
        ({15: {'mI2': 1}}, modifications['mI2'], {}),
        ({16: {'J': 1}, 17: {'mJ': 1}, 18: {'mJ2': 1}}, modifications['mJ', 'mJ2'], {}),
    ]
    pprint.pprint(found)
    pprint.pprint(expected)
    print([e[1].name for e in expected])
    print([e[1].name for e in found])
    assert len(expected) == len(found)
    e_matches, e_mods, e_refs = zip(*expected)
    f_matches, f_mods, f_refs = zip(*found)

    for f_match in f_matches:
        assert f_match in e_matches
    for f_ref in f_refs:
        assert f_ref in e_refs
    e_mod_names = [(mod.name,) if isinstance(mod.name, str) else mod.name for mod in e_mods]
    for f_mod in f_mods:
        assert f_mod.name in e_mod_names


def test_apply_mod_mapping(modified_molecule, modifications):
    """
    Test apply_mod_mapping
    """
    graph_out = Molecule(force_field=FF_UNIVERSAL)
    graph_out.add_nodes_from([
        (0, {'atomname': 'A', 'resid': 1, 'charge_group': 1})
    ])
    mol_to_out = {0: {0: 1}}
    out_to_mol = {0: {0: 1}}
    match = ({1: {'mA': 1}}, modifications['mA'], {})

    out = apply_mod_mapping(match, modified_molecule, graph_out, mol_to_out, out_to_mol)
    print(mol_to_out)
    print(out_to_mol)
    print(graph_out.nodes[1])
    print(modifications['mA'].nodes['mA'])
    for key in modifications['mA'].nodes['mA']:
        assert graph_out.nodes[1][key] == modifications['mA'].nodes['mA'][key]
    assert out == ({}, {})
    assert mol_to_out == {0: {0: 1}, 1: {1: 1}}
    assert out_to_mol == {0: {0: 1}, 1: {1: 1}}

    graph_out.add_node(2, atomname='J', resid=2, charge_group=2)
    mol_to_out[16] = {2: 1}
    out_to_mol[2] = {16: 1}

    out = apply_mod_mapping((
        {16: {'J': 1}, 17: {'mJ': 1}, 18: {'mJ2': 1}},
        modifications['mJ', 'mJ2'], {}
    ), modified_molecule, graph_out, mol_to_out, out_to_mol)
    print(mol_to_out)
    print(out_to_mol)
    assert out == ({}, {})
    assert mol_to_out == {0: {0: 1}, 1: {1: 1}, 16: {2: 1}, 17: {3: 1}, 18: {4: 1}}
    assert out_to_mol == {0: {0: 1}, 1: {1: 1}, 2: {16: 1}, 3: {17: 1}, 4: {18: 1}}


def test_do_mapping_mods(modified_molecule, modifications):
    """
    Test do_mapping on a molecule with modifications
    """
    mappings = {}
    for name in 'ABCDEFGHIJ':
        block = Block(force_field=FF_UNIVERSAL, name=name)
        block.add_node(name, atomname=name, resid=1)
        mappings[name] = Mapping(block, block, mapping={name: {name: 1}},
                                 references={}, ff_from=FF_UNIVERSAL,
                                 ff_to=FF_UNIVERSAL, names=(name,))
    for mod in modifications.values():
        name = (mod.name,) if isinstance(mod.name, str) else mod.name
        mappings[name] = Mapping(mod, mod, mapping={idx: {idx: 1} for idx in mod},
                                 references={}, ff_from=FF_UNIVERSAL,
                                 ff_to=FF_UNIVERSAL, names=name, type='modification')
    mappings = {'charmm': {'charmm': mappings}}

    out = do_mapping(modified_molecule, mappings, FF_UNIVERSAL,
                     attribute_keep=('chain', 'resid'))
    pprint.pprint(list(out.nodes(data=True)))
    pprint.pprint(list(out.edges))

    expected = modified_molecule.copy()
    for node in expected.nodes:
        expected.nodes[node]['mapping_weights'] = {node: 1}
        expected.nodes[node]['graph'] = expected.subgraph([node])

    expected = nx.relabel_nodes(expected, {idx: idx+1 for idx in expected})
    pprint.pprint(list(expected.nodes(data='atomname')))
    pprint.pprint(list(expected.edges))

    assert equal_graphs(expected, out, node_attrs=['atomname', 'resid', 'mapping_weights'])
    assert out.interactions['bond'] == [Interaction(atoms=(10, 11), parameters=(3, 4), meta={})]
