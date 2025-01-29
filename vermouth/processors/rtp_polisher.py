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
from collections import defaultdict
import networkx as nx

from ..molecule import Interaction
from .processor import Processor
from ..utils import first_alpha


def _generate_paths(graph, length):
    if length < 1:
        return
    elif length == 1:
        yield from graph.edges
        yield from (e[::-1] for e in graph.edges)
        return
    for p in _generate_paths(graph, length-1):
        for neighbour in set(graph[p[-1]]) - set(p):  # Or should this be `graph[p[-1]] - p[-2]`?
            yield p + (neighbour,)
    return


def _keep_dihedral(dihedral, known_dihedral_centers, known_improper_centers, all_dihedrals, remove_dihedrals):
    # https://manual.gromacs.org/current/reference-manual/file-formats.html#rtp
    # gmx pdb2gmx automatically generates one proper dihedral for every
    # rotatable bond, preferably on heavy atoms. When the [dihedrals] field is
    # used, no other dihedrals will be generated for the bonds corresponding to
    # the specified dihedrals. It is possible to put more than one dihedral on
    # a rotatable bond.
    # Column 5 : This controls the generation of dihedrals from the bonding.
    #            All possible dihedrals are generated automatically. A value of
    #            1 here means that all these are retained. A value of
    #            0 here requires generated dihedrals be removed if
    #              * there are any dihedrals on the same central atoms
    #                specified in the residue topology, or
    #              * there are other identical generated dihedrals
    #                sharing the same central atoms, or
    #              * there are other generated dihedrals sharing the
    #                same central bond that have fewer hydrogen atoms
    # Column 8: Remove proper dihedrals if centered on the same bond
    #           as an improper dihedral
    # https://gitlab.com/gromacs/gromacs/-/blob/main/src/gromacs/gmxpreprocess/gen_ad.cpp#L249
    center = tuple(dihedral[1:3])
    if (not all_dihedrals) and (center in known_dihedral_centers or center[::-1] in known_dihedral_centers):
        return False
    if remove_dihedrals and center in known_improper_centers:
        return False
    return True


def add_angles(mol, *, bond_type=5):
    known_angles = {tuple(i.atoms) for i in mol.get_interaction("angles")}
    for p in _generate_paths(mol, 2):
        if p not in known_angles and p[::-1] not in known_angles:
            mol.add_interaction('angles', atoms=p, parameters=[bond_type], meta={'comment': 'implicit angle'})
            known_angles.add(p)


def add_dihedrals_and_pairs(mol, *, bond_type=2, all_dihedrals=True,
                            HH14_pair=False, remove_dihedrals=True):
    # Generate missing dihedrals and pair interactions
    # As pdb2gmx generates all the possible dihedral angles by default,
    # RTP files are written assuming they will be generated. A RTP file
    # have some control over these dihedral angles through the bondedtypes
    # section.

    explicit_dihedral_centers = {frozenset(dih.atoms[1:3]) for dih in mol.interactions.get('dihedrals', [])}
    implicit_dihedrals = defaultdict(set)
    explicit_improper_centers = {frozenset(dih.atoms[1:3]) for dih in mol.interactions.get('impropers', [])}
    distances = dict(nx.all_pairs_shortest_path_length(mol, cutoff=3))

    known_pairs = {frozenset(inter.atoms) for inter in mol.interactions.get('pairs', [])}
    hydrogens = {n for n in mol if mol.nodes[n].get('element', first_alpha(mol.nodes[n]['atomname'])) == 'H'}
    for path in _generate_paths(mol, 3):
        center = frozenset(path[1:3])
        # https://manual.gromacs.org/current/reference-manual/file-formats.html#rtp
        # gmx pdb2gmx automatically generates one proper dihedral for every
        # rotatable bond, preferably on heavy atoms. When the [dihedrals] field is
        # used, no other dihedrals will be generated for the bonds corresponding to
        # the specified dihedrals. It is possible to put more than one dihedral on
        # a rotatable bond.
        # Column 5 : This controls the generation of dihedrals from the bonding.
        #            All possible dihedrals are generated automatically. A value of
        #            1 here means that all these are retained. A value of
        #            0 here requires generated dihedrals be removed if
        #              * there are any dihedrals on the same central atoms
        #                specified in the residue topology, or
        #              * there are other identical generated dihedrals
        #                sharing the same central atoms, or
        #              * there are other generated dihedrals sharing the
        #                same central bond that have fewer hydrogen atoms
        # Column 8: Remove proper dihedrals if centered on the same bond
        #           as an improper dihedral
        # https://gitlab.com/gromacs/gromacs/-/blob/main/src/gromacs/gmxpreprocess/gen_ad.cpp#L249
        if (not (center not in explicit_dihedral_centers and remove_dihedrals and center in explicit_improper_centers)
                and path not in implicit_dihedrals[center] and path[::-1] not in implicit_dihedrals[center]):
            implicit_dihedrals[center].add(path)

        pair = frozenset({path[0], path[-1]})
        if (HH14_pair or not (pair <= hydrogens)) and pair not in known_pairs and distances[path[0]][path[-1]] == 3:
            # Pair interactions are generated for all pairs of atoms which are separated
            # by 3 bonds (except pairs of hydrogens).
            # TODO: correct for specified exclusions
            mol.add_interaction('pairs', atoms=tuple(sorted(pair)), parameters=[1])
            known_pairs.add(pair)

    for center in implicit_dihedrals:
        if all_dihedrals:
            # Just add everything
            # TODO: Also sort the dihedrals by index.
            # See src/gromacs/gmxpreprocess/gen_add.cpp::dcomp in the
            # Gromacs source code (see version 2016.3 for instance).

            dihedrals = [Interaction(atoms=p, parameters=[bond_type], meta={'comment': 'implicit dihedral'})
                         for p in implicit_dihedrals[center]]
            mol.interactions['dihedrals'] = mol.interactions.get('dihedrals', []) + dihedrals
        else:
            # Find the dihedral with the least amount of hydrogens
            best_dihedral = min(implicit_dihedrals[center],
                                key=lambda p: sum(mol.nodes[idx].get('atomname', '') == 'H' for idx in p))
            mol.add_interaction('dihedrals', atoms=best_dihedral, parameters=[bond_type], meta={'comment': 'implicit dihedral'})


class RTPPolisher(Processor):
    def run_molecule(self, molecule):
        bondedtypes = molecule.force_field.variables['bondedtypes']
        # bond_type = bondedtypes.bonds
        angle_type = bondedtypes.angles
        dihedral_type = bondedtypes.dihedrals
        # improper_type = bondedtypes.impropers
        all_dihedrals = bondedtypes.all_dihedrals
        HH14_pair = bondedtypes.HH14
        remove_dihedrals = bondedtypes.remove_dih
        add_angles(molecule, bond_type=angle_type)
        add_dihedrals_and_pairs(molecule, bond_type=dihedral_type, all_dihedrals=all_dihedrals, HH14_pair=HH14_pair, remove_dihedrals=remove_dihedrals)
        return molecule
