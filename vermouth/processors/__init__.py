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
Provides Processors, VerMoUTH's work horses.
"""


from .gro_reader import GROInput
from .make_bonds import MakeBonds
from .pdb_reader import PDBInput
from .repair_graph import RepairGraph
from .do_mapping import DoMapping
from .do_links import DoLinks
from .average_beads import DoAverageBead
from .apply_posres import ApplyPosres
from .set_molecule_meta import SetMoleculeMeta
from .locate_charge_dummies import LocateChargeDummies
from .attach_mass import AttachMass
from .apply_rubber_band import ApplyRubberBand
from .merge_chains import MergeChains
from .canonicalize_modifications import CanonicalizeModifications
from .rename_modified_residues import RenameModifiedResidues
from .tune_cystein_bridges import (
    RemoveCysteinBridgeEdges,
    AddCysteinBridgesThreshold,
)
from .add_molecule_edges import AddMoleculeEdgesAtDistance, MergeNucleicStrands
from .name_moltype import NameMolType
from .quote import Quoter
from .sort_molecule_atoms import SortMoleculeAtoms
from .merge_all_molecules import MergeAllMolecules
from .annotate_mut_mod import AnnotateMutMod
from .water_bias import ComputeWaterBias
from .annotate_idrs import AnnotateIDRs
from .rtp_polisher import RTPPolisher
