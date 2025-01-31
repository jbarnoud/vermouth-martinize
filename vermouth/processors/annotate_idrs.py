#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""
Provides processors that can add and remove IDR specific bonds
"""

from .processor import Processor
from ..rcsu.go_utils import _in_resid_region
from ..log_helpers import StyleAdapter, get_logger
LOGGER = StyleAdapter(get_logger(__name__))


def annotate_disorder(molecule, id_regions, annotation="cgidr"):
    """
    Annotate the disordered regions of the molecule

    molecule: :class:`vermouth.molecule.Molecule`
            the molecule
    idr_regions: list
        list of tuples defining the resids of the idrs in the molecule
    annotation: str
        name of the annotation in the node
    """

    for key, node in molecule.nodes.items():
        _old_resid = node['_old_resid']
        if _in_resid_region(_old_resid, id_regions):
            molecule.nodes[key][annotation] = True
            if "cgsecstruct" in molecule.nodes[key]:
                if molecule.nodes[key]["cgsecstruct"] != "C":
                    molecule.nodes[key]["cgsecstruct"] = "C"
                    molecule.meta['modified_cgsecstruct'] = True
        else:
            molecule.nodes[key][annotation] = False

class AnnotateIDRs(Processor):
    """
    Processor to annotate intrinsically disordered regions of a molecule.

    This processor is designed primarily for the work described in the reference
    M3_GO, but is generally applicable for such circumstances where extra
    addition/removals are necessary.

    """

    def __init__(self, id_regions=None):
        """
        Parameters
        ----------
        id_regions:
            regions defining the IDRs
        """
        self.id_regions = id_regions

    def run_molecule(self, molecule):
        """
        Assign disordered regions for a single molecule
        """

        annotate_disorder(molecule, self.id_regions)

        return molecule

    def run_system(self, system):
        """
        Assign the water bias of the Go model to file. Biasing
        is always molecule specific i.e. no two different
        vermouth molecules can have the same bias.

        Parameters
        ----------
        system: :class:`vermouth.system.System`
        """
        if not self.id_regions:
            return system
        LOGGER.info("Annotating disordered regions.", type="step")
        super().run_system(system)
