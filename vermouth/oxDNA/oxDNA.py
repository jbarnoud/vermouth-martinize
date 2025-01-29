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
Provides functionality to read and write oxDNA files.
"""

from itertools import chain

import numpy as np

from ..molecule import Molecule


def read_oxDNA(file_name, exclude=()):
    """
    Parse a oxDNA file to create a molecule.

    Parameters
    ----------
    filename: str
        The file to read.
    exclude: collections.abc.Container[str]
        Atoms that have one of these residue names will not be included.

    Returns
    -------
    vermouth.molecule.Molecule
        The parsed molecules. Will not contain edges.
    """
    molecule = Molecule()

    idx = 0
    with open(str(file_name)) as oxDNA:
        # Skip header
        for _ in range(3):
            next(oxDNA)

        # We need the first line to figure out the exact format. In particular,
        # which columns are populated
        first_line = next(oxDNA)
        has_b= first_line.count(' ') > 2
        has_n = first_line.count(' ') > 5

        # Start parsing the file in earnest. And let's not forget the first
        # line.
        for line in chain([first_line], oxDNA):
            properties = {}
            columns = line.split()

            pos = columns[:3]
            properties['position'] = np.array(pos, dtype=float)

            if has_b:
                vector = columns[3:6]
                properties['base vector'] = np.array(vector, dtype=float)
                if has_n:
                    vector = columns[6:9]
                    properties['base normal vector'] = np.array(vector, dtype=float)

            molecule.add_node(idx, **properties)
            idx += 1

    return molecule
