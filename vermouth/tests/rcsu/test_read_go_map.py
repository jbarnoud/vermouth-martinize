# -*- coding: utf-8 -*-
# Copyright 2022 University of Groningen
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
Unit tests for the Go contact map reader.
"""
import pytest
from vermouth.rcsu.contact_map import read_go_map
import vermouth
from vermouth.tests.helper_functions import test_molecule

@pytest.mark.parametrize('lines, contacts', 
        # two sets of contacts same chain
        (("""
         ID    I1  AA  C I(PDB)     I2  AA  C I(PDB)        DCA       CMs    rCSU   Count Model
         ============================================================================================
         R      1     1  LYS A    1        2  VAL A    2       3.8094     1 1 1 1    11     369    0

         R      6     1  LYS A    1       40  THR A   40       5.4657     1 1 1 1   222     238    0
         R     13     2  VAL A    2       37  ASN A   37       7.9443     0 1 0 1   -75      75    0
         R     15     2  VAL A    2       39  ASN A   39       4.2809     0 1 1 1   -13     121    0
         """,
         [(1, "A", 2, "A"), (1, "A", 40, "A"), (2, "A", 37, "A"), (2, "A", 39, "A")]
        ),
        # two sets of contacts same chain but skip two non-OV/rCSU
        ("""
         ID    I1  AA  C I(PDB)     I2  AA  C I(PDB)        DCA       CMs    rCSU   Count Model
         ============================================================================================
         R      1     1  LYS A    1        2  VAL A    2       3.8094     1 1 1 1    11     369    0

         R      6     1  LYS A    1       40  THR A   40       5.4657     0 1 1 0   222     238    0
         R     13     2  VAL A    2       37  ASN A   37       7.9443     0 1 0 0   -75      75    0
         R     15     2  VAL A    2       39  ASN A   39       4.2809     0 1 1 1   -13     121    0
         """,
         [(1, "A", 2, "A"), (2, "A", 39, "A")]
        ),
        # two sets of contacts different chains
        ("""
         ID    I1  AA  C I(PDB)     I2  AA  C I(PDB)        DCA       CMs    rCSU   Count Model
         ============================================================================================
         R      1     1  LYS A    1        2  VAL B    2       3.8094     1 1 1 1    11     369    0
         R      6     1  LYS A    1       40  THR B   40       5.4657     1 1 1 1   222     238    0
         R     13     2  VAL C    2       37  ASN D   37       7.9443     0 1 0 1   -75      75    0
         R     15     2  VAL C    2       39  ASN D   39       4.2809     0 1 1 1   -13     121    0
         """,
         [(1, "A", 2, "B"), (1, "A", 40, "B"), (2, "C", 37, "D"), (2, "C", 39, "D")]
        )))
def test_go_map(test_molecule, tmp_path, lines, contacts):
    # write the go contact map file
    with open(tmp_path / "go_file.txt", "w") as in_file:
        in_file.write(lines)

    system = vermouth.System()
    system.add_molecule(test_molecule)

    # read go map
    read_go_map(system, tmp_path / "go_file.txt")
    assert system.go_params["go_map"][0] == contacts

def test_go_error(test_molecule, tmp_path):
    lines="""
          ID    I1  AA  C I(PDB)     I2  AA  C I(PDB)        DCA       CMs    rCSU   Count Model
          ============================================================================================
          No valid contacts in this file.
          """
    # write the go contact map file
    with open(tmp_path / "go_file.txt", "w") as in_file:
        in_file.write(lines)

    system = vermouth.System()
    system.add_molecule(test_molecule)

    with pytest.raises(IOError):
        read_go_map(system, tmp_path / "go_file.txt")
