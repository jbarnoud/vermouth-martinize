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
Test for the tune idp bonds processor.
"""
import pytest
from vermouth.processors.annotate_idrs import AnnotateIDRs
from vermouth.tests.helper_functions import create_sys_all_attrs, test_molecule

@pytest.mark.parametrize('idr_regions, expected', [
    (
    [(1,2)],
    [True, True, True, True, True, False, False, False, False]
    ),
    (
    [],
    [False, False, False, False, False, False, False, False, False]
    )
])
def test_make_disorder_string(test_molecule,
                              idr_regions,
                              expected):
    atypes = {0: "P1", 1: "SN4a", 2: "SN4a",
              3: "SP1", 4: "C1",
              5: "TP1",
              6: "P1", 7: "SN3a", 8: "SP4"}
    # the molecule resnames
    resnames = {0: "A", 1: "A", 2: "A",
                3: "B", 4: "B",
                5: "C",
                6: "D", 7: "D", 8: "D"}
    secstruc = {1: "H", 2: "H", 3: "H", 4: "H"}

    system = create_sys_all_attrs(test_molecule,
                                  moltype="molecule_0",
                                  secstruc=secstruc,
                                  defaults={"chain": "A"},
                                  attrs={"resname": resnames,
                                         "atype": atypes})

    AnnotateIDRs(id_regions=idr_regions).run_system(system)
    result = []
    for key, node in system.molecules[0].nodes.items():
        if system.molecules[0].nodes[key].get("cgidr"):
            result.append(True)
        else:
            result.append(False)
    assert result == expected

@pytest.mark.parametrize('idr_regions, expected',(
        ([(1, 4)],
         {0: "C", 1: "C", 2: "C",
          3: "C", 4: "C",
          5: "C",
          6: "C", 7: "C", 8: "C"}),
        ([(1, 2)],
         {0: "C", 1: "C", 2: "C",
          3: "C", 4: "C",
          5: "H",
          6: "H", 7: "H", 8: "H"}),

))
def test_ss_reassign(test_molecule, idr_regions, expected):
    secstruc = {1: "H", 2: "H", 3: "H", 4: "H"}
    resnames = {0: "A", 1: "A", 2: "A",
                3: "B", 4: "B",
                5: "C",
                6: "D", 7: "D", 8: "D"}
    atypes = {0: "P1", 1: "SN4a", 2: "SN4a",
              3: "SP1", 4: "C1",
              5: "TP1",
              6: "P1", 7: "SN3a", 8: "SP4"}
    cgsectruc = {0: "H", 1: "H", 2: "H",
                 3: "H", 4: "H",
                 5: "H",
                 6: "H", 7: "H", 8: "H"}  # i.e. all ss is H to begin

    system = create_sys_all_attrs(test_molecule,
                                  moltype="molecule_0",
                                  secstruc=secstruc,
                                  defaults={"chain": "A"},
                                  attrs={"resname": resnames,
                                         "atype": atypes,
                                         "cgsecstruct": cgsectruc})

    AnnotateIDRs(id_regions=idr_regions).run_system(system)

    for key, node in system.molecules[0].nodes.items():
        assert system.molecules[0].nodes[key]["cgsecstruct"] == expected[key]
