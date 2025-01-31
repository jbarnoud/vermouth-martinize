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
Test the writing of TOP file.
"""
import os
import textwrap
import pytest
import vermouth
from vermouth.file_writer import DeferredFileWriter
from vermouth.gmx.topology import (sigma_epsilon_to_C6_C12,
                                   write_atomtypes,
                                   write_nonbond_params,
                                   write_gmx_topology,
                                   Atomtype,
                                   NonbondParam)
from vermouth.tests.gmx.test_itp import dummy_molecule

@pytest.mark.parametrize('atomtypes, expected, C6C12',
        # simple atomtype
        #'Atomtype', 'molecule node sigma epsilon meta
        ((
        [{"node": 0, "sigma": 0.43, "epsilon": 2.3, "meta": {}}],
        ["[ atomtypes ]\n", 
         "A 72 0 A 0.43000000 2.30000000 \n"], False),
        (
        [{"node": 0, "sigma": 0.43, "epsilon": 2.3, "meta": {}}],
        ["[ atomtypes ]\n", 
         "A 72 0 A 254.62172908 37693.15402307 \n"], True),
        (
        [{"node":0, "sigma": 0.43, "epsilon": 2.3, "meta": {}},
         {"node":1, "sigma": 0.47, "epsilon": 2.5, "meta": {}}],
        ["[ atomtypes ]\n", 
         "A 72 0 A 0.43000000 2.30000000 \n",
         "B 72 0 A 0.47000000 2.50000000 \n"], False),
        (
        [{"node": 0, "sigma": 0.43, "epsilon": 2.3, "meta": {"comment": ["comment"]}}],
        ["[ atomtypes ]\n", 
         "A 72 0 A 0.43000000 2.30000000 ;comment\n"], False),
        (
        [{"node":0, "sigma": 0.43, "epsilon": 2.3, "meta": {"group": "g1"}},
         {"node":1, "sigma": 0.44, "epsilon": 3.3, "meta": {}},
         {"node":2, "sigma": 0.47, "epsilon": 2.5, "meta": {"group": "g1"}}],
        ['[ atomtypes ]\n', 
         'B 72 0 A 0.44000000 3.30000000 \n', 
         '; g1\n', 
         'A 72 0 A 0.43000000 2.30000000 \n', 
         'C 72 0 A 0.47000000 2.50000000 \n'], False
        ),
        (
        [{"node":0, "sigma": 0.43, "epsilon": 2.3, "meta": {"ifdef": "g1"}},
         {"node":1, "sigma": 0.44, "epsilon": 3.3, "meta": {}},
         {"node":2, "sigma": 0.47, "epsilon": 2.5, "meta": {"ifdef": "g1"}}],
        ['[ atomtypes ]\n', 
         'B 72 0 A 0.44000000 3.30000000 \n', 
         '#ifdef g1\n', 
         'A 72 0 A 0.43000000 2.30000000 \n', 
         'C 72 0 A 0.47000000 2.50000000 \n',
         '#endif'], False
        ),
        ))
def test_atomtypes(tmp_path, dummy_molecule, atomtypes, expected, C6C12):
    """
    Test that the atomtypes directive is properly written.
    """
    dummy_sys = vermouth.system.System("")
    for atype in atomtypes:
        # somewhat hacky workaround because pytest doesn't
        # allow the use of fixtures in parametrize
        atype['molecule'] = dummy_molecule
        dummy_sys.gmx_topology_params['atomtypes'].append(Atomtype(**atype))

    outpath = tmp_path / 'out.itp'
    write_atomtypes(dummy_sys, outpath, C6C12=C6C12)
    DeferredFileWriter().write()

    with open(str(outpath)) as infile:
        for line, ref_line in zip(infile, expected):
            assert line == ref_line


@pytest.mark.parametrize('nbparams, expected, C6C12',
        # simple atomtype
        #'Atomtype', 'molecule node sigma epsilon meta
        ((
        [{"atoms": ("A", "B"), "sigma": 0.43, "epsilon": 2.3, "meta": {}}],
        ["[ nonbond_params ]\n", 
         "A B 1 0.43000000 2.30000000 \n"], False),
        (
        [{"atoms": ("A", "B"), "sigma": 0.43, "epsilon": 2.3, "meta": {}}],
        ["[ nonbond_params ]\n", 
         "A B 1 254.62172908 37693.15402307 \n"], True),
        (
        [{"atoms": ("A", "B"), "sigma": 0.43, "epsilon": 2.3, "meta": {}},
         {"atoms": ("A", "C"), "sigma": 0.47, "epsilon": 2.5, "meta": {}}],
        ["[ nonbond_params ]\n", 
         "A B 1 0.43000000 2.30000000 \n",
         "A C 1 0.47000000 2.50000000 \n"], False),
        # self interaction
        (
        [{"atoms": ("A",), "sigma": 0.43, "epsilon": 2.3, "meta": {}},
         {"atoms": ("A", "C"), "sigma": 0.47, "epsilon": 2.5, "meta": {}}],
        ["[ nonbond_params ]\n", 
         "A A 1 0.43000000 2.30000000 \n",
         "A C 1 0.47000000 2.50000000 \n"], False),
        (
        [{"atoms": ("A", "B"), "sigma": 0.43, "epsilon": 2.3, "meta": {"comment": ["comment"]}}],
        ["[ nonbond_params ]\n", 
         "A B 1 0.43000000 2.30000000 ;comment\n"], False),
        (
        [{"atoms":("A", "B"), "sigma": 0.43, "epsilon": 2.3, "meta": {"group": "g1"}},
         {"atoms":("B", "C"), "sigma": 0.44, "epsilon": 3.3, "meta": {}},
         {"atoms":("A", "C"), "sigma": 0.47, "epsilon": 2.5, "meta": {"group": "g1"}}],
        ['[ nonbond_params ]\n', 
         'B C 1 0.44000000 3.30000000 \n', 
         '; g1\n', 
         'A B 1 0.43000000 2.30000000 \n', 
         'A C 1 0.47000000 2.50000000 \n'], False
        ),
        (
        [{"atoms":("A", "B"), "sigma": 0.43, "epsilon": 2.3, "meta": {"ifdef": "g1"}},
         {"atoms":("B", "C"), "sigma": 0.44, "epsilon": 3.3, "meta": {}},
         {"atoms":("A", "C"), "sigma": 0.47, "epsilon": 2.5, "meta": {"ifdef": "g1"}}],
        ['[ nonbond_params ]\n', 
         'B C 1 0.44000000 3.30000000 \n', 
         '#ifdef g1\n', 
         'A B 1 0.43000000 2.30000000 \n', 
         'A C 1 0.47000000 2.50000000 \n',
         '#endif'], False
        ),
        ))
def test_nonbond_params(tmp_path, nbparams, expected, C6C12):
    """
    Test that the atomtypes directive is properly written.
    """
    dummy_sys = vermouth.system.System("")
    for nbparam in nbparams:
        dummy_sys.gmx_topology_params['nonbond_params'].append(NonbondParam(**nbparam))

    outpath = tmp_path / 'out.itp'
    write_nonbond_params(dummy_sys, outpath, C6C12=C6C12)
    DeferredFileWriter().write()

    with open(str(outpath)) as infile:
        for line, ref_line in zip(infile, expected):
            assert line == ref_line

def test_toplevel_topology(tmp_path, dummy_molecule):
    """
    Make sure the toplevel topology file
    is correctly written. Note that the individual
    molecule itp files are seperately checked in
    gmx itp module or in the previous tests.
    """
    os.chdir(tmp_path)
    system = vermouth.System()
    system.add_molecule(dummy_molecule)
    dummy_molecule.meta['moltype'] = "molecule_0"
    # "node": 0, "sigma": 0.43, "epsilon": 2.3, "meta": {}}
    system.gmx_topology_params['atomtypes'].append(Atomtype(node=0,
                                                   molecule=dummy_molecule,
                                                   sigma=0.43,
                                                   meta={},
                                                   epsilon=2.3))
    system.gmx_topology_params['nonbond_params'].append(NonbondParam(atoms=("A", "B"),
                                                                     sigma=0.43,
                                                                     epsilon=2.3,
                                                                     meta={}))
    outpath = tmp_path / 'out.itp'
    atompath = tmp_path / 'atomtypes.itp'
    nbpath = tmp_path / 'nonbond_params.itp'
    write_gmx_topology(system,
                       outpath,
                       header=['first header comment', 'second header comment'],
                       defines=('random', ),
                       itp_paths={"atomtypes": atompath, "nonbond_params": nbpath},
                       # at this level C6C12 doesn't matter; it gets
                       # checked in previous texts
                       C6C12=False)
    DeferredFileWriter().write()

    reference =f"""#define random
#include "martini.itp"
#include "molecule_0.itp"

[ system ]
Title of the system

[ molecules ]
molecule_0    1
"""
    ref_lines = textwrap.dedent(reference).splitlines()
    with open(str(outpath)) as infile:
        for line, ref_line in zip(infile, ref_lines):
            assert line.strip() == ref_line
