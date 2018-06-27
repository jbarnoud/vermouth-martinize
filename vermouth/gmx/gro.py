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

from ..molecule import Molecule
from ..utils import first_alpha
from ..truncating_formatter import TruncFormatter

from functools import partial
from itertools import chain

import numpy as np


def read_gro(file_name, exclude=('SOL',), ignh=False):
    molecule = Molecule()
    idx = 0
    field_types = [int, str, str, int, float, float, float]
    field_names = ['resid', 'resname', 'atomname', 'atomid', 'x', 'y', 'z']
    field_widths = [5, 5, 5, 5]

    with open(file_name) as gro:
        next(gro)  # skip title
        num_atoms = int(next(gro))

        # We need the first line to figure out the exact format. In particular,
        # the precision and whether it has velocities.
        first_line = next(gro)
        has_vel = first_line.count('.') == 6
        first_dot = first_line.find('.', 25)
        second_dot = first_line.find('.', first_dot+1)
        precision = second_dot - first_dot

        field_widths.extend([precision]*3)
        if has_vel:
            field_widths.extend([precision]*3)
            field_types.extend([float]*3)
            field_names.extend(['vx', 'vy', 'vz'])

        start = 0
        slices = []
        for width in field_widths:
            if width > 0:
                slices.append(slice(start, start + width))
            start = start + abs(width)

        # Start parsing the file in earnest. And let's not forget the first
        # line.
        for line_idx, line in enumerate(chain([first_line], gro)):
            properties = {}
            # This (apart maybe from adhering to the number of lines specified
            # by the file) is the fastest method of checking whether we are at
            # the last line (box) of the file. Other things tested: regexp
            # matching, looking ahead, and testing whether the line looks like
            # a box-line. I think the reason this is faster is because the try
            # block will almost never raise an exception.
            try:
                for name, type_, slice_ in zip(field_names, field_types, slices):
                    properties[name] = type_(line[slice_].strip())
            except ValueError:
                if line_idx != num_atoms:
                    raise
                break

            properties['element'] = first_alpha(properties['atomname'])
            properties['chain'] = ''
            if properties['resname'] in exclude or (ignh and properties['element'] == 'H'):
                continue

            pos = (properties.pop('x'), properties.pop('y'), properties.pop('z'))
            properties['position'] = np.array(pos, dtype=float)

            if has_vel:
                vel = (properties.pop('vx'), properties.pop('vy'), properties.pop('vz'))
                properties['velocity'] = np.array(vel, dtype=float)

            molecule.add_node(idx, **properties)
            idx += 1
    return molecule


def write_gro(system, file_name, precision=7):
    def keyfunc(graph, node_idx):
        # TODO add something like idx_in_residue
        return graph.node[node_idx]['chain'], graph.node[node_idx]['resid'], graph.node[node_idx]['resname']

    formatter = TruncFormatter()
    pos_format_string = '{{:{ntx}.3ft}}'.format(ntx=precision+1)
    format_string = '{:5dt}{:<5st}{:>5st}{:5dt}' + pos_format_string*3
    # Pick an arbitrary node from the first molecule to see if all molecules
    # have velocities. Somehow I don't think we can write velocities for some
    # molecules but not others...
    has_vel = all('velocity' in next(iter(mol.nodes.values())) for mol in system.molecules)
    if has_vel:
        vel_format_string = '{{:{ntx}.4ft}}'*3
        vel_format_string = vel_format_string.format(ntx=precision+1)

    with open(file_name, 'w') as out:
        out.write('Martinized!\n')  # Title
        out.write(formatter.format('{:5dt}\n', system.num_particles))  # number of atoms
        atomid = 1
        for molecule in system.molecules:
            node_order = sorted(molecule, key=partial(keyfunc, molecule))
            for node_idx in node_order:
                node = molecule.node[node_idx]
                atomname = node['atomname']
                resname = node['resname']
                resid = node['resid']
                x, y, z = node['position']

                line = formatter.format(format_string, resid, resname, atomname,
                                        atomid, x, y, z)
                if has_vel:
                    vx, vy, vz = node['velocity']/10  # A to nm
                    line += formatter.format(vel_format_string, vx, vy, vz)
                atomid += 1
                out.write(line + '\n')
        # Box
        box_fmt = '{:10.5f}'*3 + '\n'
        out.write(box_fmt.format(0, 0, 0))
