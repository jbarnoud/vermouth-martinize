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
Provides a class to describe a system.
"""
from collections import defaultdict
from .forcefield import DUMMY_FF

class System:
    """
    A system of molecules.

    Attributes
    ----------
    molecules: list[:class:`~vermouth.molecule.Molecule`]
        The molecules in the system.
    """
    def __init__(self, force_field=None):
        self.molecules = []
        self._force_field = None
        if force_field is None:
            # Create a dummy FF
            force_field = DUMMY_FF
        self.force_field = force_field
        self.gmx_topology_params = defaultdict(list)
        self.go_params = defaultdict(list)

    @property
    def force_field(self):
        """
        The forcefield used to describe the molecules in this system.
        """
        return self._force_field

    @force_field.setter
    def force_field(self, value):
        """
        Set the forcefield for all the molecules in the system.

        Parameters
        ----------
        value: :class:`~vermouth.forcefield.ForceField`
        """
        self._force_field = value
        for molecule in self.molecules:
            molecule._force_field = value  # pylint: disable=protected-access

    def add_molecule(self, molecule):
        """
        Add a molecule to the system.

        Parameters
        ----------
        molecule: :class:`~vermouth.molecule.Molecule`
        """
        if molecule.force_field is None:
            molecule._force_field = self.force_field
        if molecule.force_field != self.force_field:
            if self.force_field is DUMMY_FF:
                # We didn't have a force field, so take the one from the
                # incoming molecule
                self.force_field = molecule.force_field
            else:
                raise KeyError(f'Cannot add Molecule with force field '
                               f'{molecule.force_field} to system with force field '
                               f'{self.force_field}')
        self.molecules.append(molecule)

    @property
    def num_particles(self):
        """
        The total number of particles in all the molecules in this system.
        """
        return sum(len(mol) for mol in self.molecules)

    def copy(self):
        """
        Creates a copy of this system and it's molecules.

        Returns
        -------
        System
            A deep copy of this system.
        """
        new_system = self.__class__()
        new_system.molecules = [mol.copy() for mol in self.molecules]
        new_system.force_field = self.force_field
        return new_system
