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

import itertools
from glob import glob
import os
from .gmx.rtp import read_rtp
from .ffinput import read_ff
from . import DATA_PATH

FORCE_FIELD_PARSERS = {'.rtp': read_rtp, '.ff': read_ff}


class ForceField(object):
    """
    Description of a force field.

    Attributes
    ----------
    blocks: dict
    links: list
    modifications: list
    renamed_residues: dict
    name: str
    variables: dict
    reference_graphs: dict
    """
    def __init__(self, directory):
        self.blocks = {}
        self.links = []
        self.modifications = []
        self.renamed_residues = {}
        self.name = os.path.basename(directory)
        self.variables = {}
        self.read_from(directory)

    def read_from(self, directory):
        """
        Populate or update the force field from a directory.

        The provided directory must contain a subdirectory with the same name
        as the force field.
        """
        source_files = iter_force_field_files(directory)
        for source in source_files:
            extension = os.path.splitext(source)[-1]
            with open(source) as infile:
                FORCE_FIELD_PARSERS[extension](infile, self)

    @property
    def reference_graphs(self):
        return self.blocks

    @property
    def features(self):
        """
        List the features declared by the links.

        Returns
        -------
        set
        """
        return set(feature for link in self.links for feature in link.features)

    def has_feature(self, feature):
        """
        Test if a feature is declared by the links.

        Parameters
        ----------
        feature: str
            The name of the feature of interest.

        Returns
        -------
        bool
        """
        return feature in self.features


def find_force_fields(directory, force_fields=None):
    """
    Read all the force fields in the given directory.

    A force field is defined as a directory that contains at least one RTP
    file. The name of the force field is the base name of the directory.

    If the force field argument is not ``None``, then it must be a dictionary
    with force field names as keys and instances of :class:`ForceField` as
    values. The force fields in the dictionary will be updated if force fields
    with the same names are found in the directory.

    Parameters
    ----------
    directory: pathlib.Path or str
        The path to the directory containing the force fields.
    force_fields: dict (optional)
        A dictionary of force fields to update.

    Returns
    -------
    dict
        A dictionary of force fields read or updated. Keys are force field
        names as strings, and values are instances of :class:`ForceField`. If a
        dictionary was provided as the "force_fields" argument, then the
        returned dictionary is the same instance as the one provided but with
        updated content.
    """
    if force_fields is None:
        force_fields = {}
    directory = str(directory)  # Py<3.6 compliance
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        try:
            next(iter_force_field_files(path))
        except StopIteration:
            pass
        else:
            try:
                if name not in force_fields:
                    force_fields[name] = ForceField(path)
                else:
                    force_fields[name].read_from(path)
            except IOError:
                msg = 'An error occured while reading the force field in  "{}".'
                raise IOError(msg.format(path))
    return force_fields


def iter_force_field_files(directory, extensions=FORCE_FIELD_PARSERS.keys()):
    """
    Returns a generator over the path of all the force field files in the directory.
    """
    return itertools.chain(*(
        glob(os.path.join(directory, '*' + extension))
        for extension in extensions
    ))


FORCE_FIELDS = find_force_fields(os.path.join(DATA_PATH, 'force_fields'))
