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
Provides a class used to describe a forcefield and all associated data.
"""


import itertools
from glob import glob
import os
from .gmx.rtp import read_rtp
from .ffinput import read_ff
from .citation_parser import read_bib
from . import DATA_PATH

FORCE_FIELD_PARSERS = {'.rtp': read_rtp, '.ff': read_ff, '.bib': read_bib}
# All data files that are in the root DATA_PATH, and apply to all force fields.
COMMON_DATA_FILES = {'citations.bib'}

# Cache the force fields.
# It should only be used by the get_native_force_field function, else it would
# allow to request a "native" force field that is not actually native.
_FORCE_FIELDS = {}


class ForceField:
    """
    Description of a force field.

    A force field can be created empty or read from a directory. In any case, a
    force field must be named. If read from a directory, the base name of the
    directory is used as force field name, unless the `name` attribute is
    provided. If the force field is created empty, then `name` must be
    provided.

    Parameters
    ----------
    directory: str or pathlib.Path, optional
        A directory to read the force field from.
    name: str, optional
        The name of the force field.

    Attributes
    ----------
    blocks: dict
    links: list
    modifications: dict
    citations: dict
    renamed_residues: dict
    name: str
    variables: dict
    """

    def __init__(self, directory=None, name=None):
        self.blocks = {}
        self.links = []
        self.modifications = {}
        self.renamed_residues = {}
        self.variables = {}
        self.name = None
        self.citations = {}
        for filename in COMMON_DATA_FILES:
            # This is kind of inefficient. Better to parse the common files
            # once, and pass around the data.
            path = DATA_PATH / filename
            self._read_from_file(path)
        if directory is not None:
            self.read_from(directory)
            self.name = os.path.basename(str(directory))
        if name is not None:
            self.name = name
        if self.name is None:
            msg = 'At least one of `directory` or `name` must be provided.'
            raise TypeError(msg)

    def __str__(self):
        return f'ForceField({self.name})'

    def __eq__(self, other):
        # Note, we cannot compare blocks/links/modifications, since those
        # compare their forcefields for equality.
        return self.name == other.name

    def read_from(self, directory):
        """
        Populate or update the force field from a directory.

        The provided directory must contain a subdirectory with the same name
        as the force field.
        """
        source_files = iter_force_field_files(directory)
        for source in source_files:
            self._read_from_file(source)

    def _read_from_file(self, path):
        extension = os.path.splitext(path)[-1]
        with open(path) as infile:
            FORCE_FIELD_PARSERS[extension](infile, self)

    @property
    def reference_graphs(self):
        """
        Returns all known blocks.

        Returns
        -------
        dict
        """
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
    force_fields: dict
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
        glob(os.path.join(str(directory), '*' + extension))
        for extension in extensions
    ))


def get_native_force_field(name):
    """
    Get a force field from the distributed library knowing its name.

    Parameters
    ----------
    name: str
        The name of the requested force field.

    Returns
    -------
    ForceField

    Raises
    ------
    KeyError
        There is no force field with the requested name in the distributed
        library.
    """
    # This function is a *temporary* solution. It reads all the distributed
    # force fields and keep a cache of them. A better solution would only parse
    # the requested force field! There would still be a need to cache the read
    # force fields, though. Indeed, force field comparison is based on instance
    # identity,so we want each force field to be  singleton.
    # TODO: Implement a better way to request a force field by name.
    global _FORCE_FIELDS
    try:
        return _FORCE_FIELDS[name]
    except KeyError:
        _FORCE_FIELDS = find_force_fields(os.path.join(DATA_PATH, 'force_fields'))
        return _FORCE_FIELDS[name]


DUMMY_FF = ForceField(name='dummy')
