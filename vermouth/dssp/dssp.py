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
Assign protein secondary structures using DSSP.
"""

import collections
from functools import partial
import logging
import os
import subprocess
import tempfile
import re

from ..file_writer import deferred_open
from ..pdb import pdb
from ..system import System
from ..processors.processor import Processor
from ..processors import SortMoleculeAtoms
from ..selectors import is_protein, selector_has_position, filter_minimal, select_all
from .. import utils
from ..log_helpers import StyleAdapter, get_logger

SS_CG = {'1': 'H', '2': 'H', '3': 'H', 'H': 'H', 'G': 'H', 'I': 'H',
           'B': 'E', 'E': 'E', 'T': 'T', 'S': 'S', 'C': 'C'}

try:
    import mdtraj
except ImportError:
    HAVE_MDTRAJ = False
else:
    HAVE_MDTRAJ = True

LOGGER = StyleAdapter(get_logger(__name__))
SUPPORTED_DSSP_VERSIONS = ("2.2.1", "3.0.0")


class DSSPError(Exception):
    """
    Exception raised if DSSP fails.
    """


def read_dssp2(lines):
    """
    Read the secondary structure from a DSSP output.

    Only the first column of the "STRUCTURE" block is read. See the
    `documentation of the DSSP format`_ for more details.

    The secondary structures that can be read are:

    :H: α-helix
    :B: residue in isolated β-bridge
    :E: extended strand, participates in β ladder
    :G: 3-helix (3-10 helix)
    :I: 5 helix (π-helix)
    :T: hydrogen bonded turn
    :S: bend
    :C: loop or irregular

    The "C" code for loops and random coil is translated from the gap used in
    the DSSP file for an improved readability.

    Only the version 2 and 3 of DSSP is supported. If the format is not
    recognized as comming from that version of DSSP, then a :exc:`IOError` is
    raised.

    .. _`documentation of the DSSP format`: http://swift.cmbi.ru.nl/gv/dssp/DSSP_3.html

    Parameters
    ----------
    lines:
        An iterable over the lines of the DSSP output. This can be *e.g.* a
        list of lines, or a file handler. The new line character is ignored.

    Returns
    -------
    secstructs: list[str]
        The secondary structure assigned by DSSP as a list of one-letter
        secondary structure code.

    Raises
    ------
    IOError
        When a line could not be parsed, or if the version of DSSP
        is not supported.
    """
    secstructs = []
    # We use the line number for the error messages. It is more natural for a
    # user to count lines in a file starting from 1 rather than 0.
    numbered_lines = enumerate(lines, start=1)

    # The function can only read output from DSSP version 2 and 3. Hopefully, if
    # the input file is not in this format, then the parser will break as it
    # reads the file; we can expect that the end of the header will not be found
    # or the secondary structure will be an unexpected character.
    # We could predict from the first line that the format is not the one we
    # expect if it does not start with "===="; however, the first lines of the
    # file are non-essential and could have been trimmed. (For instance, the
    # first line of a DSSPv2 file contains the date of execution of the
    # program, which is annoying when comparing files.) Failing at the
    # first line is likely unnecessary.
    # Yet, we can identify files from DSSP v1 from the first line. These files
    # start with "****" instead of "====". If we identify such a file, we can
    # fail with a useful error message.
    _, first_line = next(numbered_lines)
    if first_line and first_line.startswith('****'):
        msg = ('Based on its header, the input file could come from a '
               'pre-July 1995 version of DSSP (or the compatibility mode '
               'of a more recent version). Only output from the version 2 and 3'
               'of DSSP are supported.')
        raise IOError(msg)

    # First we skip the header and the histogram.
    for line_num, line in numbered_lines:
        if line.startswith('  #  RESIDUE AA'):
            break
    else:  # no break
        msg = ('No secondary structure assignation could be read because the '
               'file is not formated correctly. No line was found that starts '
               'with "  #  RESIDUE AA".')
        raise IOError(msg)

    # Now, every line should be a secondary structure assignation.
    for line_num, line in numbered_lines:
        if '!' in line or not line:
            # This is a TER record or an empty line, we ignore it.
            continue
        elif len(line) >= 17:
            secondary_structure = line[16]
            if secondary_structure not in 'HBEGITS ':
                msg = 'Unrecognize secondary structure "{}" in line {}: "{}"'
                raise IOError(msg.format(secondary_structure, line_num, line))
            # DSSP represents the coil with a space. While this works in a
            # column based file, it is much less convenient to handle in
            # our code, and it is much less readable in our debug logs.
            # We translate the space to "C" in our representation.
            if secondary_structure == ' ':
                secondary_structure = 'C'
            secstructs.append(secondary_structure)
        else:
            raise IOError('Line {} is too short: "{}".'.format(line_num, line))

    return secstructs


def run_mdtraj(system):
    """
    Compute DSSP secondary structure assignments for the system by using
    ``mdtraj.compute_dssp``.

    During processing, a PDB file is produced. Therefore, all the molecules
    in the system must contain the required attributes for such a file to be
    generated. Also, the atom names are assumed to be compatible with the
    'charmm' force field for MDTraj to recognize them.
    However, the molecules do not require the edges to be defined.

    Parameters
    ----------
    system: System
        The system to process

    Returns
    -------
    list[str]
        The assigned secondary structures as a list of one-letter codes.
        The secondary structure sequences of all the molecules are combined
        in a single list without delimitation.
    """
    sys_copy = system.copy()
    # precaution for large systems; mdtraj requires all residues to be
    # grouped together otherwise dssp fails
    SortMoleculeAtoms(target_attr='atomid').run_system(sys_copy)
    tmpfile_handle, tmpfile_name = tempfile.mkstemp(suffix='.pdb', text=True,
                                                    dir='.', prefix='dssp_in_')
    tmpfile_handle = os.fdopen(tmpfile_handle, mode='w')
    tmpfile_handle.write(pdb.write_pdb_string(sys_copy, conect=False))
    tmpfile_handle.close()

    try:
        struct = mdtraj.load_pdb(tmpfile_name)
        dssp = mdtraj.compute_dssp(struct, simplified=False)
    except Exception as error:
        # Don't delete the temporary file
        message = "MDTraj encountered an error. The message was {err}. "\
                  "The input file provided to MDTraj can be found at {file}."
        raise DSSPError(message.format(err=str(error), file=tmpfile_name)) from error
    else:
        dssp = ['C' if ss == ' ' else ss for mol in dssp for ss in mol]
        if LOGGER.getEffectiveLevel() > logging.DEBUG:
            os.remove(tmpfile_name)
    return dssp


def _savefile_path(system, savedir=None):
    savefile = None
    if savedir is not None:
        chains = set()
        for molecule in system.molecules:
            first_atom = list(molecule.nodes.keys())[0]
            chain = molecule.nodes[first_atom].get('chain')
            if chain is not None:
                chains.add(chain)
        if not chains:
            msg = 'The "savedir" argument can only be used if chains are set.'
            raise ValueError(msg)
        savefile = os.path.join(savedir, 'chain_{}.ssd'.format(','.join(sorted(chains))))
    return savefile


def run_dssp(system, executable='dssp', savedir=None, defer_writing=True):
    """
    Run DSSP on a system and return the assigned secondary structures.

    Run DSSP using the path (or name in the research PATH) given by
    "executable". Return the secondary structure parsed from the output of the
    program.

    In order to call DSSP, a PDB file is produced. Therefore, all the molecules
    in the system must contain the required attributes for such a file to be
    generated. Also, the atom names are assumed to be compatible with the
    'charmm' force field for DSSP to recognize them.
    However, the molecules do not require the edges to be defined.

    DSSP is assumed to be in version 2 or 3. The secondary structure codes are
    described in :func:`read_dssp2`.

    Parameters
    ----------
    system: System
    executable: str
        Where to find the DSSP executable.
    savefile: None or str or pathlib.Path
        If set to a path, the output of DSSP is written in this directory.
    defer_writing: bool
        Whether to use :meth:`~vermouth.file_writer.DeferredFileWriter.write` for writing data

    Returns
    -------
    list[str]
        The assigned secondary structures as a list of one-letter codes.
        The secondary structure sequences of all the molecules are combined
        in a single list without delimitation.

    Raises
    ------
    DSSPError
        DSSP failed to run.
    IOError
        The output of DSSP could not be parsed.

    See Also
    --------
    read_dssp2
        Parse a DSSP output.
    """
    if savedir:
        # I don't love this. A system could contain multiple molecules, a mol
        # could contain multiple chains. Adapt _savefile_path to iterate over
        # all atoms to collect all chains sounds expensive though.
        savefile = _savefile_path(system, savedir)
    else:
        savefile = None

    try:
        # check version
        process = subprocess.run([executable, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise DSSPError("DSSP executable not found. Perhaps MDTraj was not installed in the environment?")

    match = re.search('\d+\.\d+\.\d+', process.stdout.decode('UTF8'))
    version = match[0] if match else None
    if not version:
        raise DSSPError('Failed to get DSSP version information.')
    if not version in SUPPORTED_DSSP_VERSIONS:
        LOGGER.warning("Vermouth is tested only with DSSP versions {}. "
                       "The provided DSSP (version {}) may result in inaccurate "
                       "secondary structure assignments. As alternative you can "
                       "provide a secondary structure assignment string using "
                       "the `-ss` option.",
                       SUPPORTED_DSSP_VERSIONS, version,
                       type='DSSP-version')

    tmpfile_handle, tmpfile_name = tempfile.mkstemp(suffix='.pdb', text=True,
                                                    dir='.', prefix='dssp_in_')
    tmpfile_handle = os.fdopen(tmpfile_handle, mode='w')
    tmpfile_handle.write(pdb.write_pdb_string(system, conect=False))
    tmpfile_handle.close()

    process = subprocess.run(
        [executable, '-i', tmpfile_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        universal_newlines=True
    )

    status = process.returncode
    # If an error is encountered, or the loglevel is low enough, preserve the
    # DSSP input file, and print a nice message.
    if not status and LOGGER.getEffectiveLevel() > logging.DEBUG:
        os.remove(tmpfile_name)
    if status:
        message = 'DSSP encountered an error. The message was {err}. The input' \
                  ' file provided to DSSP can be found at {file}.'
        raise DSSPError(message.format(err=process.stderr, file=tmpfile_name))
    else:
        LOGGER.debug('DSSP input file written to {}', tmpfile_name)

    if savefile is not None:
        if defer_writing:
            open = deferred_open
        with open(str(savefile), 'w') as outfile:
            outfile.write(process.stdout)

    return read_dssp2(process.stdout.split('\n'))


def annotate_dssp(molecule, callable=None, attribute='secstruct'):
    """
    Adds the DSSP assignation to the atoms of a molecule.

    Runs DSSP on the molecule and adds the secondary structure assignation as
    an attribute of its atoms. The attribute name in which the assignation is
    stored is controlled with the "attribute" argument.

    Only proteins can be annotated. Non-protein molecules are returned
    unmodified, so are empty molecules, and molecules for which no positions
    are set.

    The atom names are assumed to be compatible with DSSP. Atoms with no known
    position are not passed to DSSP which may lead to an error in DSSP.

    .. warning::

        The molecule is annotated **in-place**.

    Parameters
    ----------
    molecule: Molecule
        The molecule to annotate. Its atoms must have the attributes required
        to write a PDB file; other atom attributes, edges, or molecule
        attributes are not used.
    callable: Callable
        The function to call to generate DSSP secondary structure assignments.
        See also: :func:`run_dssp`, :func:`run_mdtraj`
    attribute: str
        The name of the atom attribute in which to store the annotation.

    See Also
    --------
    run_mdtraj, run_dssp, read_dssp2
    """
    if not is_protein(molecule):
        return

    clean_pos = molecule.subgraph(
        filter_minimal(molecule, selector=selector_has_position)
    )

    # We ignore empty molecule, there is no point at running DSSP on them.
    if not clean_pos:
        return

    system = System()
    system.add_molecule(clean_pos)
    secstructs = callable(system)

    annotate_residues_from_sequence(molecule, attribute, secstructs)


def convert_dssp_to_martini(sequence):
    """
    Convert a sequence of secondary structure to martini secondary sequence.

    Martini treats some secondary structures with less resolution than dssp.
    For instance, the different types of helices that dssp discriminates are
    seen the same by martini. Yet, different parts of the same helix are seen
    differently in martini.

    In the Martini force field, the B and E secondary structures from DSSP are
    both treated as extended regions. All the DSSP helices are treated the
    same, but the different part of the helices (beginning, end, core of a
    short helix, core of a long helix) are treated differently.

    After the conversion, the secondary structures are:
    * :F: Collagenous Fiber
    * :E: Extended structure (β sheet)
    * :H: Helix structure
    * :1: Helix start (H-bond donor)
    * :2: Helix end (H-bond acceptor)
    * :3: Ambivalent helix type (short helices)
    * :T: Turn
    * :S: Bend
    * :C: Coil

    Parameters
    ----------
    sequence: str
        A sequence of secondary structures as read from dssp. One letter per
        residue.

    Returns
    -------
    str
        A sequence of secondary structures usable for martini. One letter per
        residue.
    """
    patterns = collections.OrderedDict([
        ('.H.', '.3.'), ('.HH.', '.33.'), ('.HHH.', '.333.'),
        ('.HHHH.', '.3333.'), ('.HHHHH.', '.13332.'),
        ('.HHHHHH.', '.113322.'), ('.HHHHHHH.', '.1113222.'),
        ('.HHHH', '.1111'), ('HHHH.', '2222.'),
    ])
    cg_sequence = ''.join(SS_CG[secstruct] for secstruct in sequence)
    wildcard_sequence = ''.join('H' if secstruct == 'H' else '.'
                                for secstruct in cg_sequence)
    # Flank the sequence with dots. Otherwise in a sequence consisting of only
    # H will not have a start or end. See also issue 566.
    # This should not cause further issues, since '..' doesn't map to anything
    wildcard_sequence = '.' + wildcard_sequence + '.'
    for pattern, replacement in patterns.items():
        while pattern in wildcard_sequence:  # EXPENSIVE! :'(
            wildcard_sequence = wildcard_sequence.replace(pattern, replacement)
    # And remove the flanking dots again
    wildcard_sequence = wildcard_sequence[1:-1]
    result = ''.join(
        wildcard if wildcard != '.' else cg
        for wildcard, cg in zip(wildcard_sequence, cg_sequence)
    )
    return result


def sequence_from_residues(molecule, attribute, default=None):
    """
    Generates a sequence of `attribute`, one per residue in `molecule`.

    Parameters
    ----------
    molecule: vermouth.molecule.Molecule
        The molecule to process.
    attribute: collections.abc.Hashable
        The attribute of interest.
    default: object
        Yielded if the first node of a residue has no attribute `attribute`.

    Yields
    ------
    object
        The value of `attribute` for every residue in `molecule`.
    """
    for residue_nodes in molecule.iter_residues():
        # TODO: Make sure they're the same for every node in residue.
        first_name = residue_nodes[0]
        first_node = molecule.nodes[first_name]
        value = first_node.get(attribute, default)
        yield value


def annotate_residues_from_sequence(molecule, attribute, sequence):
    """
    Sets the attribute `attribute` to a value from `sequence` for every node in
    `molecule`. Nodes in the n'th residue of `molecule` are given the n'th
    value of `sequence`.

    Parameters
    ----------
    molecule: networkx.Graph
        The molecule to annotate. Is modified in-place.
    attribute: collections.abc.Hashable
        The attribute to set.
    sequence: collections.abc.Sequence
        The values assigned.

    Raises
    ------
    ValueError
        If the length of `sequence` is different from the number of residues in
        `molecule`.
    """
    residues = list(molecule.iter_residues())
    if len(sequence) == 1:
        sequence = sequence * len(residues)
    elif len(sequence) != len(residues):
        msg = ('The sequence length does not match the number of residues. '
               'The sequence has {} elements for {} residues.')
        raise ValueError(msg.format(len(sequence), len(residues)))
    for residue_nodes, value in zip(residues, sequence):
        for node_name in residue_nodes:
            molecule.nodes[node_name][attribute] = value


def convert_dssp_annotation_to_martini(
        molecule, from_attribute='secstruct', to_attribute='cgsecstruct'):
    """
    For every node in `molecule`, translate the `from_attribute` with
    :func:`convert_dssp_to_martini`, and assign it to the attribute
    `to_attribute`.

    Parameters
    ----------
    molecule: networkx.Graph
        The molecule to process. Is modified in-place.
    from_attribute: collections.abc.Hashable
        The attribute to read.
    to_attribute: collections.abc.Hashable
        The attribute to set.

    Raises
    ------
    ValueError
        If not all nodes have a `from_attribute`.
    """
    dssp_sequence = list(sequence_from_residues(molecule, from_attribute))
    if None not in dssp_sequence:
        cg_sequence = list(convert_dssp_to_martini(dssp_sequence))
        annotate_residues_from_sequence(molecule, to_attribute, cg_sequence)
    elif all(elem is None for elem in dssp_sequence):
        # There is no DSSP assignation for the molecule. This is likely due to
        # the molecule not being a protein. Anyway, we issue a debug message
        # as it *could* be due to the DSSP assignation having been skipped
        # for some reason.
        msg = 'No DSSP assignation to convert to Martini secondary structure intermediates.'
        LOGGER.debug(msg)
    else:
        # This is more of a problem. For now, we do not know what to do with
        # incomplete DSSP assignation. This may come later as a problem if
        # a molecule is attached to a protein.
        raise ValueError('Not all residues have a DSSP assignation.')


class AnnotateDSSP(Processor):
    name = 'AnnotateDSSP'

    def __init__(self, executable=None, savedir=None):
        super().__init__()
        if executable is None:
            if HAVE_MDTRAJ:
                self.dssp = run_mdtraj
            else:
                self.dssp = partial(run_dssp, executable='dssp', savedir=savedir)
        elif isinstance(executable, str):
            self.dssp = partial(run_dssp, executable=executable, savedir=savedir)
        else:
            self.dssp = executable

    def run_molecule(self, molecule):
        annotate_dssp(molecule, self.dssp)
        molecule.citations.add('MDTraj')
        return molecule


class AnnotateMartiniSecondaryStructures(Processor):
    name = 'AnnotateMartiniSecondaryStructures'

    @staticmethod
    def run_molecule(molecule):
        convert_dssp_annotation_to_martini(molecule)
        return molecule


class AnnotateResidues(Processor):
    """
    Set an attribute of the nodes from a sequence with one element per residue.

    Read a sequence with one element per residue and assign an attribute of
    each node based on that sequence, so each node has the value corresponding
    to its residue. In most cases, the length of the sequence has to match the
    total number of residues in the system. The sequence must be ordered in the
    same way as the residues in the system. If all the molecules have the same
    number of residues, and if the length of the sequence corresponds to the
    number of residue of one molecule, then the sequence is repeated to all
    molecules. If the sequence contains only one element, then it is repeated
    to all the residues ofthe system.

    Parameters
    ----------
    attribute: str
        Name of the node attribute to populate.
    sequence: collections.abc.Sequence
        Per-residue sequence.
    molecule_selector: collections.abc.Callable
        Function that takes an instance of :class:`vermouth.molecule.Molecule`
        as argument and returns `True` if the molecule should be considered,
        else `False`.
    """
    name = 'AnnotateResidues'

    def __init__(self, attribute, sequence,
                 molecule_selector=select_all):
        self.attribute = attribute
        self.sequence = sequence
        self.molecule_selector = molecule_selector

    def run_molecule(self, molecule):
        """
        Run the processor on a single molecule.

        Parameters
        ----------
        molecule: vermouth.molecule.Molecule

        Returns
        -------
        vermouth.molecule.Molecule
        """
        if self.molecule_selector(molecule):
            annotate_residues_from_sequence(molecule, self.attribute, self.sequence)
        return molecule

    def run_system(self, system):
        """
        Run the processor on a system.

        Parameters
        ----------
        system: vermouth.system.System

        Returns
        -------
        vermouth.system.System
        """
        # Test and adjust the length of the sequence. There are 3 valid scenarios:
        # * the length of the sequence matches the number of residues in the
        #   selection;
        # * all the molecules in the selection have the same number of residues
        #   and the sequence length matches the number of residue of one
        #   molecule; in this case the equence is repeated for each molecule;
        # * the sequence has a length of one; in this case the sequence is
        #   repeated for each residue.
        # The case were there is no molecule in the selection is only valid if
        # the sequence is empty. Then we are in the first valid scenario.
        molecule_lengths = [
            len(list(molecule.iter_residues()))
            for molecule in system.molecules
            if self.molecule_selector(molecule)
        ]
        if self.sequence and not molecule_lengths:
            raise ValueError('There is no molecule to which '
                             'to apply the sequence.')
        if (molecule_lengths
                and len(self.sequence) == molecule_lengths[0]
                and utils.are_all_equal(molecule_lengths)):
            sequence = list(self.sequence) * len(molecule_lengths)
        elif len(self.sequence) == 1:
            sequence = list(self.sequence) * sum(molecule_lengths)
        elif len(self.sequence) != sum(molecule_lengths):
            raise ValueError(
                'The length of the sequence ({}) does not match the '
                'number of residues in the selection ({}).'
                .format(len(self.sequence), sum(molecule_lengths))
            )
        else:
            sequence = self.sequence

        end = 0
        begin = 0
        for molecule, nres in zip(system.molecules, molecule_lengths):
            end += nres
            annotate_residues_from_sequence(
                molecule,
                self.attribute,
                sequence[begin:end]
            )
            begin += nres
