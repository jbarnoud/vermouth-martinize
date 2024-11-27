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
Read GROMACS .itp files.

"""

import collections
from vermouth.molecule import (Block, Interaction)
from vermouth.parser_utils import (SectionLineParser, _tokenize)
from vermouth.parser_utils import split_comments

class ITPDirector(SectionLineParser):
    """
    class for reading itp files.
    """
    COMMENT_CHAR = ';'

    atom_idxs = {'bonds': [0, 1],
                 'position_restraints': [0],
                 'angles': [0, 1, 2],
                 'constraints': [0, 1],
                 'dihedrals': [0, 1, 2, 3],
                 'pairs': [0, 1],
                 'pairs_nb': [0, 1],
                 'exclusions': [slice(None, None)],
                 'virtual_sites1': [0],
                 'virtual_sites2': [0, 1, 2],
                 'virtual_sites3': [0, 1, 2, 3],
                 'virtual_sites4': [slice(0, 5)],
                 'virtual_sitesn': [0, slice(2, None)],
                 'settles': [0],
                 'distance_restraints':  [0, 1],
                 'dihedral_restraints':  [slice(0, 4)],
                 'orientation_restraints': [0, 1],
                 'angle_restraints': [slice(0, 4)],
                 'angle_restraints_z': [0, 1]}

    def __init__(self, force_field):
        super().__init__()
        self.force_field = force_field
        self.current_block = None
        self.current_meta = None
        self.blocks = collections.OrderedDict()
        self.header_actions = {
            ('moleculetype', ): self._new_block
        }
        # a list of nodes of current-block
        self.current_atom_names = []
        self.current_comment = None

    def parse(self, file_handle):
        """
        Reads lines from `file_handle`, and calls :meth:`dispatch` to find
        which method to call to do the actual parsing. Yields the result of
        that call, if it's not `None`.
        At the end, calls :meth:`finalize`, and yields its results, iff
        it's not None.
        Parameters
        ----------
        file_handle: collections.abc.Iterable[str]
            The data to parse. Should produce lines of data.
        Yields
        ------
        object
            The results of dispatching to parsing methods, and of
            :meth:`finalize`.
        """
        lineno = 0
        for lineno, line in enumerate(file_handle, 1):
            line, comment = split_comments(line, self.COMMENT_CHAR)
            self.current_comment = comment
            if not line:
                continue
            result = self.dispatch(line)(line, lineno)

            if result is not None:
                yield result

        result = self.finalize(lineno)
        if result is not None:
            yield result

    def dispatch(self, line):
        """
        Looks at `line` to see what kind of line it is, and returns either
        :meth:`parse_header` if `line` is a section header or
        :meth:`vermouth.parser_utils.SectionLineParser.parse_section` otherwise.
        Calls :meth:`vermouth.parser_utils.SectionLineParser.is_section_header` to see
        whether `line` is a section header or not.

        Parameters
        ----------
        line: str

        Returns
        -------
        collections.abc.Callable
            The method that should be used to parse `line`.
        """

        if self.is_section_header(line):
            return self.parse_header
        elif self.is_pragma(line):
            return self.parse_pragma
        else:
            return self.parse_section

    @staticmethod
    def is_pragma(line):
        """
        Parameters
        ----------
        line: str
            A line of text.

        Returns
        -------
        bool
            ``True`` iff `line` is a def statement.
        """
        return line.startswith('#')

    def parse_pragma(self, line, lineno=0):
        """
        Parses the beginning and end of define sections
        with line number `lineno`. Sets attr current_meta
        when applicable. Does check if ifdefs overlap.

        Parameters
        ----------
        line: str
        lineno: str

        Returns
        -------
        object
            The result of calling :meth:`finalize_section`, which is called
            if a section ends.

        Raises
        ------
        IOError
            If the def sections are missformatted
        """
        if line == '#endif':
            if self.current_meta is not None:
                self.current_meta = None
            elif self.current_meta is None:
                raise IOError("Your #ifdef section is orderd incorrectly."
                              "At line {} I read #endif but I haven not read"
                              "a ifdef before.".format(lineno))

        elif line.startswith("#else"):
            if self.current_meta is None:
               raise IOError("Your #ifdef section is orderd incorrectly."
                             "At line {} I read #endif but I haven not read"
                             "a ifdef before.".format(lineno))

            inverse = {"ifdef": "ifndef", "ifndef": "ifdef"}
            tag = self.current_meta["tag"]
            condition = inverse[self.current_meta["condition"]]
            self.current_meta = {'tag': tag, 'condition': condition.replace("#", "")}

        elif line.startswith("#ifdef") or line.startswith("#ifndef"):
            if self.current_meta is None:
                condition, tag = line.split()
                self.current_meta = {'tag': tag, 'condition': condition.replace("#", "")}
            elif self.current_meta is not None:
                raise IOError("Your #ifdef/#ifndef section is orderd incorrectly."
                              "At line {} I read {} but there is still"
                              "an open #ifdef/#ifndef section from"
                              "before.".format(lineno, line.split()[0]))
        # Guard against unkown pragmas like #if or #include
        else:
            raise IOError("Don't know how to parse pargma {} at"
                          "line {}.".format(line, lineno))

    def parse_header(self, line, lineno=0):
        """
        Parses a section header with line number `lineno`. Sets
        :attr:`vermouth.parser_utils.SectionLineParser.section`
        when applicable. Does not check whether `line` is a valid section
        header.

        Parameters
        ----------
        line: str
        lineno: str

        Returns
        -------
        object
            The result of calling :meth:`finalize_section`, which is called
            if a section ends.

        Raises
        ------
        KeyError
            If the section header is unknown.
        """
        prev_section = self.section

        ended = []
        section = self.section + [line.strip('[ ]').casefold()]
        if tuple(section[-1:]) in self.METH_DICT:
            self.section = section[-1:]
        else:
            while tuple(section) not in self.METH_DICT and len(section) > 1:
                ended.append(section.pop(-2))  # [a, b, c, d] -> [a, b, d]
            self.section = section

        result = None

        if len(prev_section) != 0:
            result = self.finalize_section(prev_section, ended)

        action = self.header_actions.get(tuple(self.section))
        if action:
            action()

        return result

    def finalize_section(self, previous_section, ended_section):
        """
        Called once a section is finished. It appends the current_links list
        to the links and update the block dictionary with current_block. Thereby it
        finishes reading a given section.

        Parameters
        ---------
        previous_section: list[str]
            The last parsed section.
        ended_section: list[str]
            The sections that have been ended.
        """
        if "atoms" in ended_section:
            self.current_atom_names = list(self.current_block.nodes)

        if self.current_block is not None:
            self.force_field.blocks[self.current_block.name] = self.current_block

    def finalize(self, lineno=0):
        """
        Called at the end of the file and checks that all pragmas are closed
        before calling the parent method.
        """
        if self.current_meta is not None:
            raise IOError("Your #ifdef/#ifndef section is orderd incorrectly."
                          "There is no #endif for the last pragma.")

        super().finalize()

    def _new_block(self):
        self.current_block = Block(force_field=self.force_field)

    @SectionLineParser.section_parser('moleculetype')
    def _block(self, line, lineno=0):
        """
        Parses the line directly following the '[moleculetype]'
        directive and stores the block name and exclusions.
         """
        name, nrexcl = line.split()
        self.current_block.name = name
        self.current_block.nrexcl = int(nrexcl)

    @SectionLineParser.section_parser('moleculetype', 'atoms')
    def _block_atoms(self, line, lineno=0):
        """
        Parses the lines of the [atoms] directive.
        """
        tokens = collections.deque(_tokenize(line))
        self._parse_block_atom(tokens, self.current_block)

    @SectionLineParser.section_parser('moleculetype', 'bonds')
    @SectionLineParser.section_parser('moleculetype', 'angles')
    @SectionLineParser.section_parser('moleculetype', 'dihedrals')
    @SectionLineParser.section_parser('moleculetype', 'impropers')
    @SectionLineParser.section_parser('moleculetype', 'constraints')
    @SectionLineParser.section_parser('moleculetype', 'pairs')
    @SectionLineParser.section_parser('moleculetype', 'exclusions')
    @SectionLineParser.section_parser('moleculetype', 'virtual_sites1')
    @SectionLineParser.section_parser('moleculetype', 'virtual_sites2')
    @SectionLineParser.section_parser('moleculetype', 'virtual_sites3')
    @SectionLineParser.section_parser('moleculetype', 'virtual_sites4')
    @SectionLineParser.section_parser('moleculetype', 'virtual_sitesn')
    @SectionLineParser.section_parser('moleculetype', 'position_restraints')
    @SectionLineParser.section_parser('moleculetype', 'pairs_nb')
    @SectionLineParser.section_parser('moleculetype', 'settles')
    @SectionLineParser.section_parser('moleculetype', 'distance_restraints')
    @SectionLineParser.section_parser('moleculetype', 'dihedral_restraints')
    @SectionLineParser.section_parser('moleculetype', 'orientation_restraints')
    @SectionLineParser.section_parser('moleculetype', 'angle_restraints')
    @SectionLineParser.section_parser('moleculetype', 'angle_restraints_z')
    def _interactions(self, line, lineno=0):
        """
        Parses all interaction lines that are not directives (i.e. within []).
        Note that each interaction is enumerated explicitly to guard against typos
        and also interactions for which the format is unknown.
        """
        context = self.current_block
        interaction_name = self.section[-1]
        tokens = collections.deque(_tokenize(line))

        atom_idxs = self.atom_idxs.get(interaction_name)

        self._base_parser(
            tokens,
            context,
            section=interaction_name,
            atom_idxs=atom_idxs,
            )

    def _split_atoms_and_parameters(self, tokens, atom_idxs):
        """
        Returns atoms from line based on the indices defined in `atom_idxs`.
        It also interprets slices etc. stored as strings.

        Parameters:
        ------------
        tokens: collections.deque[str]
            Deque of token to inspect. The deque **can be modified** in place.
        atom_idxs: list of ints or strings that are valid python slices

        Returns:
        -----------
        list
        """

        atoms = []
        remove = []
        # first we extract the atoms from the indices given using
        # ints or slices
        tokens = list(tokens)
        for idx in atom_idxs:
            if isinstance(idx, int):
                atoms.append([tokens[idx], {}])
                remove.append(idx)
            elif isinstance(idx, slice):
                atoms += [[atom, {}] for atom in tokens[idx]]
                idx_range = range(0, len(tokens))
                remove += idx_range[idx]
            else:
                raise IOError

        # everything that is left are parameters, which we
        # get by simply deleting the atoms from tokens

        for index in sorted(remove, reverse=True):
            del tokens[index]

        return atoms, tokens

    def _treat_block_interaction_atoms(self, atoms, context, section):
        """
        Takes the atom indices associated with an interaction line
        and converts it to zero based indices. It also performas some
        format checks. It checks that:

              - the indices are not negative or zero
              - the indices refere to an existing atom
              - atoms have no prefixes

        Parameters
        -----------
        atom_idxs: list of ints or strings that are valid python slices
        context: :class:`vermouth.molecule.Block`
            The current block we parse
        section: str
            The current section header
        """
        all_references = []
        for atom in atoms:
            reference = atom[0]
            if reference.isdigit():
                if int(reference) < 1:
                    msg = 'In section {} is a negative atom reference, which is not allowed.'
                    raise IOError(msg.format(section.name))

               # The indices in the file are 1-based
                reference = int(reference) - 1
                try:
                    reference = self.current_atom_names[reference]
                except IndexError:
                    msg = ('There are {} atoms defined in the block "{}". '
                           'Interaction in section "{}" cannot refer to '
                           'atom index {}.')
                    raise IOError(msg.format(len(context), context.name,
                                             section, reference + 1))
                atom[0] = reference
            else:
                if reference not in context:
                    msg = ('There is no atom "{}" defined in the block "{}". '
                           'Section "{}" cannot refer to it.')
                    raise IOError(msg.format(reference, context.name, section))
                if reference[0] in '+-<>':
                    msg = ('Atom names in blocks cannot be prefixed with + or -. '
                           'The name "{}", used in section "{}" of the block "{}" '
                           'is not valid in a block.')
                    raise IOError(msg.format(reference, section, context.name))
            all_references.append(reference)
        return all_references

    def _base_parser(self, tokens, context, section, atom_idxs):
        """
        Converts an interaction line into a vermouth interaction
        tuple. It updates the block interactions in place.
        Parameters
        ----------
        tokens: collections.deque[str]
            Deque of token to inspect. The deque **can be modified** in place.
        context: :class:`vermouth.molecule.Block`
            The current block we parse
        section: str
            The current section header
        atom_idxs: list of ints or strings that are valid python slices
        """
        # split atoms and parameters

        atoms, parameters = self._split_atoms_and_parameters(tokens, atom_idxs)

        # perform check on the atom ids
        treated_atoms = self._treat_block_interaction_atoms(atoms, context, section)

        if self.current_meta:
            meta = {self.current_meta['condition']: self.current_meta['tag']}
        else:
            meta = {}

        if self.current_comment:
            meta['comment'] = self.current_comment

        interaction = Interaction(
            atoms=treated_atoms,
            parameters=parameters,
            meta=meta,)

        context.interactions[section] = context.interactions.get(section, []) + [interaction]

    def _parse_block_atom(self, tokens, context):
        """
        Converts the lines of the atom directive to graph nodes and
        sets the values (i.e. atomtype) as attributes.

        Parameters
        ----------
        tokens: collections.deque[str]
            Deque of token to inspect. The deque **can be modified** in place.
        context: :class:`vermouth.molecule.Block`
            The current block we parse
        """
        # deque does not support slicing
        first_six = (tokens.popleft() for _ in range(6))
        index, atype, resid, resname, name, charge_group = first_six
        # since the index becomes the node name and all graphs start with 0
        # index it makes more sense to also directly start at 0

        if int(index) < 1:
            msg = 'One of your atoms has a negative atom reference, which is not allowed.'
            raise IOError(msg)

        index = int(index) - 1

        if index in context:
            msg = ('There is already an atom with index "{}" in the block "{}". '
                   'Atom indices must be unique within a block.')
            raise IOError(msg.format(name, context.name))

        atom = {
            # for bookkeeping purposes let's keep the actual index i.e. +1 here
            'index'   : index + 1,
            'atomname': name,
            'atype': atype,
            'resname': resname,
            'resid': int(resid),
            'charge_group': int(charge_group),
        }

       # charge and mass are optional, but charge has to be defined for mass to be
        if tokens:
            atom['charge'] = float(tokens.popleft())
        if tokens:
            atom['mass'] = float(tokens.popleft())

        attributes = {}

        context.add_node(index, **dict(collections.ChainMap(attributes, atom)))

def read_itp(lines, force_field):
    """
    Parses `lines` of itp format and adds the
    molecule as a block to `force_field`.

    Parameters
    ----------
    lines: list
        list of lines of an itp file
    force_field: :class:`vermouth.forcefield.ForceField`
    """
    director = ITPDirector(force_field)
    return list(director.parse(iter(lines)))
