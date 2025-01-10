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
Handle the RTP format from Gromacs.
"""

import collections
import itertools
import networkx as nx

from ..molecule import Block, Link, Interaction
from .. import utils

__all__ = ['read_rtp']

# Name of the subsections in RTP files.
# Names starting with a '_' are for internal use.
RTP_SUBSECTIONS = ('atoms', 'bonds', 'angles', 'dihedrals',
                   'impropers', 'cmap', 'exclusions',
                   '_bondedtypes')


_BondedTypes = collections.namedtuple(
    '_BondedTypes',
    'bonds angles dihedrals impropers all_dihedrals nrexcl HH14 remove_dih'
)


class _IterRTPSubsectionLines:
    """
    Iterate over the lines of an RTP file within a subsection.
    """

    def __init__(self, parent):
        self.parent = parent
        self.lines = parent.lines
        self.running = True

    def __next__(self):
        if not self.running:
            raise StopIteration
        line = next(self.lines)
        if line.strip().startswith('['):
            self.parent.buffer.append(line)
            self.running = False
            raise StopIteration
        return line

    def __iter__(self):
        return self

    def flush(self):
        """
        Move the iterator after the last line of the subsection.
        """
        for _ in self:
            pass


class _IterRTPSubsections:
    """
    Iterate over the subsection of a RTP file within a section.

    For each subsections, yields its name and  an iterator over its lines.
    """

    def __init__(self, parent):
        self.parent = parent
        self.lines = parent.lines
        self.buffer = collections.deque()
        self.current_subsection = None
        self.running = True

    def __next__(self):
        if not self.running:
            raise StopIteration
        if self.current_subsection is not None:
            self.current_subsection.flush()
        if self.buffer:
            line = self.buffer.popleft()
        else:
            line = next(self.lines)
        stripped = line.strip()
        if stripped.startswith('['):
            # A section header looks like "[ name ]". It matches the following
            # regexp: r"^\s*\[\s*(?P<name>)\s*\]\s*$". Once stripped, the
            # trailing spaces at the beginning and at the end are removed so
            # the string starts with '[' and ends with ']'. The slicing remove
            # these brackets. The final call to strip remove the potential
            # white characters between the brackets and the section name.
            name = stripped[1:-1].strip()
            if name in RTP_SUBSECTIONS:
                subsection = _IterRTPSubsectionLines(self)
                self.current_subsection = subsection
                return name, subsection
            self.parent.buffer.append(line)
            self.running = False
            raise StopIteration
        raise IOError('I am almost sure I should not be here...')

    def __iter__(self):
        return self

    def flush(self):
        """
        Move the iterator after the last subsection of the section.
        """
        for _ in self:
            pass


class _IterRTPSections:
    """
    Iterate over the sections of a RTP file.

    For each section, yields the name of the sections and an iterator over its
    subsections.
    """

    def __init__(self, lines):
        self.lines = lines
        self.buffer = collections.deque()
        self.current_section = None

    def __next__(self):
        if self.current_section is not None:
            self.current_section.flush()
        if self.buffer:
            line = self.buffer.popleft()
        else:
            line = next(self.lines)
        stripped = line.strip()
        if stripped.startswith('['):
            name = stripped[1:-1].strip()
            section = _IterRTPSubsections(self)
            self.current_section = section
            # The "[ bondedtypes ]" is special in the sense that it does
            # not have subsection, but instead have its content directly
            # under it. This breaks the neat hierarchy the rest of the file
            # has. Here we restore the hierarchy by faking that the file
            # contains:
            #
            #    [ bondedtypes ]
            #     [ _bondedtypes ]
            #
            # For now, I do that on the basis of the section name. If other
            # sections are special that way, I'll detect them with a look
            # ahead.
            if name == 'bondedtypes':
                section.buffer.append(' [ _bondedtypes ]')
            return name, section
        raise IOError('Hum... There is a bug in the RTP reader.')

    def __iter__(self):
        return self


def _atoms(subsection, block):
    for line in subsection:
        name, atype, charge, charge_group = line.split()
        atom = {
            'atomname': name,
            'atype': atype,
            'charge': float(charge),
            'charge_group': int(charge_group),
        }
        block.add_atom(atom)


def _base_rtp_parser(interaction_name, natoms):
    def wrapped(subsection, block):
        """
        Parse the lines from a RTP subsection and populate the block.
        """
        interactions = []
        for line in subsection:
            splitted = line.strip().split()
            atoms = splitted[:natoms]
            parameters = splitted[natoms:]
            interactions.append(Interaction(atoms=atoms,
                                            parameters=parameters,
                                            meta={}))
        block.interactions[interaction_name] = interactions
    return wrapped


def _parse_bondedtypes(section):
    # Default taken from
    # 'src/gromacs/gmxpreprocess/resall.cpp::read_resall' in the Gromacs
    # source code.
    defaults = _BondedTypes(bonds=1, angles=1, dihedrals=1,
                            impropers=1, all_dihedrals=0,
                            nrexcl=3, HH14=1, remove_dih=1)

    # The 'bondedtypes' section contains its line directly under it. In
    # order to match the hierarchy model of the rest of the file, the
    # iterator actually yields a subsection named '_bondedtypes'. We need
    # to read the fist line of that first virtual subsection.
    _, lines = next(section)
    line = next(lines)
    read = [int(x) for x in line.split()]

    # Fill with the defaults. The file gives the values in order so we
    # need to append the missing values from the default at the end.
    bondedtypes = _BondedTypes(*(read + list(defaults[len(read):])))

    # Make sure there is no unexpected lines in the section.
    # Come on Jonathan! There must be a more compact way of doing it.
    try:
        next(lines)
    except StopIteration:
        pass
    else:
        raise IOError('"[ bondedtypes ]" section is missformated.')
    try:
        next(section)
    except StopIteration:
        pass
    else:
        raise IOError('"[ bondedtypes ]" section is missformated.')
    return bondedtypes


def _count_hydrogens(names):
    return len([name for name in names if utils.first_alpha(name) == 'H'])


def _keep_dihedral(center, block, bondedtypes):
    if (not bondedtypes.all_dihedrals) and block.has_dihedral_around(center):
        return False
    if bondedtypes.remove_dih and block.has_improper_around(center):
        return False
    return True


def _complete_block(block, bondedtypes):
    """
    Add information from the bondedtypes section to a block.

    Generate implicit dihedral angles, and add function types to the
    interactions.
    """
    block.make_edges_from_interactions()

    # Generate missing dihedrals
    # As pdb2gmx generates all the possible dihedral angles by default,
    # RTP files are written assuming they will be generated. A RTP file
    # have some control over these dihedral angles through the bondedtypes
    # section.
    all_dihedrals = []
    had_atoms = []
    for center, dihedrals in itertools.groupby(sorted(block.guess_dihedrals(), key=_dihedral_sorted_center), _dihedral_sorted_center):
        #print(center)
        if _keep_dihedral(center, block, bondedtypes):
            # TODO: Also sort the dihedrals by index.
            # See src/gromacs/gmxpreprocess/gen_add.cpp::dcomp in the
            # Gromacs source code (see version 2016.3 for instance).
     #       atoms = sorted(dihedrals, key=_count_hydrogens)[0]
            for atoms in dihedrals:
                if atoms[::-1] not in had_atoms:
                    all_dihedrals.append(Interaction(atoms=atoms, parameters=[], meta={}))
                    had_atoms.append(atoms)
    # TODO: Sort the dihedrals by index
    block.interactions['dihedrals'] = (
        block.interactions.get('dihedrals', []) + all_dihedrals
    )

    # Add all the angles
    all_angles = []
    had_angles = []
    for atoms in block.guess_angles():
        if frozenset(atoms) not in had_angles:
            all_angles.append(Interaction(atoms=atoms, parameters=[], meta={}))
            had_angles.append(frozenset(atoms))

    # TODO: Sort the dihedrals by index
    block.interactions['angles'] = (block.interactions.get('angles', []) + all_angles)

    # TODO: generate 1-4 interactions between all atoms and keep pairs of hydrogen atoms
    # if HH14 is set to 1
    pairs = []
    for node in block.nodes:
        paths = nx.single_source_shortest_path(G=block, source=node, cutoff=4)
        neighbours = [node for node, path in paths.items() if 4 == len(path)]
        for ndx in neighbours:
             if frozenset([ndx, node]) not in pairs:
                block.interactions['pairs'].append(Interaction(atoms=(node, ndx), parameters=[], meta={}))
                pairs.append(frozenset([ndx, node]))

    # Add function types to the interaction parameters. This is done as a
    # post processing step to cluster as much interaction specific code
    # into this method.
    # I am not sure the function type can be set explicitly in the RTP
    # file except through the bondedtypes section. If it is possible, then the
    # following code can break and atoms can have the function type written
    # twice. Yet, none of the RTP files distributed with Gromacs 2016.3 causes
    # issue.
    functypes = {
        'bonds': str(bondedtypes.bonds),
        'angles': str(bondedtypes.angles),
        'dihedrals': str(bondedtypes.dihedrals),
        'impropers': str(bondedtypes.impropers),
        'exclusions': '1',
        'cmap': '1',
        'pairs': '1',
    }
    for name, interactions in block.interactions.items():
        for interaction in interactions:
            interaction.parameters.insert(0, functypes[name])

    # Set the nrexcl to the block.
    block.nrexcl = bondedtypes.nrexcl


def _split_blocks_and_links(pre_blocks):
    """
    Split all the pre-blocks from `pre_block` into blocks and links.

    Parameters
    ----------
    pre_blocks: dict
        A dict with residue names as keys and instances of :class:`Block`
        as values.

    Returns
    -------
    blocks: dict
        A dict like `pre_block` with all inter-residues information
        stripped out.
    links: list
        A list of instances of :class:`Link` containing the inter-residues
        information from `pre_blocks`.

    See Also
    --------
    _split_block_and_link
        Split an individual pre-block into a block and a link.
    """
    blocks = {}
    links = []
    for name, pre_block in pre_blocks.items():
        block, link = _split_block_and_link(pre_block)
        blocks[name] = block
        links.append(link)
    return blocks, links


def _split_block_and_link(pre_block):
    """
    Split `pre_block` into a block and a link.

    A pre-block is a block as read from an RTP file. It contains both
    intra-residue and inter-residues information. This method split this
    information so that the intra-residue interactions are put in a block
    and the inter-residues ones are put in a link.

    Parameters
    ----------
    pre_block: Block
        The block to split.

    Returns
    -------
    block: Block
        All the intra-residue information.
    link: Link
        All the inter-residues information.
    """
    block = Block(force_field=pre_block.force_field)
    link = Link()

    # It is easier to fill the interactions using a defaultdict,
    # yet defaultdicts are more annoying when reading as querying them
    # creates the keys. So the interactions are revert back to regular
    # dict at the end.
    block.interactions = collections.defaultdict(list)
    link.interactions = collections.defaultdict(list)

    block.name = pre_block.name
    try:
        block.nrexcl = pre_block.nrexcl
    except AttributeError:
        pass

    # Filter the particles from neighboring residues out of the block.
    for atom in pre_block.atoms:
        if not atom['atomname'].startswith('+-'):
            atom['resname'] = pre_block.name
            block.add_atom(atom)
        link.add_node(atom['atomname'])

    # Create the edges of the link and block based on the edges in the pre-block.
    # This will create too many edges in the link, but the useless ones will be
    # pruned latter.
    link.add_edges_from(pre_block.edges)
    block.add_edges_from(edge for edge in pre_block.edges
                         if not any(node[0] in '+-' for node in edge))

    # Split the interactions from the pre-block between the block (for
    # intra-residue interactions) and the link (for inter-residues ones).
    # The "relevant_atoms" set keeps track of what particles are
    # involved in the link. This will allow to prune the link without
    # iterating again through its interactions.
    relevant_atoms = set()
    for name, interactions in pre_block.interactions.items():
        for interaction in interactions:
            for_link = any(atom[0] in '+-' for atom in interaction.atoms)
            if for_link:
                link.interactions[name].append(interaction)
                relevant_atoms.update(interaction.atoms)
            else:
                block.interactions[name].append(interaction)

    # Prune the link to keep only the edges and particles that are
    # relevant.
    nodes = set(link.nodes())
    link.remove_nodes_from(nodes - relevant_atoms)
    # Some interactions do not generate nodes (impropers for instance). If a
    # node is only described in one of such interactions, then the node never
    # appears in the link. Here we make sure these nodes exists even if they
    # are note connected.
    link.add_nodes_from(relevant_atoms)

    # Atoms from a links are matched against a molecule based on its node
    # attributes. The name is a primary criterion, but other criteria can be
    # the residue name (`resname` key) or the residue order in the chain
    # (`order` key). The residue order is 0 when refering to the current
    # residue, +int to refer to residues after the current one in the sequence,
    # -int to refer to a previous residue in the sequence, and '*' for any
    # residue but the current one.
    # RTP files convey the order by prefixing the names with + or -. We need to
    # get rid of these prefixes.
    order = {'+': +1, '-': -1}
    relabel_mapping = {}
    for idx, node in enumerate(link.nodes()):
        atomname = node
        if node[0] in '+-':
            link.nodes[node]['order'] = order[node[0]]
            atomname = atomname[1:]
        else:
            link.nodes[node]['order'] = 0
            link.nodes[node]['resname'] = block.name
        link.nodes[node]['atomname'] = atomname
        relabel_mapping[node] = idx
    nx.relabel_nodes(link, relabel_mapping, copy=False)

    # By relabelling the nodes, we lost the relations between interactions and
    # nodes, so we need to relabel the atoms in the interactions
    new_interactions = collections.defaultdict(list)
    for name, interactions in link.interactions.items():
        for interaction in interactions:
            atoms = tuple(relabel_mapping[atom] for atom in interaction.atoms)
            new_interactions[name].append(Interaction(
                atoms=atoms,
                parameters=interaction.parameters,
                meta=interaction.meta
            ))
    link.interactions = new_interactions


    # Revert the interactions back to regular dicts to avoid creating
    # keys when querying them.
    block.interactions = dict(block.interactions)
    link.interactions = dict(link.interactions)

    return block, link


def _clean_lines(lines):
    # TODO: merge continuation lines
    for line in lines:
        splitted = line.split(';', 1)
        if splitted[0].strip():
            yield splitted[0]


def _dihedral_sorted_center(atoms):
    #return sorted(atoms[1:-1])
    return atoms[1:-1]

def _angle_sorted_center(atoms):
    return atoms[1]

def read_rtp(lines, force_field):
    """
    Read blocks and links from a Gromacs RTP file to populate a force field

    Parameters
    ----------
    lines: collections.abc.Iterator
        An iterator over the lines of a RTP file (e.g. a file handle, or a
        list of string).
    force_field: vermouth.forcefield.ForceField
        The force field to populate in place.

    Raises
    ------
    IOError
        Something in the file could not be parsed.
    """
    _subsection_parsers = {
        'atoms': _atoms,
        'bonds': _base_rtp_parser('bonds', natoms=2),
        'angles': _base_rtp_parser('angles', natoms=3),
        'impropers': _base_rtp_parser('impropers', natoms=4),
        'cmap': _base_rtp_parser('cmap', natoms=5),
    }
    # An RTP file contains both the blocks and the links.
    # We first read everything in "pre-blocks"; then we separate the
    # blocks from the links.
    pre_blocks = {}
    bondedtypes = None
    cleaned = _clean_lines(lines)
    for section_name, section in _IterRTPSections(cleaned):
        if section_name == 'bondedtypes':
            bondedtypes = _parse_bondedtypes(section)
            continue
        block = Block(force_field=force_field)
        pre_blocks[section_name] = block
        block.name = section_name

        for subsection_name, subsection in section:
            if subsection_name in _subsection_parsers:
                _subsection_parsers[subsection_name](subsection, block)

    # Pre-blocks only contain the interactions that are explicitly
    # written in the file. Some are incomplete (missing implicit defaults)
    # or must be built from the "bondedtypes" rules.
    for pre_block in pre_blocks.values():
        _complete_block(pre_block, bondedtypes)

    # At this point, the pre-blocks contain both the intra- and
    # inter-residues information. We need to split the pre-blocks into
    # blocks and links.
    blocks, links = _split_blocks_and_links(pre_blocks)
    force_field.blocks.update(blocks)
    force_field.links.extend(links)
