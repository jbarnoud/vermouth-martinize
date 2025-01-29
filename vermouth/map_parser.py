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
Contains the Mapping object and the associated parser.
"""
from collections import defaultdict
from functools import partial

from .ffinput import _tokenize, _parse_atom_attributes
from .graph_utils import MappingGraphMatcher
from .log_helpers import StyleAdapter, get_logger
from .molecule import Block
from .parser_utils import SectionLineParser


LOGGER = StyleAdapter(get_logger(__name__))


class Mapping:
    """
    A mapping object that describes a mapping from one resolution to another.

    Attributes
    ----------
    block_from: networkx.Graph
        The graph which this :class:`Mapping` object can transform.
    block_to: vermouth.molecule.Block
        The :class:`vermouth.molecule.Block` we can transform to.
    references: collections.abc.Mapping
        A mapping of node keys in :attr:`block_to` to node keys in
        :attr:`block_from` that describes which node in blocks_from should be
        taken as a reference when determining node attributes for nodes in
        block_to.
    ff_from: vermouth.forcefield.ForceField
        The forcefield of :attr:`block_from`.
    ff_to: vermouth.forcefield.ForceField
        The forcefield of :attr:`block_to`.
    names: tuple[str]
        The names of the mapped blocks.
    mapping: dict[collections.abc.Hashable, dict[collections.abc.Hashable, float]]
        The actual mapping that describes for every node key in
        :attr:`block_from` to what node key in :attr:`block_to` it
        contributes to with what weight.
        ``{node_from: {node_to: weight, ...}, ...}``.

    Note
    ----
    Only nodes described in :attr:`mapping` will be used.

    Parameters
    ----------
    block_from: networkx.Graph
        As per :attr:`block_from`.
    block_to: vermouth.molecule.Block
        As per :attr:`block_to`.
    mapping: dict[collections.abc.Hashable, dict[collections.abc.Hashable, float]]
        As per :attr:`mapping`.
    references: collections.abc.Mapping
        As per :attr:`references`.
    ff_from: vermouth.forcefield.ForceField
        As per :attr:`ff_from`.
    ff_to: vermouth.forcefield.ForceField
        As per :attr:`ff_to`.
    extra: tuple
        Extra information to be attached to :attr:`block_to`.
    normalize_weights: bool
        Whether the weights should be normalized such that the sum of the
        weights of nodes mapping to something is 1.
    names: tuple
        As per :attr:`names`.
    """
    def __init__(self, block_from, block_to, mapping, references,
                 ff_from=None, ff_to=None, extra=(), normalize_weights=False,
                 type='block', names=tuple()):
        self.block_from = block_from.copy()
        self.block_to = block_to.copy()
        self.block_to.extra = extra
        self.block_from.name = names
        self.block_to.name = names
        self.references = references
        self.ff_from = ff_from
        self.ff_to = ff_to
        self.names = names
        self.mapping = mapping
        self.type = type
        # Remove nodes not mapped from blocks_from
        unmapped = set(self.block_from.nodes.keys()) - set(self.mapping.keys())
        self.block_from.remove_nodes_from(unmapped)

        # Normalize the weights
        if normalize_weights:
            self._normalize_weights()

    @property
    def reverse_mapping(self):
        """
        The reverse of :attr:`mapping`.
        ``{node_to: {node_from: weight, ...}, ...}``
        """
        rev_mapping = defaultdict(dict)  # {idx to: {idx from: weight}}
        for idx_from in self.mapping:
            for idx_to, weight in self.mapping[idx_from].items():
                rev_mapping[idx_to][idx_from] = weight
        return dict(rev_mapping)

    def map(self, graph, node_match=None, edge_match=None):
        """
        Performs the partial mapping described by this object on `graph`. It
        first find the induced subgraph isomorphisms between `graph` and
        :attr:`block_from`, after which it will process the found isomorphisms
        according to :attr:`mapping`.

        None of the yielded dictionaries will refer to node keys of
        :attr:`block_from`. Instead, those will be translated to node keys of
        `graph` based on the found isomorphisms.

        Note
        ----
        Only nodes described in :attr:`mapping` will be used in the
        isomorphism.

        Parameters
        ----------
        graph: networkx.Graph
            The graph on which this partial mapping should be applied.
        node_match: collections.abc.Callable or None
            A function that should take two dictionaries with node attributes,
            and return `True` if those nodes should be considered equal, and
            `False` otherwise. If None, all nodes will be considered equal.
        edge_match: collections.abc.Callable or None
            A function that should take six arguments: two graphs, and four
            node keys. The first two node keys will be in the first graph and
            share an edge; and the last two node keys will be in the second
            graph and share an edge. Should return `True` if a pair of edges
            should be considered equal, and `False` otherwise. If None, all
            edges will be considered equal.

        Yields
        ------
        dict[collections.abc.Hashable, dict[collections.abc.Hashable, float]]
            the correspondence between nodes in `graph` and nodes in
            :attr:`block_to`, with the associated weights.
        vermouth.molecule.Block
            :attr:`block_to`.
        dict
            :attr:`references` on which :attr:`mapping` has been applied.
        """
        if edge_match is not None:
            edge_match = partial(edge_match, graph, self.block_from)

        return self._graph_map(graph, node_match, edge_match)

    def _graph_map(self, graph, node_match, edge_match):
        """
        Performs the partial mapping described in :meth:`map` using a
        "classical" subgraph isomorphism algorithm.
        """
        # 1 Find subgraph isomorphism between blocks_from and graph
        # 2 Translate found match ({graph idx: blocks from idx}) to indices in
        #   block_to using self.mapping
        # 3 Return found matches and block_to?
        # PS. I don't really like this, because this object is becoming too
        # intelligent by also having to do the isomorphism. On the other hand,
        # it makes sense from the maths point of view.
        matcher = MappingGraphMatcher(graph, self.block_from,
                                      node_match=node_match,
                                      edge_match=edge_match)
        for match in matcher.subgraph_isomorphisms_iter():
            rev_match = {v: k for k, v in match.items()}

            new_match = defaultdict(dict)
            for graph_idx, from_idx in match.items():
                new_match[graph_idx].update(self.mapping[from_idx])
            references = {out_idx: rev_match[ref_idx]
                          for out_idx, ref_idx in self.references.items()}
            yield dict(new_match), self.block_to, references

    def _normalize_weights(self):
        """
        Normalize weights in :attr:`mapping` such that the sum of the weights
        of nodes mapping to something is 1.
        """
        rev_mapping = self.reverse_mapping
        for idx_from in self.mapping:
            for idx_to, weight in self.mapping[idx_from].items():
                self.mapping[idx_from][idx_to] = weight/sum(rev_mapping[idx_to].values())


class MappingBuilder:
    """
    An object that is in charge of building the arguments needed to create a
    :class:`Mapping` object. It's attributes describe the information
    accumulated so far.

    Attributes
    ----------
    mapping: collections.defaultdict
    blocks_from: None or vermouth.molecule.Block
    blocks_to: None or vermouth.molecule.Block
    ff_from: None or vermouth.forcefield.ForceField
    ff_to: None or vermouth.forcefield.ForceField
    names: list
    references: dict
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the object to a clean initial state.
        """
        self.mapping = defaultdict(dict)
        self.blocks_from = Block()
        self.blocks_to = Block()
        self.ff_from = None
        self.ff_to = None
        self.names = []
        self.references = {}

    def to_ff(self, ff_name):
        """
        Sets :attr:`ff_to`

        Parameters
        ----------
        ff_name
        """
        self.ff_to = ff_name

    def from_ff(self, ff_name):
        """
        Sets :attr:`ff_from`

        Parameters
        ----------
        ff_name
        """
        self.ff_from = ff_name

    @staticmethod
    def _add_block(current_block, new_block):
        """
        Helper method that adds `new_block` to `current_block`, if the latter
        is not `None`. Otherwise, create a new block from `new_block`.

        Parameters
        ----------
        current_block: None or vermouth.molecule.Block
        new_block: vermouth.molecule.Block

        Returns
        -------
        vermouth.molecule.Block
            The combination of `current_block` and `new_block`
        """
        if not current_block:
            current_block = new_block.to_molecule(default_attributes={})
        else:
            current_block.merge_molecule(new_block)
        return current_block

    def add_block_from(self, block):
        """
        Add a block to :attr:`blocks_from`. In addition, apply any 'replace'
        operation described by nodes on themselves::

            {'atomname': 'C', 'charge': 0, 'replace': {'charge': -1}}

        becomes::

            {'atomname': 'C', 'charge': -1}

        Parameters
        ----------
        block: vermouth.molecule.Block
            The block to add.
        """
        self.blocks_from = self._add_block(self.blocks_from, block)

    def add_name(self, name):
        """
        Add a name to the mapping.

        Parameters
        ----------
        name: str
            The name to add
        """
        self.names.append(name)

    def add_block_to(self, block):
        """
        Add a block to :attr:`blocks_to`.

        Parameters
        ----------
        block: vermouth.molecule.Block
            The block to add.
        """
        self.blocks_to = self._add_block(self.blocks_to, block)

    def add_node_from(self, attrs):
        """
        Add a single node to :attr:`blocks_from`.

        Parameters
        ----------
        attrs: dict[str]
            The attributes the new node should have.
        """
        if not self.blocks_from:
            self.blocks_from = Block()
            idx = 0
        else:
            idx = max(self.blocks_from.nodes) + 1
        self.blocks_from.add_node(idx, **attrs)

    def add_node_to(self, attrs):
        """
        Add a single node to :attr:`blocks_to`.

        Parameters
        ----------
        attrs: dict[str]
            The attributes the new node should have.
        """
        if not self.blocks_to:
            self.blocks_to = Block()
            idx = 0
        else:
            idx = max(self.blocks_to.nodes) + 1
        self.blocks_to.add_node(idx, **attrs)

    def add_edge_from(self, attrs1, attrs2, edge_attrs):
        """
        Add a single edge to :attr:`blocks_from` between two nodes in
        :attr:`blocks_from` described by `attrs1` and `attrs2`. The nodes
        described should not be the same.

        Parameters
        ----------
        attrs1: dict[str]
            The attributes that uniquely describe a node in
            :attr:`blocks_from`
        attrs2: dict[str]
            The attributes that uniquely describe a node in
            :attr:`blocks_from`
        edge_attrs: dict[str]
            The attributes that should be assigned to the new edge.
        """
        nodes1 = list(self.blocks_from.find_atoms(**attrs1))
        nodes2 = list(self.blocks_from.find_atoms(**attrs2))
        assert len(nodes1) == len(nodes2) == 1
        assert nodes1 != nodes2
        self.blocks_from.add_edge(nodes1[0], nodes2[0], **edge_attrs)

    def add_edge_to(self, attrs1, attrs2, edge_attrs):
        """
        Add a single edge to :attr:`blocks_to` between two nodes in
        :attr:`blocks_to` described by `attrs1` and `attrs2`. The nodes
        described should not be the same.

        Parameters
        ----------
        attrs1: dict[str]
            The attributes that uniquely describe a node in
            :attr:`blocks_to`
        attrs2: dict[str]
            The attributes that uniquely describe a node in
            :attr:`blocks_to`
        edge_attrs: dict[str]
            The attributes that should be assigned to the new edge.
        """
        nodes1 = list(self.blocks_to.find_atoms(**attrs1))
        nodes2 = list(self.blocks_to.find_atoms(**attrs2))
        assert len(nodes1) == len(nodes2) == 1
        assert nodes1 != nodes2
        self.blocks_to.add_edge(nodes1[0], nodes2[0], **edge_attrs)

    def add_mapping(self, attrs_from, attrs_to, weight):
        """
        Add part of a mapping to :attr:`mapping`. `attrs_from` uniquely
        describes a node in :attr:`blocks_from` and `attrs_to` a node in
        :attr:`blocks_to`. Adds a mapping between those nodes with the given
        `weight`.

        Parameters
        ----------
        attrs_from: dict[str]
            The attributes that uniquely describe a node in
            :attr:`blocks_from`
        attrs_to: dict[str]
            The attributes that uniquely describe a node in
            :attr:`blocks_to`
        weight: float
            The weight associated with this partial mapping.
        """

        nodes_from = list(self.blocks_from.find_atoms(**attrs_from))
        nodes_to = list(self.blocks_to.find_atoms(**attrs_to))
        assert len(nodes_from) == len(nodes_to) == 1
        self.mapping[nodes_from[0]][nodes_to[0]] = weight

    def add_reference(self, attrs_to, attrs_from):
        """
        Add a reference to :attr:`references`.

        Parameters
        ----------
        attrs_to: dict[str]
            The attributes that uniquely describe a node in
            :attr:`blocks_to`
        attrs_from: dict[str]
            The attributes that uniquely describe a node in
            :attr:`blocks_from`
        """
        nodes_to = list(self.blocks_to.find_atoms(**attrs_to))
        assert len(nodes_to) == 1
        node_to = nodes_to[0]
        nodes_from = set(self.blocks_from.find_atoms(**attrs_from))
        mapped_nodes = {from_ for from_ in self.mapping if node_to in self.mapping[from_]}
        nodes_from = nodes_from.intersection(mapped_nodes)
        assert len(nodes_from) == 1
        self.references[node_to] = next(iter(nodes_from))

    def get_mapping(self, type):
        """
        Instantiate a :class:`Mapping` object with the information accumulated
        so far, and return it.

        Returns
        -------
        Mapping
            The mapping object made from the accumulated information.
        """
        #if self.blocks_from is None:
        #    return None
        mapping = Mapping(self.blocks_from, self.blocks_to, dict(self.mapping),
                          self.references, ff_from=self.ff_from, ff_to=self.ff_to,
                          type=type, names=tuple(self.names))
        return mapping


class MappingDirector(SectionLineParser):
    """
    A director in charge of parsing the new mapping format. It constructs a new
    :class:`Mapping` object by calling methods of it's builder (default
    :class:`MappingBuilder`) with the correct arguments.

    Parameters
    ----------
    force_fields: dict[str, ForceField]
        Dict of known force fields.
    builder: MappingBuilder

    Attributes
    ----------
    builder
        The builder used to build the :class:`Mapping` object. By default
        :class:`MappingBuilder`.
    identifiers: dict[str, dict[str]]
        All known identifiers at this point. The key is the actual identifier,
        prefixed with either "to\\_" or "from\\_", and the values are the
        associated node attributes.
    section: str
        The name of the section currently being processed.
    from_ff: str
        The name of the forcefield from which this mapping describes a
        transfomation.
    to_ff: str
        The name of the forcefield to which this mapping describes a
        transfomation.
    macros: dict[str, str]
        A dictionary of known macros.
    """
    RESNAME_NUM_SEP = '#'
    """
    The character that separates a resname from a resnumber in shorthand block
    formats.
    """
    RESIDUE_ATOM_SEP = ':'
    """The character that separates a residue identifier from an atomname."""
    COMMENT_CHAR = ';'
    """The character that starts a comment."""
    NO_FETCH_BLOCK = '!'
    """The character that specifies no block should be fetched automatically."""
    SECTION_ENDS = ['block', 'modification']

    def __init__(self, force_fields, builder=None):
        if builder is None:
            self.builder = MappingBuilder()
        else:
            self.builder = builder
        self.force_fields = force_fields
        super().__init__()
        self._reset_mapping()

    def _reset_mapping(self):
        """
        Reinitialize attributes for a new mapping.
        """
        self._current_id = {'from': None, 'to': None}
        self.ff = {'to': None, 'from': None}
        self.identifiers = {}
        self.builder.reset()

    def finalize_section(self, previous_section, ended_section):
        """
        Wraps up parsing of a single mapping.

        Parameters
        ----------
        previous_section: collections.abc.Sequence[str]
            The previously parsed section.
        ended_section: collections.abc.Iterable[str]
            The just finished sections.

        Returns
        -------
        Mapping or None
            The accumulated mapping if the mapping is complete, None otherwise.
        """
        if any(ended in self.SECTION_ENDS for ended in ended_section):
            map_type = previous_section[0]
            mapping = self.builder.get_mapping(map_type)
            self._reset_mapping()
            return mapping
        return None

    def _parse_blocks(self, line):
        """
        Helper method for parsing to_blocks and from_blocks. It parses a line
        containing either a single longhand block description, or multiple
        shorthand block descriptions.

        Parameters
        ----------
        line: str

        yields
        ------
        tuple[str, dict[str]]
            A tuple of an identifier, and it's associated attributes.
        """
        tokens = list(_tokenize(line))
        resid = 0
        if len(tokens) == 2 and tokens[1].startswith('{') and tokens[1].endswith('}'):
            # It's definitely full spec.
            identifier = tokens[0]
            attrs = _parse_atom_attributes(tokens[1])
            yield identifier, attrs
        else:  # It must be shorthand
            for identifier in tokens:
                resname, found_resid = self._parse_block_shorthand(identifier)
                if found_resid is None:
                    resid += 1
                else:
                    resid = found_resid
                if resname.startswith(self.NO_FETCH_BLOCK):
                    resname = resname[len(self.NO_FETCH_BLOCK):]
                attrs = {'resname': resname, 'resid': resid}
                yield identifier, attrs

    def _parse_block_shorthand(self, token):
        """
        Helper method for parsing block shorthand.

        Parameters
        ----------
        token: str

        Returns
        -------
        tuple[str, int]
            A tuple of a resname and a resid.
        """
        if self.RESNAME_NUM_SEP in token:
            # PO4#3 or ALA#2
            resname, resid = token.split(self.RESNAME_NUM_SEP)
            resid = int(resid)
        else:
            # ALA
            resname = token
            resid = None
        return resname, resid

    def _resolve_atom_spec(self, atom_str, prefix=None):
        """
        Helper method that, given an atom token and a prefix ("to" or "from")
        will find the associated node attributes for that atom token. It will
        either separate the given identifier and atomname and look up the
        associated node attributes in :attr:`identifiers`; or take the last
        specified identifier.

        For example, if given the `atom_str` "ALA:CA" it will look up the
        node attributes associated with the identifier "ALA" as specified in
        the (from|to) blocks section, and add them to the attribute
        `{"atomname": "CA"}`. If given the `atom_str` "CA", it will either take
        the previously used identifier if available. If not, it will check
        whether there is only one identifier defined, and use that.

        Parameters
        ----------
        atom_str: str
        prefix: str

        Returns
        -------
        dict[str]
            The node attributes that describe this atom.
        """
        if self.RESIDUE_ATOM_SEP in atom_str:
            id_, name = atom_str.split(self.RESIDUE_ATOM_SEP)
        else:
            id_, name = None, atom_str

        if id_ is None:
            options = {name[1] for name in self.identifiers if name[0] == prefix}
            if len(options) == 1:
                id_ = next(iter(options))

        if id_ is None:
            attrs = self._current_id[prefix].copy()
        else:
            attrs = self.identifiers[(prefix, id_)].copy()
            self._current_id[prefix] = self.identifiers[(prefix, id_)]

        attrs['atomname'] = name

        return attrs

    @SectionLineParser.section_parser('modification', 'to', direction='to')
    @SectionLineParser.section_parser('modification', 'from', direction='from')
    @SectionLineParser.section_parser('block', 'to', direction='to')
    @SectionLineParser.section_parser('block', 'from', direction='from')
    def _ff(self, line, lineno=0, direction=None):
        """
        Parses a "to" or "from" section and sets :attr:`ff`.
        Every line should contain a separate forcefield name.

        Parameters
        ----------
        line: str
        """
        builder_methods = {'to': self.builder.to_ff,
                           'from': self.builder.from_ff}
        self.ff[direction] = line
        builder_methods[direction](line)

    @SectionLineParser.section_parser('modification', 'from blocks',
                                      direction='from', map_type='modification')
    @SectionLineParser.section_parser('modification', 'to blocks',
                                      direction='to', map_type='modification')
    @SectionLineParser.section_parser('block', 'from blocks',
                                      direction='from', map_type='block')
    @SectionLineParser.section_parser('block', 'to blocks',
                                      direction='to', map_type='block')
    def _blocks(self, line, lineno=0, direction=None, map_type=None):
        """
        Parses a "from blocks" or "to_blocks" section and add to
        :attr:`identifiers`. Calls :method:`builder.add_block_from` and
        :method:`builder.add_block_from`.

        A blocks section can be in two formats: shorthand and longhand:
            Shorthand:
                resname_ids := <resname_id>[ <resname_ids>]
                resname_id  := <resname>[#<resid>]
                identifier  == resname_id
                resname     := string that does not contain "#"
                resid       := [0-9]+ (residue index)
            Longhand:
                full_res_specs := <full_res_spec>[\n<full_res_specs>]
                full_res_spec  := [!]<identifier> <res_attrs>
                identifier     := string that does not start with '!'
                res_attrs      := json dict
                An "!" signifies that no block should be fetched based on the
                resname automatically.

        Identifier *must* be unique for this mapping/direction.

        Examples
        --------
        Shorthand:
            ALA#1 ALA#2
            GLY
        Longhand:
            ALA1 {"resname": "ALA", "resid": 1}
            ALA2 {"resname": "ALA", "resid": 2}

        Parameters
        ----------
        line: str
        lineno: int
        direction: str
            Should be 'to' or 'from'
        map_type: str
            The type of mapping.
        """
        builder_methods = {'to': self.builder.add_block_to,
                           'from': self.builder.add_block_from}
        for identifier, attrs in self._parse_blocks(line):
            fetch = True
            if identifier.startswith(self.NO_FETCH_BLOCK):
                identifier = identifier[len(self.NO_FETCH_BLOCK):]
                fetch = False
            if fetch and attrs.get('resname') is not None:
                block = getattr(self.force_fields[self.ff[direction]], map_type+'s')[attrs['resname']]
                if map_type == 'modification':
                    # Add the modifications to the mapping block_from, so they
                    # participate in the matching criterion.

                    # Caveat emptor
                    ###############
                    # Copying the modification here before modifying it would be
                    # a *good* idea, but that doesn't work (probably because
                    # Links don't have a sane copy method). Better than this
                    # would be to add the modifications attribute at the ff
                    # parser level. This hack-around (and adding it to the ff
                    # parser) probably have unintended side effects.
                    # block = block.copy()
                    for idx in block.nodes:
                        block.nodes[idx]['modifications'] = [block]
                builder_methods[direction](block)
            if direction == 'from':
                if 'resname' in attrs and isinstance(attrs['resname'], str):
                    name = attrs.get('resname')
                else:
                    name = identifier
                self.builder.add_name(name)
            if map_type == 'modification' and 'resname' in attrs:
                del attrs['resname']
            self.identifiers[(direction, identifier)] = attrs

    @SectionLineParser.section_parser('modification', 'from nodes', direction='from')
    @SectionLineParser.section_parser('modification', 'to nodes', direction='to')
    @SectionLineParser.section_parser('block', 'from nodes', direction='from')
    @SectionLineParser.section_parser('block', 'to nodes', direction='to')
    def _nodes(self, line, lineno=0, direction=None):

        """
        Parses a "from nodes" or "to nodes" section. Calls
        :method:`builder.add_node_from` and :method:`builder.add_node_to`.

        atoms      := <atom>[\n<atoms>]
        atom       := [<identifier>:]<atomname>[ <atom_attrs>]
        atomname   := string that does not contain ':'
        atom_attrs := json dict

        If no identifier is specified, the previous one used is taken. If none
        was used before and there is only one option, that is taken. Will add a
        new node with as attributes the union of the attributes associated with
        the identifier and the atom attributes specified. Atom attributes take
        precedence.

        Examples
        --------
        AA1:N {"resname": "ALA"}
        HN
        AA2:CA
        HA {"resid": 2}

        Parameters
        ----------
        line: str
        lineno: int
        direction: str
            Should be 'to' or 'from'
        """
        builder_methods = {'to': self.builder.add_node_to,
                           'from': self.builder.add_node_from}
        name, *new_attrs = _tokenize(line)
        attrs = self._resolve_atom_spec(name, direction)
        if new_attrs:
            new_attrs = _parse_atom_attributes(*new_attrs)
        else:
            new_attrs = {}
        attrs.update(new_attrs)
        builder_methods[direction](attrs)

    @SectionLineParser.section_parser('modification', 'from edges', direction='from')
    @SectionLineParser.section_parser('modification', 'to edges', direction='to')
    @SectionLineParser.section_parser('block', 'from edges', direction='from')
    @SectionLineParser.section_parser('block', 'to edges', direction='to')
    def _edges(self, line, lineno=0, direction=None):
        """
        Parses a "from edges" or "to edges" section. Calls
        :method:`builder.add_edge_from` and :method:`builder.add_edge_to`.

        edges      := <edge>[\n<edges>]
        edge       := <atom> +<atom>[ <edge_attrs>]
        atom       := [<identifier>:]<atomname>
        edge_attrs := json dict

        The combination of attributes defined by the identifier + the specified
        atom name *must* resolve to a unique node in the block.

        Example
        -------
        ALA#1:C ALA#2:N

        Parameters
        ----------
        line: str
        lineno: int
        direction: str
            Should be 'to' or 'from'
        """
        builder_methods = {'from': self.builder.add_edge_from,
                           'to': self.builder.add_edge_to}
        at1, at2, *attrs = _tokenize(line)
        attrs1 = self._resolve_atom_spec(at1, direction)
        attrs2 = self._resolve_atom_spec(at2, direction)
        if attrs:
            attrs = _parse_atom_attributes(*attrs)
        else:
            attrs = {}
        builder_methods[direction](attrs1, attrs2, attrs)

    @SectionLineParser.section_parser('modification', 'mapping')
    @SectionLineParser.section_parser('block', 'mapping')
    def _mapping(self, line, lineno=0):
        """
        Parses a "mapping" section. Calls :method:`builder.add_mapping`.

        mappings := <mapping>[\n<mappings>]
        weight   := float | int
        mapping  := <atom from> <atom to> <weight>

        <atom from> and <atom to> have the same definition as <atom> in the
        "edges" section. See :meth:`_edges`.

        Examples
        --------
        ALA:C    ALA:BB
        H        BB  0

        Parameters
        ----------
        line: str
        lineno: int
        """

        from_, to_, *weight = line.split()
        if weight:
            weight = int(weight[0])
        else:
            weight = 1

        attrs_from = self._resolve_atom_spec(from_, 'from')
        attrs_to = self._resolve_atom_spec(to_, 'to')

        self.builder.add_mapping(attrs_from, attrs_to, weight)

    @SectionLineParser.section_parser('modification', 'reference atoms')
    @SectionLineParser.section_parser('block', 'reference atoms')
    def _reference_atoms(self, line, lineno=0):
        """
        Parses a "reference atom" section. Calls
        :method:`builder.add_reference`.

        reference := <atom to> <atom from>

        Parameters
        ----------
        line: str
        lineno: int
        """
        to_, from_ = line.split()
        attrs_to = self._resolve_atom_spec(to_, 'to')
        attrs_from = self._resolve_atom_spec(from_, 'from')
        self.builder.add_reference(attrs_to, attrs_from)

    @SectionLineParser.section_parser('molecule')
    def _molecule(self, line, lineno=0):
        raise IOError("It look like you're trying to parse an old style "
                      "backmapping file. These some minor modifications to be "
                      "used compared to the files used by `backwards.py`.")


def parse_mapping_file(filepath, force_fields):
    """
    Parses a mapping file.

    Parameters
    ----------
    filepath: str
        The path of the file to parse.
    force_fields: dict[str, ForceField]
        Dict of known forcefields

    Returns
    -------
    list[Mapping]
        A list of all mappings described in the file.
    """
    with open(filepath) as map_in:
        director = MappingDirector(force_fields)
        mappings = list(director.parse(map_in))
    return mappings
