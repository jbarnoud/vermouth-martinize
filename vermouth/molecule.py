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

from collections import defaultdict, OrderedDict, namedtuple
import copy
from functools import partial
import itertools
import re

import networkx as nx
import numpy as np

from . import graph_utils
from . import geometry
from . import utils


Interaction = namedtuple('Interaction', 'atoms parameters meta')
DeleteInteraction = namedtuple('DeleteInteraction',
                               'atoms atom_attrs parameters meta')


class LinkPredicate:
    """
    Comparison criteria for node and molecule attributes in links.

    When comparing an attribute from a link to a corresponding attribute from
    a molecule or a molecule node, the default behavior is to use the equality
    as criterion for the correspondence. Some correspondence, however must be
    broader for the link to be usable. Such alternative criteria are defined
    as link predicates.

    If an attribute in a link is set to an instance of a predicate, then the
    correspondence is defined as the boolean result of the ``match`` method.

    This is the base class for such predicate. It must be subclassed, and
    subclasses must define a :meth:`match` method that takes a dictionary and
    a potential key from that dictionary as arguments.

    Parameters
    ----------
    value:
        The per-instance value that serve as reference. How this value is
        treated depends on the subclass.
    """
    def __init__(self, value):
        self.value = value

    def match(self, node, key):
        """
        Do the comparison with the reference value.

        Notes
        -----
        This function **must** be defined by the subclasses. This docstring
        describe the *expected* format of the method.

        Parameters
        ----------
        node: dict
            A dictionary of attributes in which to look up. This can be a
            node dictionary of a molecule ``meta`` attribute.
        key:
            A potential key from the ``node`` dictionary.

        Returns
        -------
        bool
        """
        raise NotImplementedError

    def __eq__(self, other):
        # Should maybe be:
        # return (isinstance(other, self.__class__) or isinstance(self, other.__class__))\
        #        and self.value == other.value
        return other.__class__ == self.__class__ and self.value == other.value

    def __repr__(self):
        return '<{} at {:x} value={}>'.format(self.__class__.__name__, id(self), self.value)


class Choice(LinkPredicate):
    """
    Test if an attribute is defined and in a predefined list.

    Parameters
    ----------
    value: list
        The list of value in which to look for the attribute.
    """
    def match(self, node, key):
        """
        Apply the comparison.
        """
        return node.get(key) in self.value


class Regex(LinkPredicate):
    """
    Test if an attribute matches the provided regular expression.

    Parameters
    ----------
    value: re.Pattern
        The regular expression
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = re.compile(self.value)

    def match(self, node, key):
        return self.value.fullmatch(node.get(key, '<ABSENT>')) is not None


class NotDefinedOrNot(LinkPredicate):
    """
    Test if an attribute is not the reference value.

    This test passes if the attribute is not defined, if it is set to ``None``,
    or if its value is different from the reference.

    Notes
    -----
    If the reference is set to ``None``, then the test does not pass if the
    attribute is explicitly set to ``None``. It still passes if the attribute
    is not defined.

    Parameters
    ----------
    value:
        The value the attribute is tested not to be.
    """
    def match(self, node, key):
        """
        Apply the comparison.
        """
        return key not in node or node[key] != self.value


class LinkParameterEffector:
    """
    Rule to calculate an interaction parameter in a link.

    This class allows to store dynamic parameters in link interactions. The
    value of the parameter can be computed from the graph using the node keys
    given when creating the instance.

    An instance of this class is first initialized with a list of node keys
    from the link in which it is defined. The instance is latter called
    like a function, and takes as arguments a molecule and a match dictionary
    linking the link nodes with the molecule ones. The format of the dictionary
    is expected to be ``{link key: molecule key}``.

    An instance can also have a format defined. If defined, that format will be
    applied to the value computed by the :meth:`_apply` method causing the
    output to be a string. The format is given as a 'format_spec' from the
    python format string syntax. This format spec corresponds to what follows
    the column the column in string templates. For instance, formating a
    floating number to have 2 decimal places will be obtained by setting format
    to `.2f`. If no format is defined, then the calculated value is not
    modified.

    This is a base class; it needs to be subclassed. A subclass must define an
    :meth:`_apply` method that takes a molecule and a list of node keys from
    that molecule as arguments. This method is not called directly by the user,
    instead it is called by the :meth:`__call__` method when the user calls the
    instance as a function. A subclass can also set the :attr:`n_keys_asked`
    class attribute to the number of required keys. If the
    attribute is set, then the number of keys provided when initializing a new
    instance will be validated against that number; else, the user can pass an
    arbitrary number of keys without validation.

    .. automethod:: __call__
    .. automethod:: _apply
    """
    n_keys_asked = None
    """Class attribute describing the number of keys required."""

    def __init__(self, keys, format_spec=None):
        """
        Parameters
        ----------
        keys: list
            A list of node keys from the link. If the :attr:`n_keys_asked`
            class argument is set, the number of keys must correspond to the
            value of the attribute.
        format_spec: str
            Format specification.

        Raises
        ------
        ValueError
            Raised if the :attr:`n_keys_asked` class attribute is set and the
            number of keys does not correspond.
        """
        self.keys = keys
        if self.n_keys_asked is not None and len(self.keys) != self.n_keys_asked:
            raise ValueError(
                'Unexpected number of keys provided in {}: '
                '{} were expected, but {} were provided.'
                .format(self.__class__.__name__, self.n_keys_asked, len(keys))
            )
        self.format = format_spec

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.keys == other.keys
                and self.format == other.format)

    def __call__(self, molecule, match):
        """
        Parameters
        ----------
        molecule: Molecule
            The molecule from which to calculate the parameter value.
        match: dict
            The correspondence between the nodes from the link (keys), and the
            nodes from the molecule (values).

        Returns
        -------
        float:
            The calculated parameter value, formatted if required.
        """
        keys = [match[key] for key in self.keys]
        result = self._apply(molecule, keys)
        if self.format is not None:
            result = '{value:{format}}'.format(value=result, format=self.format)
        return result

    def _apply(self, molecule, keys):
        """
        Calculate the parameter value from the molecule.

        Notes
        -----
        This method **must** be defined in a subclass.

        Parameters
        ----------
        molecule: Molecule
            The molecule from which to compute the parameter value.
        keys: list
            A list of keys to use from the molecule.

        Returns
        -------
        float:
            The value for the parameter.
        """
        msg = 'The method need to be implemented by the children class.'
        raise NotImplementedError(msg)


class ParamDistance(LinkParameterEffector):
    """
    Calculate the distance between a pair of nodes.
    """
    n_keys_asked = 2

    def _apply(self, molecule, keys):
        # This will raise a ValueError if an atom is missing, or if an
        # atom does not have position.
        positions = np.stack([molecule.nodes[key]['position'] for key in keys])
        # We assume there are two rows; which we can since we checked earlier
        # that exactly two atom keys were passed.
        distance = np.sqrt(np.sum(np.diff(positions, axis=0)**2))
        return distance


class ParamAngle(LinkParameterEffector):
    """
    Calculate the angle in degrees between three consecutive nodes.
    """
    n_keys_asked = 3

    @staticmethod
    def _apply(molecule, keys):
        # This will raise a ValueError if an atom is missing, or if an
        # atom does not have position.
        positions = np.stack([molecule.nodes[key]['position'] for key in keys])
        vector_ba = positions[0, :] - positions[1, :]
        vector_bc = positions[2, :] - positions[1, :]
        angle = geometry.angle(vector_ba, vector_bc)
        return np.degrees(angle)


class ParamDihedral(LinkParameterEffector):
    """
    Calculate the dihedral angle in degrees defined by four nodes.
    """
    n_keys_asked = 4

    @staticmethod
    def _apply(molecule, keys):
        # This will raise a ValueError if an atom is missing, or if an
        # atom does not have position.
        positions = np.stack([molecule.nodes[key]['position'] for key in keys])
        angle = geometry.dihedral(positions)
        return np.degrees(angle)


class ParamDihedralPhase(LinkParameterEffector):
    """
    Calculate the dihedral angle in degrees defined by four nodes shifted by
    -180 degrees.
    """
    n_keys_asked = 4

    @staticmethod
    def _apply(molecule, keys):
        # This will raise a ValueError if an atom is missing, or if an
        # atom does not have position.
        positions = np.stack([molecule.nodes[key]['position'] for key in keys])
        angle = geometry.dihedral_phase(positions)
        return np.degrees(angle)


class Molecule(nx.Graph):
    """
    Represents a molecule as per a specific force field. Consists of atoms
    (nodes), bonds (edges) and interactions such as angle potentials.

    Two molecules are equal if:

    * the exclusion distance (nrexcl) are equal
    * the force fields are equal (but may be different instances)
    * the nodes are equal and in the same order
    * the edges are equal (but order is not accounted for)
    * the interactions are the same and in the same order within an interaction
      type

    When comparing molecules, the order of the nodes is considered as it
    determines in what order atoms will be written in the output. Same goes for
    the interactions within an interaction type. The order of edges is not
    guaranteed anywhere in the code, and they are not writen in the output.

    Attributes
    ----------
    meta: dict
    nrexcl: int
    interactions: dict[str, list[Interaction]]
        All the known interactions. Each item of the dictionary is a type of
        interaction, with the key being the name of the kind of interaction
        using Gromacs itp/rtp conventions ('bonds', 'angles', ...) and the
        value being a list of all the interactions of that type in the residue.
        An interaction is a dict with a key 'atoms' under which is stored the
        list of the atoms involved (referred by their name), a key 'parameters'
        under which is stored an arbitrary list of non-atom parameters as
        written in a RTP file, and arbitrary keys to store custom metadata. A
        given interaction can have a comment under the key 'comment'.
    citations: set[str]
        The citation keys associated with this molecule.
    """
    # As the particles are stored as nodes, we want the nodes to stay
    # ordered.
    node_dict_factory = OrderedDict

    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], Molecule):
            super().__init__(*args, **kwargs)
            other = args[0]
            self.meta = copy.deepcopy(other.meta)
            self._force_field = other._force_field
            self.nrexcl = other.nrexcl
            self.interactions = copy.deepcopy(other.interactions)
            self.citations = copy.deepcopy(other.citations)
            self.log_entries = copy.deepcopy(other.log_entries)
            self.max_node = other.max_node
            self.box = other.box
            return
        self.meta = kwargs.pop('meta', {})
        self._force_field = kwargs.pop('force_field', None)
        self.nrexcl = kwargs.pop('nrexcl', None)
        super().__init__(*args, **kwargs)
        self.interactions = defaultdict(list)
        self.citations = set()
        # {loglevel: {entry: [fmt_args]}}
        self.log_entries = defaultdict(lambda: defaultdict(list))
        self.max_node = None
        # box of the system the molecule is in
        self.box = None

    def __eq__(self, other):
        return (
            self.nrexcl == other.nrexcl and
            self._force_field == other._force_field and
            self.same_nodes(other) and
            self.same_edges(other) and
            self.same_interactions(other)
        )

    @staticmethod
    def sort_interactions(all_interactions):
        """
        Returns keys in interactions sorted by (number_of_atoms, name). Keys
        with no interactions are skipped.
        """
        sort_keys = {}
        for interaction_type, interactions in all_interactions.items():
            if not interactions:
                continue
            sort_keys[interaction_type] = len(interactions[0].atoms), interaction_type
        return sorted(sort_keys, key=lambda k: sort_keys[k])

    def __str__(self):
        moltype = self.meta.get('moltype', 'molecule')

        interaction_count = OrderedDict()
        # Make sure atoms and edges get sorted first.
        interaction_count['atoms'] = len(self.nodes)
        if len(self.interactions.get('bonds', [])) != len(self.edges):
            interaction_count['edges'] = len(self.edges)

        for itype in self.sort_interactions(self.interactions):
            interaction_count[itype] = len(self.interactions[itype])

        # interaction_count will always contain at least 'atoms'.
        last_item = interaction_count.popitem(last=True)
        out = "{} with ".format(moltype)
        out += ', '.join('{} {}'.format(number, itype)
                         for itype, number in interaction_count.items())
        if interaction_count:
            out += ', and '
        out += '{} {}'.format(last_item[1],
                              last_item[0])
        return out

    @property
    def force_field(self):
        """
        The force field the molecule is described for.

        The force field is assumed to be consistent for all the molecules of
        a system. While it is possible to reassign attribute
        `Molecule._force_field`, it is recommended to assign the force
        field at the system level as reassigning :attr:`~vermouth.system.System.force_field`
        will propagate the change to all the molecules in that system.
        """
        return self._force_field

    @property
    def atoms(self):
        """
        All atoms in this molecule. Alias for `nodes`.
        """
        # TODO: should just be an alias for nodes. If you need the attributes,
        #       do g.nodes(data=<attr>) or g.nodes(data=True)
        for node in self.nodes():
            node_attr = self.nodes[node]
            yield node, node_attr

    @property
    def sorted_nodes(self):
        yield from sorted(self.nodes, key=lambda n_idx: self.nodes[n_idx].get('atomid', np.inf))

    def copy(self):
        """
        Creates a copy of the molecule.

        Returns
        -------
        Molecule
        """
        new = self.subgraph(self.nodes)
        new.name = self.name
        new.citations = self.citations.copy()
        new.log_entries = copy.deepcopy(self.log_entries)
        return new

    def subgraph(self, nodes):
        """
        Creates a subgraph from the molecule.


        Returns
        -------
        Molecule
        """
        subgraph = self.__class__()
        subgraph.name = self.name
        subgraph.meta = copy.copy(self.meta)
        subgraph._force_field = self._force_field
        subgraph.nrexcl = self.nrexcl
        subgraph.name = self.name

        # copy citations
        subgraph.citations = self.citations
        node_copies = [(node, copy.copy(self.nodes[node])) for node in nodes]
        subgraph.add_nodes_from(node_copies)

        nodes = set(nodes)

        #edges_to_add = [
        #    (node, node2)
        #    for node in nodes
        #    for node2 in set(self[node]) & nodes
        #]
        subgraph.add_edges_from(self.edges_between(nodes, nodes, data=True))

        for interaction_type, interactions in self.interactions.items():
            for interaction in interactions:
                if all(atom in nodes for atom in interaction.atoms):
                    subgraph.interactions[interaction_type].append(interaction)

        return subgraph

    def add_interaction(self, type_, atoms, parameters, meta=None):
        """
        Add an interaction of the specified type with the specified parameters
        involving the specified atoms.

        Parameters
        ----------
        type_: str
            The type of interaction, such as 'bonds' or 'angles'.
        atoms: collections.abc.Sequence
            The atoms that are involved in this interaction. Must be in this
            molecule
        parameters: collections.abc.Iterable
            The parameters for this interaction.
        meta: collections.abc.Mapping
            Metadata for this interaction, such as comments to be written to
            the output.

        Raises
        ------
        KeyError
            If one of the atoms is not in this molecule.
        """
        if meta is None:
            meta = {}
        for atom in atoms:
            if atom not in self:
                raise KeyError('Unknown atom {}'.format(atom))
        self.interactions[type_].append(
            Interaction(atoms=tuple(atoms), parameters=parameters, meta=meta)
        )

    def add_or_replace_interaction(self, type_, atoms, parameters, meta=None, citations=None):
        """
        Adds a new interaction if it doesn't exists yet, and replaces it
        otherwise. Interactions are deemed the same if they're the same type,
        and they involve the same atoms, and their ``meta['version']`` is the
        same.

        Parameters
        ----------
        type_: str
            The type of interaction, such as 'bonds' or 'angles'.
        atoms: collections.abc.Sequence
            The atoms that are involved in this interaction. Must be in this
            molecule
        parameters: collections.abc.Iterable
            The parameters for this interaction.
        meta: collections.abc.Mapping
            Metadata for this interaction, such as comments to be written to
            the output.
        citations: set
            set of citations that apply when this link is addded to molecule

        See Also
        --------
        :meth:`add_interaction`
        """
        if meta is None:
            meta = {}
        for idx, interaction in enumerate(self.interactions[type_]):
            if (interaction.atoms == tuple(atoms)
                    and interaction.meta.get('version', 0) == meta.get('version', 0)):
                new_interaction = Interaction(
                    atoms=tuple(atoms), parameters=parameters, meta=meta,
                )
                self.interactions[type_][idx] = new_interaction
                break
        else:  # no break
            self.add_interaction(type_, atoms, parameters, meta)

        if citations:
            self.citations.update(citations)

    def get_interaction(self, type_):
        """
        Returns all interactions of `type_`

        Parameters
        ----------
        type_: collections.abc.Hashable
            The type which interactions should be found.

        Returns
        -------
        list[Interaction]
            The interactions of the specified type.
        """
        return self.interactions[type_]

    def remove_interaction(self, type_, atoms, version=0):
        """
        Removes the specified interaction.

        Parameters
        ----------
        type_: str
            The type of interaction, such as 'bonds' or 'angles'.
        atoms: collections.abc.Sequence
            The atoms that are involved in this interaction.
        version: int
            Sometimes there can be multiple distinct interactions between the
            same group of atoms. This is reflected with their `version` meta
            attribute.

        Raises
        ------
        KeyError
            If the specified interaction could not be found
        """
        idx = 0
        for idx, interaction in enumerate(self.interactions[type_]):
            if interaction.atoms == atoms and interaction.meta.get('version', 0) == version:
                break
        else:  # no break
            msg = ("Can't find interaction of type {} between atoms {} "
                   "and with version {}")
            raise KeyError(msg.format(type_, atoms, version))
        del self.interactions[type_][idx]
        if not self.interactions[type_]:
            del self.interactions[type_]

    def remove_matching_interaction(self, type_, template_interaction):
        """
        Removes any interactions that match the template.

        Parameters
        ----------
        type_: collections.abc.Hashable
            The type of interaction to look for.
        template_interaction: Interaction

        See Also
        --------
        :func:`interaction_match`
        """
        for idx, interaction in enumerate(self.interactions[type_]):
            if interaction_match(self, interaction, template_interaction):
                del self.interactions[type_][idx]
                break
        else:  # no break
            raise ValueError('Cannot find a matching interaction.')

    def find_atoms(self, **attrs):
        """
        Yields all indices of atoms that match `attrs`

        Parameters
        ----------
        **attrs: collections.abc.Mapping
            The attributes and their desired values.

        Yields
        ------
        collections.abc.Hashable
            All atom indices that match the specified `attrs`
        """
        for node_idx in self:
            node = self.nodes[node_idx]
            if attributes_match(node, attrs):
                yield node_idx

    def __getattr__(self, name):
        # TODO: DRY
        if name.startswith('get_') and name.endswith('s'):
            type_ = name[len('get_'):-len('s')]
            return partial(self.get_interaction, type_)
        elif name.startswith('add_'):
            type_ = name[len('add_'):]
            return partial(self.add_interaction, type_)
        elif name.startswith('remove_'):
            type_ = name[len('remove_'):]
            return partial(self.remove_interaction, type_)
        else:
            raise AttributeError('Unknown attribute "{}".'.format(name))

    def add_node(self, *args, **kwargs):
        super().add_node(*args, **kwargs)
        if self.max_node:
            self.max_node += 1
        else:
            self.max_node = 0

    def merge_molecule(self, molecule):
        """
        Add the atoms and the interactions of a molecule at the end of this
        one.

        Atom and residue index of the new atoms are offset to follow the last
        atom of this molecule.

        Parameters
        ----------
        molecule: Molecule
            The molecule to merge at the end.

        Returns
        -------
        dict
            A dict mapping the node indices of the added `molecule` to their
            new indices in this molecule.
        """
        if self.force_field != molecule.force_field:
            raise ValueError(
                'Cannot merge molecules with different force fields.'
            )
        if self.nrexcl is None and not self:
            self.nrexcl = molecule.nrexcl
        if self.nrexcl != molecule.nrexcl:
            raise ValueError(
                'Cannot merge molecules with different nrexcl. '
                'This molecule has nrexcl={}, while the other has nrexcl={}.'
                .format(self.nrexcl, molecule.nrexcl)
            )
        if self.nodes():
            if not self.max_node:
                # hopefully it is a small graph when this is called.
                self.max_node = max(self)

            # We assume that the last id is always the largest.
            last_node_idx = self.max_node
            offset = last_node_idx
            residue_offset = self.nodes[last_node_idx].get('resid', 1)
            offset_charge_group = self.nodes[last_node_idx].get('charge_group', 1)
        else:
            offset = 0
            residue_offset = 0
            offset_charge_group = 0
            self.max_node = 0

        correspondence = {}
        for idx, node in enumerate(molecule.nodes(), start=offset + 1):
            correspondence[node] = idx
            new_atom = copy.copy(molecule.nodes[node])
            new_atom['resid'] = (new_atom.get('resid', 1) + residue_offset)
            new_atom['charge_group'] = (new_atom.get('charge_group', 1)
                                        + offset_charge_group)
            self.add_node(idx, **new_atom)

        for name, interactions in molecule.interactions.items():
            for interaction in interactions:
                atoms = tuple(correspondence[atom] for atom in interaction.atoms)
                self.add_interaction(name, atoms, interaction.parameters, interaction.meta)
        for node1, node2 in molecule.edges:
            if correspondence[node1] != correspondence[node2]:
                attrs = molecule.edges[(node1, node2)]
                self.add_edge(correspondence[node1], correspondence[node2], **attrs)
        # merge the citation sets
        self.citations.update(molecule.citations)
        # Merge the log entries
        for loglevel, entries in molecule.log_entries.items():
            for entry, fmt_args in entries.items():
                # Renumber any existing formatting maps
                fmt_args = [{name: correspondence[old] for (name, old) in fmt_arg.items()} for fmt_arg in fmt_args]
                self.log_entries[loglevel][entry] += fmt_args + [correspondence]
        return correspondence

    def share_moltype_with(self, other):
        """
        Checks whether `other` has the same shape as this molecule.

        Parameters
        ----------
        other: Molecule

        Returns
        -------
        bool
            True iff other has the same shape as this molecule.
        """
        # Almost identical to __eq__, except that some node attributes don't
        # contribute, such as position and chain.
        # Note that isomorphic molecules get separate moltypes now, since the
        # order of the nodes is coupled to the order of the atoms in the output
        # PDB
        ignore_attrs = ('position', 'chain', 'graph', 'mapping_weights')
        return (
            self.nrexcl == other.nrexcl and
            self._force_field == other._force_field and
            self.same_nodes(other, ignore_attr=ignore_attrs) and
            self.same_edges(other) and
            self.same_interactions(other)
        )

    # TODO: Allow comparison of interactions between isomorphic molecules.
    def same_interactions(self, other):
        """
        Returns `True` if the interactions are the same.

        To be equal, two interactions must share the same node key reference,
        the same interaction parameters, and the same meta attributes. Empty
        interaction categories are ignored.

        Parameters
        ----------
        other: Molecule

        Returns
        -------
        bool
        """
        keys_self = set(
            interaction_type
            for interaction_type, interactions
            in self.interactions.items()
            if interactions
        )
        keys_other = set(
            interaction_type
            for interaction_type, interactions
            in other.interactions.items()
            if interactions
        )
        # We first make sure that the two molecules share the same relevant
        # interaction categories. A relevant category is one that actually has
        # interactions. If the molecules share the relevant categories, then we
        # can loop over the categories of one or the other, without issues with
        # mismatches.
        if keys_self != keys_other:
            return False
        return all(
            self.interactions[interaction_type] == other.interactions[interaction_type]
            for interaction_type in keys_self
        )

    # TODO: Allow default values for attributes.
    # In most cases, we assume that an unspecified attribute is equivalent to
    # the attribute being None; we should allow for this in the comparison. We
    # should also allow an attribute to have an arbitrary default value. For
    # instance, the assumed default for the `PTM_atom` attribute is False.
    def same_nodes(self, other, ignore_attr=()):
        """
        Returns `True` if the nodes are the same and in the same order.

        The equality criteria used for the attribute values are those of
        :func:`vermouth.utils.are_different`.

        Parameters
        ----------
        other: Molecule
        ignore_attr: collections.abc.Container
            Attribute keys that will not be considered in the comparison.

        Returns
        -------
        bool
        """
        # We first check that the node keys match between the two molecules.
        # The order matters here, and we count on the fact that Molecules are
        # OrderedDicts.
        if list(self.nodes.keys()) != list(other.nodes.keys()):
            return False
        zipped_nodes = zip(self.nodes.values(), other.nodes.values())
        for self_node, other_node in zipped_nodes:
            # For each pair of nodes, we compare the attribute dictionaries.
            # The order does not matter here.
            self_keys = set(key for key in self_node if key not in ignore_attr)
            other_keys = set(key for key in other_node if key not in ignore_attr)
            if self_keys != other_keys:
                return False
            # We can loop over the keys because we tested above that they were
            # matching between the two dicts.
            for key in self_keys:
                self_value = self_node[key]
                other_value = other_node[key]
                if utils.are_different(self_value, other_value):
                    return False
        return True

    def same_edges(self, other):
        """
        Compare the edges between this molecule and an other.

        Edges are unordered and undirected, but they can have attributes.

        Parameters
        ----------
        other: networkx.Graph
            The other molecule to compare the edges with.

        Returns
        -------
        bool
        """
        # The items of `graph.edges(data=True)` are formatted as
        # (from, to, {dict of attributes}).
        edges_self = {
            frozenset(edge[:2]): edge[2]
            for edge in self.edges(data=True)
        }
        edges_other = {
            frozenset(edge[:2]): edge[2]
            for edge in other.edges(data=True)
        }
        # We first start with the cheapest test: if the edges do not match
        # regardless of attributes, we can save the more costly tests.
        if set(edges_self.keys()) != set(edges_other.keys()):
            return False

        # We know that the keys in `edges_self` and `edges_other` are the same.
        return all(
            not utils.are_different(edges_self[edge], edges_other[edge])
            for edge in edges_self
        )

    def iter_residues(self):
        """
        Returns a generator over the nodes of this molecules residues.

        Returns
        -------
        collections.abc.Generator
        """
        residue_graph = graph_utils.make_residue_graph(self)
        return (tuple(residue_graph.nodes[res]['graph'].nodes) for res in sorted(residue_graph.nodes))

    def edges_between(self, n_bunch1, n_bunch2, data=False):
        """
        Returns all edges in this molecule between nodes in `n_bunch1` and
        `n_bunch2`.

        Parameters
        ----------
        n_bunch1: :class:`~collections.abc.Iterable`
            The first bunch of node indices.
        n_bunch2: :class:`~collections.abc.Iterable`
            The second bunch of node indices.

        Returns
        -------
        :class:`list`
            A list of tuples of edges in this molecule. The first element of
            the tuple will be in `n_bunch1`, the second element in `n_bunch2`.
        """
        set_1 = set(n_bunch1)
        set_2 = set(n_bunch2)
        for node1 in set_1:
            cross = set_2 & set(self[node1])
            for node2 in cross:
                if not data:
                    yield (node1, node2)
                else:
                    yield (node1, node2, self.edges[node1, node2])

    def _remove_interactions_with_node(self, node):
        """
        We iterate through the different interactions we have and
        remove the interactions where the atoms to be deleted are present.
        Further we also delete the entire interaction_type if it is
        empty after all the necessary interactions have been deleted.
        """
        for name, interactions in self.interactions.items():
            # We *must* copy interactions (list call), otherwise you change
            # interactions while iterating over it, causing it to miss
            # consecutive interactions that should be removed.
            for interaction in list(interactions):
                if node in interaction.atoms:
                    self.interactions[name].remove(interaction)

        for interaction_type in list(self.interactions):
            if not self.interactions[interaction_type]:
                self.interactions.pop(interaction_type)

    def remove_node(self, node):
        """
        Overriding the remove_node method of networkx
        as we have to delete the interaction from the interactions list
        separately which is not a part of the graph and hence does not
        get deleted.
        """
        super().remove_node(node)
        self._remove_interactions_with_node(node)

    def remove_nodes_from(self, nodes):
        """
        Overriding the remove_nodes_from method of networkx
        as we have to delete the interaction from the
        interactions list separately which is not a part of
        the graph and hence does not get deleted.
        """
        super().remove_nodes_from(nodes)
        for node in nodes:
            self._remove_interactions_with_node(node)

    def make_edges_from_interaction_type(self, type_):
        """
        Create edges from the interactions of a given type.

        The interactions must be described so that two consecutive atoms in an
        interaction should be linked by an edge. This is the case for bonds,
        angles, proper dihedral angles, and cmap torsions. It is not always
        true for improper torsions.

        Cmap are described as two consecutive proper dihedral angles. The
        atoms for the interaction are the 4 atoms of the first dihedral angle
        followed by the next atom forming the second dihedral angle with the
        3 previous ones. Each pair of consecutive atoms generate an edge.

        .. warning::

            If there is no interaction of the required type, it will be
            silently ignored.

        Parameters
        ----------
        type_: str
            The name of the interaction type the edges should be built from.
        """
        for interaction in self.interactions.get(type_, []):
            if interaction.meta.get('edge', True):
                atoms = interaction.atoms
                self.add_edges_from(zip(atoms[:-1], atoms[1:]))

    def make_edges_from_interactions(self):
        """
        Create edges from the interactions we know how to convert to edges.

        The known interactions are bonds, angles, proper dihedral angles,
        cmap torsions and constraints.
        """
        known_types = ('bonds', 'angles', 'dihedrals', 'cmap', 'constraints')
        for type_ in known_types:
            self.make_edges_from_interaction_type(type_)

class Block(Molecule):
    """
    Residue topology template

    Two blocks are equal if the underlying molecules are equal, and if the
    block names are equal.

    Parameters
    ----------
    incoming_graph_data:
        Data to initialize graph. If None (default) an empty graph is created.
    attr:
        Attributes to add to graph as key=value pairs.

    Attributes
    ----------
    name: str or None
        The name of the residue. Set to `None` if undefined.
    """
    # As the particles are stored as nodes, we want the nodes to stay
    # ordered.
    node_dict_factory = OrderedDict

    def __init__(self, incoming_graph_data=None, **attr):
        super(Block, self).__init__(incoming_graph_data, **attr)
        # Arbitrary attributes can be set during the initialization. We need
        # to set the default of some key attributes, but without overwritting
        # what has been passed in the 'attr' argument.
        defaults = {
            'name': None,
            'interactions': {},
        }
        self._set_defaults(defaults)
        self._apply_to_all_interactions = defaultdict(dict)

    def _set_defaults(self, defaults):
        for attribute, default_value in defaults.items():
            if not hasattr(self, attribute):
                setattr(self, attribute, default_value)

    def __eq__(self, other):
        return self.name == other.name and super().__eq__(other)

    def __repr__(self):
        name = self.name
        if name is None:
            name = 'Unnamed'
        return '<{} "{}" at 0x{:x}>'.format(self.__class__.__name__,
                                            name, id(self))

    def add_atom(self, atom):
        """
        Add an atom. `atom` must contain an 'atomname'. This value will be this
        atom's index.

        Parameters
        ----------
        atom: collections.abc.Mapping
            The attributes of the atom to add. Must contain 'atomname'

        Raises
        ------
        ValueError
            If `atom` does not contain 'atomname'
        """
        try:
            name = atom['atomname']
        except KeyError:
            raise ValueError('Atom has no atomname: "{}".'.format(atom))
        self.add_node(name, **atom)

    @property
    def atoms(self):
        """"
        The atoms in the residue. Each atom is a dict with *a minima* a key
        'name' for the name of the atom, and a key 'atype' for the atom type.
        An atom can also have a key 'charge', 'charge_group', 'comment', or any
        arbitrary key.

        Returns
        -------
        collections.abc.Iterator[dict]
        """
        for node in self.nodes():
            node_attr = self.nodes[node]
            # In pre-blocks, some nodes correspond to particles in neighboring
            # residues. These node do not carry particle information and should
            # not appear as particles.
            if node_attr:
                yield node_attr

    def guess_angles(self):
        """
        Generates all possible triplets of node indices that correspond to
        angles.

        Yields
        ------
        tuple[collections.abc.Hashable, collections.abc.Hashable, collections.abc.Hashable]
            All possible angles.
        """
        for a in self.nodes():
            for b in self.neighbors(a):
                for c in self.neighbors(b):
                    if c == a:
                        continue
                    yield (a, b, c)

    def guess_dihedrals(self, angles=None):
        """
        Generates all possible quadruplets of node indices that correspond to
        torsion angles.

        Parameters
        ----------
        angles: collections.abc.Iterable
            All possible angles from which to start looking for torsion angles.
            Generated from :meth:`guess_angles` if not provided.

        Yields
        ------
        tuple[collections.abc.Hashable, collections.abc.Hashable, collections.abc.Hashable, collections.abc.Hashable]
            All possible torsion angles.
        """
        angles = angles if angles is not None else self.guess_angles()
        for a, b, c in angles:
            for d in self.neighbors(c):
                if d not in (a, b):
                    yield (a, b, c, d)

    def has_dihedral_around(self, center):
        """
        Returns True if the block has a dihedral centered around the given bond.

        Parameters
        ----------
        center: tuple
            The name of the two central atoms of the dihedral angle. The
            method is sensitive to the order.

        Returns
        -------
        bool
        """
        all_centers = [tuple(dih['atoms'][1:-1])
                       for dih in self.interactions.get('dihedrals', [])]
        return tuple(center) in all_centers

    def has_improper_around(self, center):
        """
        Returns True if the block has an improper centered around the given bond.

        Parameters
        ----------
        center: tuple
            The name of the two central atoms of the improper torsion. The
            method is sensitive to the order.

        Returns
        -------
        bool
        """
        all_centers = [tuple(dih.atoms[1:-1])
                       for dih in self.interactions.get('impropers', [])]
        return tuple(center) in all_centers

    def to_molecule(self, atom_offset=0, offset_resid=0, offset_charge_group=0,
                    force_field=None, default_attributes=None):
        """
        Converts this block to a :class:`Molecule`.

        Parameters
        ----------
        atom_offset: int
            The number at which to start numbering the node indices.
        offset_resid: int
            The offset for the `resid` attributes.
        offset_charge_group: int
            The offset for the `charge_group` attributes.
        force_field: None or vermouth.forcefield.ForceField
        default_attributes: collections.abc.Mapping[str]
            Attributes to set to for nodes that are missing them.

        Returns
        -------
        Molecule
            This block as a molecule.
        """
        if force_field is None:
            force_field = self.force_field
        if default_attributes is None:
            default_attributes = {'resname': self.name}
        name_to_idx = {}
        mol = Molecule(force_field=force_field)
        mol.citations = self.citations
        mol.log_entries = copy.deepcopy(self.log_entries)

        for idx, node in enumerate(self.nodes, start=atom_offset):
            name_to_idx[node] = idx
            atom = self.nodes[node]
            new_atom = default_attributes.copy()
            new_atom.update(atom)
            new_atom['resid'] = (new_atom.get('resid', 1) + offset_resid)
            new_atom['charge_group'] = (new_atom.get('charge_group', 1) +
                                        offset_charge_group)
            mol.add_node(idx, **new_atom)
        for name, interactions in self.interactions.items():
            for interaction in interactions:
                atoms = tuple(
                    name_to_idx[atom] for atom in interaction.atoms
                )
                mol.add_interaction(
                    name, atoms,
                    interaction.parameters,
                    meta=interaction.meta
                )
        for nodea, nodeb, attrs in self.edges(data=True):
            edge = (nodea, nodeb)
            mol.add_edge(*(name_to_idx[node] for node in edge), **attrs)

        try:
            mol.nrexcl = self.nrexcl
        except AttributeError:
            pass

        return mol


class Link(Block):
    """
    Template link between two residues.

    Two links are equal if:

    * the underlying molecules are equal
    * the names are equal
    * the negative edges ("non-edges") are equal regardless of order
    * the interactions to remove are the same and in the same order
    * the meta variables are equal
    * the pattern definitions are equal and in the same order
    * the features are equals regardless of order

    A link does not match if any of the non-edges match the target; their
    order therefore is not important. Same goes for features that just need to
    be present or not. The order does matter however for interactions to remove
    as removing the interactions in a different order may lead to a different
    set of remaining interactions.

    Parameters
    ----------
    incoming_graph_data:
        Data to initialize graph. If `None` (default) an empty graph is created.
    attr:
        Attributes to add to graph as key=value pairs.
    """
    node_dict_factory = OrderedDict

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        # Arbitrary attributes can be set during the initialization. We need
        # to set the default of some key attributes, but without overwritting
        # what has been passed in the 'attr' argument.
        defaults = {
            'non_edges': [],
            'removed_interactions': {},
            'molecule_meta': {},
            'patterns': [],
            'features': set(),
        }
        self._set_defaults(defaults)
        self._apply_to_all_nodes = {}

    def __eq__(self, other):
        # TODO: There is no good reason to care about the order of patterns
        # except for the fact that order free comparison of non hashable things
        # is a pain.
        return (super().__eq__(other)
                and self.same_non_edges(other)
                and self.removed_interactions == other.removed_interactions
                and self.molecule_meta == other.molecule_meta
                and self.patterns == other.patterns
                and set(self.features) == set(other.features)
                )

    def same_non_edges(self, other):
        """
        Returns `True` if all the non-edges of an `other` link are equal to
        those of this link. Returns `False` otherwise.
        """
        # A non-edge is a list of two elements: the key to a node in the graph
        # that is used as anchor, and an attribute dict that must match none of
        # the atoms connected to the anchor.
        # For the link to match, none of the non-edges must match. Therefore,
        # their order do not matter. Though, because the attribute dicts are
        # not hashable, non-edges cannot be put in a set.

        # If the number of non-edges is not the same, we can save the hasle.
        if len(self.non_edges) != len(other.non_edges):
            return False
        # If there is no non-edges, which is likely the most common case, we
        # can also save some effort.
        if not self.non_edges:
            return True

        # We sort the non-edges to get rid of the order. However, we cannot
        # sort them on the attribute dict, and there may be more than one
        # non-edge involving a given anchor. So for each anchor, we need to
        # compare the dicts of that link with all the non already assigned
        # dicts of the other link that share the same anchor.
        sorted_self = sorted(self.non_edges, key=lambda x: x[0])
        sorted_other = sorted(other.non_edges, key=lambda x: x[0])
        zipped = zip(itertools.groupby(sorted_self, key=lambda x: x[0]),
                     itertools.groupby(sorted_other, key=lambda x: x[0]))
        for (key_self, attrs_self), (key_other, attrs_other) in zipped:
            if key_self != key_other:
                return False
            attrs_self = list(attrs_self)
            attrs_other = list(attrs_other)
            if len(attrs_self) != len(attrs_other):
                return False
            for attr_self in attrs_self:
                for idx, attr_other in enumerate(attrs_other):
                    if not utils.are_different(attr_self, attr_other):
                        del attrs_other[idx]
                        break
                else:  # no break
                    return False
        return True

class Modification(Link):
    """
    A modification which describes deviations from a :class:`Block`.
    """

def attributes_match(attributes, template_attributes, ignore_keys=()):
    """
    Compare a dict of attributes from a molecule with one from a link.

    Returns ``True`` if the attributes from the link match the ones from the
    molecule; returns ``False`` otherwise. The attributes from a link match
    with those of a molecule if all the individual attribute from the link
    match the corresponding ones in the molecule. In the simplest case, these
    attribute match if their values are equal. If the value of the link
    attribute is an instance of :class:`LinkPredicate`, then the attributes
    match if the ``match`` method of the predicate returns ``True``.

    Parameters
    ----------
    attributes: dict
        Attributes from the molecule.
    template_attributes: dict
        Attributes from the link.
    ignore_keys: list
        List of keys to ignore from 'template_attributes'.

    Returns
    -------
    bool
    """
    for attr, value in template_attributes.items():
        if attr in ignore_keys:
            continue
        if attributes.get(attr) != value:
            if isinstance(value, LinkPredicate) and value.match(attributes, attr):
                continue
            return False
    return True


def interaction_match(molecule, interaction, template_interaction):
    """
    Compare an interaction with a template interaction or interaction to delete.

    An instance of :class:`Interaction` matches a template instance of the same
    class or of :class:`DeleteInteraction` if, at the  minimum, it involves the
    same atoms in the same order. If the template defines parameters, then they
    have to match as well. In the case of of a :class:`DeleteInteraction`,
    atoms may have attributes as well, then they have to match with the
    attributes of the corresponding atoms in the molecule.

    Parameters
    ----------
    molecule: networkx.Graph
        The molecule that contains the interaction.
    interaction: Interaction
        The interaction in the molecule.
    template_interaction: Interaction or DeleteInteraction
        The template to match with the interaction.

    Returns
    -------
    bool

    See Also
    --------
    attributes_match
    """
    atoms_match = tuple(template_interaction.atoms) == tuple(interaction.atoms)
    parameters_match = (
        not template_interaction.parameters
        or tuple(template_interaction.parameters) == tuple(interaction.parameters)
    )
    if atoms_match and parameters_match:
        try:
            atom_attrs = template_interaction.atom_attrs
        except AttributeError:
            atom_attrs = [{}, ] * len(template_interaction.atoms)
        nodes = [molecule.nodes[atom] for atom in interaction.atoms]
        for atom, template_atom in zip(nodes, atom_attrs):
            if not attributes_match(atom, template_atom):
                return False
        return attributes_match(interaction.meta, template_interaction.meta)
    return False
