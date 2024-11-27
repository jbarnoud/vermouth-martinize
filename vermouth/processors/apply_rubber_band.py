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
Provides a processor that adds a rubber band elastic network.
"""
import itertools

import numpy as np
import networkx as nx
import copy

from .processor import Processor
from .. import selectors
from ..graph_utils import make_residue_graph
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))

# the bond type of the RB
DEFAULT_BOND_TYPE = 6
# the minimum distance between the resids
# of two beads to have an RB
DEFAULT_RMD = 2

def self_distance_matrix(coordinates):
    """
    Compute a distance matrix between points in a selection.

    Notes
    -----
    This function does **not** account for periodic boundary conditions.

    Parameters
    ----------
    coordinates: numpy.ndarray
        Coordinates of the points in the selection. Each row must correspond
        to a point and each column to a dimension.

    Returns
    -------
    numpy.ndarray
    """
    return np.sqrt(
        np.sum(
            (coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]) ** 2,
            axis=-1)
    )


def compute_decay(distance, shift, rate, power):
    r"""
    Compute the decay function of the force constant as function to the distance.

    The decay function for the force constant is defined as:

    .. math::

        \exp^{-r(d - s)^p}

    where :math:`r` is the decay rate given by the 'rate' argument,
    :math:`p` is the decay power given by 'power', :math:`s` is a shift
    given by 'shift', and :math:`d` is the distance between the two atoms given
    in 'distance'. If the rate or the power are set to 0, then the decay
    function does not modify the force constant.

    The 'distance' argument can be a scalar or a numpy array. If it is an
    array, then the returned value is an array of decay factors with the same
    shape as the input.
    """
    return np.exp(-rate * ((distance - shift) ** power))


def compute_force_constants(distance_matrix, lower_bound, upper_bound,
                            decay_factor, decay_power, base_constant,
                            minimum_force):
    """
    Compute the force constant of an elastic network bond.

    The force constant can be modified with a decay function, and it can be
    bounded with a minimum threshold, or a distance upper and lower bonds.
    
    If decay_factor = decay_power = 0 all forces applied are = base_constant
    
    Forces applied to distances above upper_bound are removed.
    Forces below minimum_force are removed.
    
    If decay_factor or decay_power != 0, forces below lower_bound are greater 
    than base_constant, in which case they are set back to = base_constant    
    """
    constants = compute_decay(distance_matrix, lower_bound, decay_factor, decay_power)
    np.fill_diagonal(constants, 0)
    constants *= base_constant
    constants[constants < minimum_force] = 0
    constants[constants > base_constant] = base_constant
    constants[distance_matrix > upper_bound] = 0
    return constants


def are_connected(graph, left, right, separation):
    """
    ``True`` if the nodes are at most 'separation' nodes away.

    Parameters
    ----------
    graph: networkx.Graph
        The graph/molecule to work on.
    left:
        One node key from the graph.
    right:
        One node key from the graph.
    separation: int
        The maximum number of nodes in the shortest path between two nodes of
        interest for these two nodes to be considered connected. Must be >= 0.

    Returns
    -------
    bool
    """
    nodes_are_connected = False
    try:
        shortest_path = len(nx.shortest_path(graph, left, right))
    except nx.NetworkXNoPath:
        # There is no path between left and right so they are not
        # connected; which is the default.
        pass
    else:
        # The source and the target are counted in the shortest path
        nodes_are_connected = shortest_path <= separation + 2
    return nodes_are_connected


def build_connectivity_matrix(graph, separation, node_to_idx, selected_nodes):
    """
    Build a connectivity matrix based on the separation between nodes in a graph.

    The connectivity matrix is a symmetric boolean matrix where cells contain
    ``True`` if the corresponding atoms are connected in the graph and
    separated by less or as much nodes as the given 'separation' argument.

    In the following examples, the separation between A and B is 0, 1, and 2.
    respectively:

    ```
    A - B
    A - X - B
    A - X - X - B
    ```

    Note that building the connectivity matrix with a separation of 0 is the
    same as building the adjacency matrix.

    Parameters
    ----------
    graph: networkx.Graph
        The graph/molecule to work on.
    separation: int
        The maximum number of nodes in the shortest path between two nodes of
        interest for these two nodes to be considered connected. Must be >= 0.
    selected_nodes: collections.abc.Collection
        A list of nodes to work on.

    Returns
    -------
    numpy.ndarray
        A boolean matrix.
    """
    res_graph = make_residue_graph(graph)
    distance_pairs = nx.all_pairs_shortest_path_length(res_graph, cutoff=separation)
    # only gets "positive" entries due to the cutoff argument above
    size = graph.number_of_nodes()
    # the matrix will be reduced before returning it but nx.all_pairs_shortest_path_length
    # does not take a subset of nodes
    # TODO optimize me,  try to create a sparse matrix in case scipy is available.
    connectivity = np.zeros((size, size), dtype=bool)
    for origin_residue, matchs_distances in distance_pairs:
        for target_residue in matchs_distances:
            origin_nodes = res_graph.nodes[origin_residue]['graph'].nodes()
            target_nodes = res_graph.nodes[target_residue]['graph'].nodes()
            for origin, target in itertools.product(origin_nodes, target_nodes):
                connectivity[node_to_idx[origin], node_to_idx[target]] = True
    np.fill_diagonal(connectivity, False)
    return connectivity[:, selected_nodes][selected_nodes]


def build_pair_matrix(graph, criterion, idx_to_node, selected_nodes):
    """
    Build a boolean matrix telling if a pair of nodes fulfil a criterion.

    Parameters
    ----------
    graph: networkx.Graph
        The graph/molecule to work on.
    criterion: collections.abc.Callable
        A function that determines if a pair of nodes fulfill the criterion.
        It takes a graph and two node keys as arguments and returns a boolean.
    selected_nodes: collections.abc.Collection
        A list of nodes to work on.

    Returns
    -------
    numpy.ndarray
        A boolean matrix.
    """
    size = len(graph.nodes)
    #TODO generate spare matrix with scipy
    share_domain = np.zeros((size, size), dtype=bool)
    node_combinations = itertools.combinations(selected_nodes, 2)
    for kdx, jdx in node_combinations:
        key_kdx = idx_to_node[kdx]
        key_jdx = idx_to_node[jdx]
        share_domain[kdx, jdx] = criterion(graph, key_kdx, key_jdx)
        share_domain[jdx, kdx] = share_domain[kdx, jdx]
    return share_domain[:, selected_nodes][selected_nodes]

def apply_rubber_band(molecule, selector,
                      lower_bound, upper_bound,
                      decay_factor, decay_power,
                      base_constant, minimum_force,
                      bond_type, domain_criterion, res_min_dist):
    r"""
    Adds a rubber band elastic network to a molecule.

    The elastic network is applied as bounds between the atoms selected by the
    function declared with the 'selector' argument. The equilibrium length for
    the bonds is measured from the coordinates in the molecule, the force
    constant is computed from the base force constant and an optional decay
    function.

    The decay function for the force constant is defined as:

    .. math::

        \exp^{-r(d - s)^p}

    where :math:`r` is the decay rate given by the 'decay_factor' argument,
    :math:`p` is the decay power given by 'decay_power', :math:`s` is a shift
    given by 'lower_bound', and :math:`d` is the distance between the two atoms
    in the molecule. If the rate or the power are set to 0, then the decay
    function does not modify the force constant.

    The 'selector' argument takes a callback that accepts a atom dictionary and
    returns ``True`` if the atom match the conditions to be kept.

    Only nodes that are in the same domain can be connected by the elastic
    network. The 'domain_criterion' argument accepts a callback that determines
    if two nodes are in the same domain. That callback accepts a graph and two
    node keys as argument and returns whether or not the nodes are in the same
    domain as a boolean.

    Parameters
    ----------
    molecule: vermouth.molecule.Molecule
        The molecule to which apply the elastic network. The molecule is
        modified in-place.
    selector: collections.abc.Callable
        Selection function.
    lower_bound: float
        The minimum length for a bond to be added, expressed in
        nanometers.
    upper_bound: float
        The maximum length for a bond to be added, expressed in
        nanometers.
    decay_factor: float
        Parameter for the decay function.
    decay_power: float
        Parameter for the decay function.
    base_constant: float
        The base force constant for the bonds in :math:`kJ.mol^{-1}.nm^{-2}`.
        If 'decay_factor' or 'decay_power' is set to 0, then it will be the
        used force constant.
    minimum_force: float
        Minimum force constant in :math:`kJ.mol^{-1}.nm^{-2}` under which bonds
        are not kept.
    bond_type: int
        Gromacs bond function type to apply to the elastic network bonds.
    domain_criterion: collections.abc.Callable
        Function to establish if two atoms are part of the same domain. Elastic
        bonds are only added within a domain. By default, all the atoms in
        the molecule are considered part of the same domain. The function
        expects a graph (e.g. a :class:`~vermouth.molecule.Molecule`) and two atom node keys as
        argument and returns ``True`` if the two atoms are part of the same
        domain; returns ``False`` otherwise.
    res_min_dist: int
        Minimum separation between two atoms for a bond to be kept.
        Bonds are kept is the separation is greater or equal to the value
        given.
    """
    selection = []
    coordinates = []
    missing = []
    node_to_idx = {}
    idx_to_node = {}
    for node_idx, (node_key, attributes) in enumerate(molecule.nodes.items()):
        node_to_idx[node_key] = node_idx
        idx_to_node[node_idx] = node_key
        if selector(attributes):
            selection.append(node_idx)
            coordinates.append(attributes.get('position'))
            if coordinates[-1] is None:
                missing.append(node_key)
        node_idx += 1
    if missing:
        raise ValueError('All atoms from the selection must have coordinates. '
                         'The following atoms do not have some: {}.'
                         .format(' '.join(missing)))

    if not coordinates:
        return

    coordinates = np.stack(coordinates)
    if np.any(np.isnan(coordinates)):
        LOGGER.warning("Found nan coordinates in molecule {}. "
                       "Will not generate an EN for it. ",
                       molecule.moltype,
                       type='unmapped-atom')
        return

    distance_matrix = self_distance_matrix(coordinates)
    constants = compute_force_constants(distance_matrix, lower_bound,
                                        upper_bound, decay_factor, decay_power,
                                        base_constant, minimum_force)

    connected = build_connectivity_matrix(molecule, res_min_dist, node_to_idx,
                                          selected_nodes=selection)

    same_domain = build_pair_matrix(molecule, domain_criterion, idx_to_node,
                                    selected_nodes=selection)

    can_be_linked = (~connected) & same_domain
    # Multiply the force constant by 0 if the nodes cannot be linked.
    constants *= can_be_linked
    distance_matrix = distance_matrix.round(5)  # For compatibility with legacy
    for from_idx, to_idx in zip(*np.triu_indices_from(constants)):
        # note the indices in the matrix are not anymore the idx of
        # the full molecule but the subset of nodes in selection
        from_key = idx_to_node[selection[from_idx]]
        to_key = idx_to_node[selection[to_idx]]
        force_constant = constants[from_idx, to_idx]
        length = distance_matrix[from_idx, to_idx]
        if force_constant > minimum_force:
            molecule.add_interaction(
                type_='bonds',
                atoms=(from_key, to_key),
                parameters=[bond_type, length, force_constant],
                meta={'group': 'Rubber band', 'comment': 'elastic-bond'},
            )


def always_true(*args, **kwargs):  # pylint: disable=unused-argument
    """
    Returns ``True`` whatever the arguments are.
    """
    return True


def same_chain(graph, left, right):
    """
    Returns ``True`` is the nodes are part of the same chain.

    Nodes are considered part of the same chain if they both have the same value
    under the "chain" attribute, or if neither of the 2 nodes have that attribute.

    Parameters
    ----------
    graph: networkx.Graph
        A graph the nodes are part of.
    left:
        A node key in 'graph'.
    right:
        A node key in 'graph'.

    Returns
    -------
    bool
        ``True`` if the nodes are part of the same chain.
    """
    node_left = graph.nodes[left]
    node_right = graph.nodes[right]
    return node_left.get('chain') == node_right.get('chain')

def make_same_region_criterion(regions):
    """
    Returns ``True`` is the nodes are part of the same region.

    Nodes are considered part of the same region if their value
    under the "resid" attribute are within the same residue range.
    By default the resids of the input file are used (i.e. "_old_resid"
    attribute).

    Parameters
    ----------
    graph: networkx.Graph
        A graph the nodes are part of.
    left:
        A node key in 'graph'.
    right:
        A node key in 'graph'.
    regions:
        [(resid_start_1,resid_end_1),(resid_start_2,resid_end_2),...] resid_start and resid_end are included)

    Returns
    -------
    bool
        ``True`` if the nodes are part of the same region.
    """

    regions = copy.deepcopy(regions)

    def same_region(graph, left, right):
        node_left = graph.nodes[left]
        node_right = graph.nodes[right]
        left_resid = node_left.get('_old_resid', node_left['resid'])
        right_resid = node_right.get('_old_resid', node_right['resid'])
        for region in regions:
            lower = min(region)
            upper = max(region)
            if lower <= left_resid <= upper and lower <= right_resid <= upper:
                return True
        return False
    return same_region
    
class ApplyRubberBand(Processor):
    """
    Add an elastic network to a system between particles fulfilling the
    following criteria:

    - They must be close enough together in space
    - They must be separated far enough in graph space
    - They must be either in the same chain/molecule/system
    - They must be selected by :attr:`selector`
    - The resulting elastic bond must be stiff enough

    Attributes
    ----------
    selector: collections.abc.Callable
        Selection function.
    lower_bound: float
        The minimum length for a bond to be added, expressed in
        nanometers.
    upper_bound: float
        The maximum length for a bond to be added, expressed in
        nanometers.
    decay_factor: float
        Parameter for the decay function.
    decay_power: float
        Parameter for the decay function.
    base_constant: float
        The base force constant for the bonds in :math:`kJ.mol^{-1}.nm^{-2}`.
        If 'decay_factor' or 'decay_power' is set to 0, then it will be the
        used force constant.
    minimum_force: float
        Minimum force constant in :math:`kJ.mol^{-1}.nm^{-2}` under which bonds
        are not kept.
    bond_type: int or None
        Gromacs bond function type to apply to the elastic network bonds.
    bond_type_variable: str
        If bond_type is not given, it will be taken from the force field, using
        this variable name.
    domain_criterion: collections.abc.Callable
        Function to establish if two atoms are part of the same domain. Elastic
        bonds are only added within a domain. By default, all the atoms in
        the molecule are considered part of the same domain. The function
        expects a graph (e.g. a :class:`~vermouth.molecule.Molecule`) and two
        atom node keys as argument and returns ``True`` if the two atoms are
        part of the same domain; returns ``False`` otherwise.
    res_min_dist: int or None
        Minimum separation between two atoms for a bond to be kept.
        Bonds are kept is the separation is greater or equal to the value
        given.
    res_min_dist_variable: str
        If res_min_dist is not given it will be taken from the force field using
        this variable name.

    See Also
    --------
    :func:`apply_rubber_band`
    """
    def __init__(self, lower_bound, upper_bound, decay_factor, decay_power,
                 base_constant, minimum_force,
                 res_min_dist=None,
                 bond_type=None,
                 selector=selectors.select_backbone,
                 bond_type_variable='elastic_network_bond_type',
                 res_min_dist_variable='elastic_network_res_min_dist',
                 domain_criterion=always_true):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.decay_factor = decay_factor
        self.decay_power = decay_power
        self.base_constant = base_constant
        self.minimum_force = minimum_force
        self.bond_type = bond_type
        self.selector = selector
        self.bond_type_variable = bond_type_variable
        self.domain_criterion = domain_criterion
        self.res_min_dist = res_min_dist
        self.res_min_dist_variable = res_min_dist_variable

    def run_molecule(self, molecule):
        # Choose the bond type. From high to low, the priority order is:
        # * what is set as an argument to the processor
        # * what is written in the force field variables
        #   under the key `self.bond_type_variable`
        # * the default value set in DEFAULT_BOND_TYPE
        bond_type = self.bond_type
        if self.bond_type is None:
            bond_type = molecule.force_field.variables.get(self.bond_type_variable,
                                                           DEFAULT_BOND_TYPE)

        # Same procedure for res_min_dist the minimum distance between
        # the resids of two beads for them to have a RB
        res_min_dist = self.res_min_dist
        if self.res_min_dist is None:
            res_min_dist = molecule.force_field.variables.get(self.res_min_dist_variable,
                                                              DEFAULT_RMD)

        apply_rubber_band(molecule, self.selector,
                          lower_bound=self.lower_bound,
                          upper_bound=self.upper_bound,
                          decay_factor=self.decay_factor,
                          decay_power=self.decay_power,
                          base_constant=self.base_constant,
                          minimum_force=self.minimum_force,
                          bond_type=bond_type,
                          domain_criterion=self.domain_criterion,
                          res_min_dist=res_min_dist)
        return molecule
