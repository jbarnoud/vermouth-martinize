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
Test the ApplyRubberBand processor and the related functions.
"""
import functools
import pytest
import numpy as np
import networkx as nx
import vermouth
import vermouth.forcefield
from vermouth import selectors
from vermouth.processors import apply_rubber_band
from vermouth.processors.apply_rubber_band import (same_chain,
                                                   make_same_region_criterion,
                                                   are_connected,
                                                   build_connectivity_matrix)
from vermouth.tests.helper_functions import test_molecule

# pylint: disable=redefined-outer-name



@pytest.fixture
def disconnected_graph():
    """
    A graph with two connected components.
    The node keys are integers from 0 to 15 (included). The name of each node
    is its key as a string. The first component is labelled as chain A while
    the second is labelled as chain B.
    0 - 1 - 2 - 3
                |
    4 - 5 - 6 - 7

     8 -  9 - 10 - 11
     |         |    |
    12 - 13 - 14 - 15
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(16))
    graph.add_edges_from([
        # First connected component
        [0, 1], [1, 2], [2, 3], [3, 7], [4, 5], [5, 6], [6, 7],
        # Second connected component
        [8, 9], [8, 12], [8, 9], [9, 10], [10, 14], [10, 11], [11, 15],
        [12, 13], [13, 14], [14, 15],
    ])
    for key, node in graph.nodes.items():
        node['name'] = str(key)
        if key < 8:
            node['chain'] = 'A'
        else:
            node['chain'] = 'B'
    return graph


@pytest.mark.parametrize('selection', (
    list(range(16)),  # Use all the nodes, explicitly
    list(range(8)),  # Only the first component
    list(range(8, 16)),  # Only the second component
    list(range(0, 16, 2)),  # Every other nodes
))
@pytest.mark.parametrize('extra_edges', ([], [(7, 10)]))
def test_build_pair_matrix(disconnected_graph, selection, extra_edges):
    """
    The creation of a pair matrix works as expected.

    Here, the criterion is ``True`` when nodes belong to the same domain.
    The graph is defined as having two domains: one per chain. The extra edges
    allow to make sure the connectivity does not impact the domain detection.
    """
    def have_same_chain(graph, left, right):
        return graph.nodes[left]['chain'] == graph.nodes[right]['chain']

    expected = np.zeros((16, 16), dtype=bool)
    expected[:8, :8] = True
    expected[8:, 8:] = True
    np.fill_diagonal(expected, False)
    expected = expected[:, selection][selection]

    disconnected_graph.add_edges_from(extra_edges)
    idx_to_node = dict(enumerate(disconnected_graph.nodes))
    domains = apply_rubber_band.build_pair_matrix(
        disconnected_graph, have_same_chain, idx_to_node, selection)
    assert np.all(domains == expected)


@pytest.mark.parametrize('separation, outcome', (
    (1, [[0, 2], [2, 6], [4, 6], [8, 12], [8, 10],
         [10, 14], [12, 14]]),
    (2, [[0, 2], [2, 6], [4, 6], [0, 6], [2, 4],
         [8, 12], [8, 10], [10, 14], [12, 14],
         [8, 14], [10, 12]])
))
# this only tests connectivity matrix in terms of connection
# and separation in residue space. selection is tested at the
# end with a more integral test
def test_build_connectivity_matrix(disconnected_graph, separation, outcome):
    idx_to_node = dict(enumerate(disconnected_graph.nodes))
    selection = list(range(0, 16, 2))
    resid_attr = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4,
                  8: 5, 9: 5, 10: 6, 11: 6, 12: 7, 13: 7, 14: 8, 15: 8}
    nx.set_node_attributes(disconnected_graph, resid_attr, 'resid')
    connected = build_connectivity_matrix(disconnected_graph,
                                          separation,
                                          idx_to_node,
                                          selection)
    pairs = []
    for from_idx, to_idx in zip(*np.triu_indices_from(connected)):
        if connected[from_idx, to_idx]:
            idxs = [selection[from_idx], selection[to_idx]]
            idxs.sort()
            pairs.append(idxs)

    for pair in outcome:
        assert pair in pairs

    assert len(pairs) == len(outcome)


@pytest.mark.parametrize('nodes, edges, outcome', (
    ([1, 2, 3],
     [(1, 2), (2, 3)],
     True),
    ([1, 2, 3],
     [(2, 3)],
     False)
))
def test_are_connected(nodes, edges, outcome):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    assert are_connected(graph, 1, 2, 1) == outcome


@pytest.mark.parametrize('nodes, chain, edges, outcome', (
    ([1, 2, 3],
     {1: "A", 2: "A", 3: "C"},
     [(1, 2), (2, 3)],
     True),
    ([1, 2, 3],
     {1: "A", 2: "B", 3: "C"},
     [(1, 2), (2, 3)],
     False)
))
def test_same_chain(nodes, edges, chain, outcome):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    nx.set_node_attributes(graph, chain, "chain")
    print (same_chain(graph, 1, 2))
    assert same_chain(graph, 1, 2) == outcome

@pytest.mark.parametrize('regions, left, right, nodes, chain, resid, edges, outcome', (
    ([(2, 3)],
     2,
     3,
     [1, 2, 3, 4],
     {1: "A", 2: "A", 3: "A", 4: "A"},
     {1: 1, 2: 2, 3: 3, 4: 4},
     [(1, 2), (2, 3), (3, 4)],
     True),
    ([(2, 3)],
     1,
     2,
     [1, 2, 3, 4],
     {1: "A", 2: "A", 3: "A", 4: "A"},
     {1: 1, 2: 2, 3: 3, 4: 4},
     [(1, 2), (2, 3), (3, 4)],
     False),
    ([(3, 2)],
     2,
     3,
     [1, 2, 3, 4],
     {1: "A", 2: "A", 3: "A", 4: "A"},
     {1: 1, 2: 2, 3: 3, 4: 4},
     [(1, 2), (2, 3), (3, 4)],
     True),
    ([(2, 3)],
     2,
     3,
     [1, 2, 3, 4],
     {1: "A", 2: "A", 3: "B", 4: "B"},
     {1: 1, 2: 2, 3: 3, 4: 4},
     [(1, 2), (2, 3), (3, 4)],
     True),
    ([(2, 3)],
     1,
     2,
     [1, 2, 3, 4],
     {1: "A", 2: "A", 3: "B", 4: "B"},
     {1: 1, 2: 2, 3: 3, 4: 4},
     [(1, 2), (2, 3), (3, 4)],
     False),
    ([(2, 3)],
     2,
     5,
     [1, 2, 3, 4, 5, 6],
     {1: "A", 2: "A", 3: "A", 4: "B", 5: "B", 6: "B"},
     {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3},
     [(1, 2), (2, 3), (3, 4)],
     True),
    ([(2, 3)],
     5,
     6,
     [1, 2, 3, 4, 5, 6],
     {1: "A", 2: "A", 3: "A", 4: "B", 5: "B", 6: "B"},
     {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3},
     [(1, 2), (2, 3), (3, 4)],
     True),
    ([(2, 3)],
     2,
     6,
     [1, 2, 3, 4, 5, 6],
     {1: "A", 2: "A", 3: "A", 4: "B", 5: "B", 6: "B"},
     {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3},
     [(1, 2), (2, 3), (3, 4)],
     True),
    ([(2, 3)],
     2,
     4,
     [1, 2, 3, 4, 5, 6],
     {1: "A", 2: "A", 3: "A", 4: "B", 5: "B", 6: "B"},
     {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3},
     [(1, 2), (2, 3), (3, 4)],
     False),
    ([(3, 3)],
     3,
     3,
     [1, 2, 3, 4, 5, 6],
     {1: "A", 2: "A", 3: "A", 4: "B", 5: "B", 6: "B"},
     {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3},
     [(1, 2), (2, 3), (3, 4)],
     True),
    ([(3, 3)],
     3,
     6,
     [1, 2, 3, 4, 5, 6],
     {1: "A", 2: "A", 3: "A", 4: "B", 5: "B", 6: "B"},
     {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3},
     [(1, 2), (2, 3), (3, 4)],
     True),
))
def test_make_same_region_criterion(regions, left, right, nodes, edges, chain, resid, outcome):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    nx.set_node_attributes(graph, chain, "chain")
    nx.set_node_attributes(graph, resid, "resid")
    same_region = make_same_region_criterion(regions)
    assert same_region(graph=graph, left=left, right=right) == outcome


@pytest.mark.parametrize('chain_attribute, atom_names, res_min_dist, outcome',
                         (({0: 'A', 1: 'A', 2: 'A',
                            3: 'A', 4: 'A', 5: 'A',
                            6: 'A', 7: 'A', 8: 'A'},
                           ['BB'],
                           2,
                           [vermouth.molecule.Interaction(
                               atoms=(0, 6),
                               meta={'group': 'Rubber band'},
                               parameters=[6, 1.5, 1000])]
                           ),
                          # different min_res
                          ({0: 'A', 1: 'A', 2: 'A',
                            3: 'A', 4: 'A', 5: 'A',
                            6: 'A', 7: 'A', 8: 'A'},
                           ['BB'],
                           1,
                           [vermouth.molecule.Interaction(
                               atoms=(0, 5),
                               meta={'group': 'Rubber band'},
                               parameters=[6, 1.0, 1000]),
                            vermouth.molecule.Interaction(
                                atoms=(0, 6),
                                meta={'group': 'Rubber band'},
                                parameters=[6, 1.5, 1000]),
                            vermouth.molecule.Interaction(
                                atoms=(3, 6),
                                meta={'group': 'Rubber band'},
                                parameters=[6, 1.0, 1000])]
                           ),
                          # select more than only BB atoms
                          ({0: 'A', 1: 'A', 2: 'A',
                            3: 'A', 4: 'A', 5: 'A',
                            6: 'A', 7: 'A', 8: 'A'},
                           ['BB', 'SC1'],
                           2,
                           [vermouth.molecule.Interaction(
                               atoms=(0, 6),
                               meta={'group': 'Rubber band'},
                               parameters=[6, 1.5, 1000]),
                            vermouth.molecule.Interaction(
                                atoms=(0, 7),
                                meta={'group': 'Rubber band'},
                                parameters=[6, 1.58114, 1000]),
                            vermouth.molecule.Interaction(
                                atoms=(1, 6),
                                meta={'group': 'Rubber band'},
                                parameters=[6, 1.58114, 1000]),
                            vermouth.molecule.Interaction(
                                atoms=(1, 7),
                                meta={'group': 'Rubber band'},
                                parameters=[6, 1.5, 1000])]
                           ),
                          # change chain identifier
                          ({0: 'A', 1: 'A', 2: 'A',
                            3: 'B', 4: 'B', 5: 'B',
                            6: 'B', 7: 'B', 8: 'B'},
                           ['BB'],
                           1,
                           [vermouth.molecule.Interaction(
                               atoms=(3, 6),
                               meta={'group': 'Rubber band'},
                               parameters=[6, 1.0, 1000])]
                           )))
def test_apply_rubber_bands(test_molecule, chain_attribute, atom_names, res_min_dist, outcome):
    """
    Takes molecule and sets the chain attributes. Based on chain, minimum distance
    between residues, and atom names elagible it is tested if rubber bands are applied
    for the correct atoms in molecule.
    """
    selector = functools.partial(
        selectors.proto_select_attribute_in,
        attribute='atomname',
        values=atom_names)

    domain_criterion = vermouth.processors.apply_rubber_band.same_chain
    nx.set_node_attributes(test_molecule, chain_attribute, 'chain')

    process = vermouth.processors.apply_rubber_band.ApplyRubberBand(
        selector=selector,
        lower_bound=0.0,
        upper_bound=10.,
        decay_factor=0,
        decay_power=0.,
        base_constant=1000,
        minimum_force=1,
        bond_type=6,
        domain_criterion=domain_criterion,
        res_min_dist=res_min_dist,
        write_out=False)
    process.run_molecule(test_molecule)
    assert test_molecule.interactions['bonds'] == outcome


@pytest.mark.parametrize('regions, chain_attribute, atom_names, res_min_dist, outcome',
                         (([(1, 4)],
                           {0: 'A', 1: 'A', 2: 'A',
                            3: 'A', 4: 'A', 5: 'A',
                            6: 'A', 7: 'A', 8: 'A'},
                           ['BB'],
                           2,
                           [vermouth.molecule.Interaction(
                               atoms=(0, 6),
                               meta={'group': 'Rubber band'},
                               parameters=[6, 1.5, 1000])]
                           ),
                          # different min_res and different regions
                          ([(1, 3)],
                           {0: 'A', 1: 'A', 2: 'A',
                            3: 'A', 4: 'A', 5: 'A',
                            6: 'A', 7: 'A', 8: 'A'},
                           ['BB'],
                           1,
                           [vermouth.molecule.Interaction(
                               atoms=(0, 5),
                               meta={'group': 'Rubber band'},
                               parameters=[6, 1.0, 1000])]
                           ),
                          # select more than only BB atoms
                          ([(1, 3)],
                           {0: 'A', 1: 'A', 2: 'A',
                            3: 'A', 4: 'A', 5: 'A',
                            6: 'A', 7: 'A', 8: 'A'},
                           ['BB', 'SC1'],
                           2,
                           []
                           ),
                          # change chain identifier
                          ([(1, 3)],
                           {0: 'A', 1: 'A', 2: 'A',
                            3: 'B', 4: 'B', 5: 'B',
                            6: 'B', 7: 'B', 8: 'B'},
                           ['BB'],
                           1,
                           [vermouth.molecule.Interaction(
                               atoms=(0, 5),
                               meta={'group': 'Rubber band'},
                               parameters=[6, 1.0, 1000])]
                           )))
def test_apply_rubber_bands_same_regions(test_molecule, regions, chain_attribute, atom_names, res_min_dist, outcome):
    """
    Takes molecule and sets the chain attributes. Based on chain, minimum distance
    between residues, and atom names elagible it is tested if rubber bands are applied
    for the correct atoms in molecule.
    """
    selector = functools.partial(
        selectors.proto_select_attribute_in,
        attribute='atomname',
        values=atom_names)

    domain_criterion = vermouth.processors.apply_rubber_band.make_same_region_criterion(regions)
    nx.set_node_attributes(test_molecule, chain_attribute, 'chain')

    process = vermouth.processors.apply_rubber_band.ApplyRubberBand(
        selector=selector,
        lower_bound=0.0,
        upper_bound=10.,
        decay_factor=0,
        decay_power=0.,
        base_constant=1000,
        minimum_force=1,
        bond_type=6,
        domain_criterion=domain_criterion,
        res_min_dist=res_min_dist,
        write_out=False)
    process.run_molecule(test_molecule)
    assert test_molecule.interactions['bonds'] == outcome

def test_skip_no_matches(test_molecule):
    """
    Tests that when no node matches the EN selectors, the molecule is simply skipped.
    """
    selector = functools.partial(
        selectors.proto_select_attribute_in,
        attribute='matches_none',
        values=['BB', 'SC1'])

    domain_criterion = vermouth.processors.apply_rubber_band.always_true

    process = vermouth.processors.apply_rubber_band.ApplyRubberBand(
        selector=selector,
        lower_bound=0.0,
        upper_bound=10.,
        decay_factor=0,
        decay_power=0.,
        base_constant=1000,
        minimum_force=1,
        bond_type=6,
        domain_criterion=domain_criterion,
        res_min_dist=3,
        write_out=False)
    process.run_molecule(test_molecule)
    assert test_molecule.interactions['bonds'] == []

def test_bail_out_on_nan(caplog, test_molecule):
    """
    Test the the EN processor bails out when nan coordinate
    is found.
    """
    test_molecule.nodes[0]['position'] = np.array([np.nan, np.nan, np.nan])
    test_molecule.moltype = "testmol"
    selector = functools.partial(
        selectors.proto_select_attribute_in,
        attribute='atomname',
        values=['BB'])

    domain_criterion = vermouth.processors.apply_rubber_band.always_true

    process = vermouth.processors.apply_rubber_band.ApplyRubberBand(
        selector=selector,
        lower_bound=0.0,
        upper_bound=10.,
        decay_factor=0,
        decay_power=0.,
        base_constant=1000,
        minimum_force=1,
        bond_type=6,
        domain_criterion=domain_criterion,
        res_min_dist=3,
        write_out=False)
    process.run_molecule(test_molecule)

    required_warning = ("Found nan coordinates in molecule testmol. "
                        "Will not generate an EN for it. ")
    record = caplog.records[0]
    assert record.getMessage() == required_warning
    assert len(caplog.records) == 1
    assert test_molecule.interactions['bonds'] == []


@pytest.mark.parametrize('lower_bound, upper_bound, decay_factor, decay_power, base_constant, minimum_force, expected_output', 
                          ([1, 2, 0, 0, 500, 400, np.array([[  0,500,500,  0],
                                                            [500,  0,500,500],
                                                            [500,500,  0,500],
                                                            [  0,500,500,  0]])], # no decays, return the base constant within the bounds
                           [1, 2, 0, 0, 500, 600, np.array([[  0,  0,  0,  0],
                                                            [  0,  0,  0,  0],
                                                            [  0,  0,  0,  0],
                                                            [  0,  0,  0,  0]])], # all forces less than the minumum force
                           [1, 0.5, 0, 0, 500, 600, np.array([[  0,  0,  0,  0],
                                                              [  0,  0,  0,  0],
                                                              [  0,  0,  0,  0],
                                                              [  0,  0,  0,  0]])], # all distances larger than the upper bound
                           [4, 2, 0, 0, 500, 600, np.array([[  0,  0,  0,  0],
                                                            [  0,  0,  0,  0],
                                                            [  0,  0,  0,  0],
                                                            [  0,  0,  0,  0]])], # all distances less than the lower bound
                           [1, 3, 0.1, 0, 500, 400, np.array([[  0.        , 452.41870902, 452.41870902, 452.41870902],
                                                              [452.41870902,   0.        , 452.41870902, 452.41870902],
                                                              [452.41870902, 452.41870902,   0.        , 452.41870902],
                                                              [452.41870902, 452.41870902, 452.41870902,   0.        ]])], # with a decay factor
                           [1, 3, 0, 2, 500, 400, np.array([[  0,500,500,500],
                                                            [500,  0,500,500],
                                                            [500,500,  0,500],
                                                            [500,500,500,  0]])], # with a decay factor
                           [1, 3, 0.1, 2, 500, 400, np.array([[  0.        , 500.        , 452.41870902,   0.        ],
                                                              [500.        ,   0.        , 500.        , 452.41870902],
                                                              [452.41870902, 500.        ,   0.        , 500.        ],
                                                              [  0.        , 452.41870902, 500.        ,   0.        ]])], # with a complex decay
                          )
                        )
def test_compute_force_constants(lower_bound, upper_bound, decay_factor, decay_power, base_constant, minimum_force, expected_output):
    # Define the constant distance_matrix for the test cases
    distance_matrix = np.array([[0, 1, 2, 3],
                                [1, 0, 1, 2],
                                [2, 1, 0, 1],
                                [3, 2, 1, 0]])

    # Call the compute_force_constants function with the constant distance_matrix and varied parameters
    result = vermouth.processors.apply_rubber_band.compute_force_constants(distance_matrix, lower_bound, upper_bound, 
                                                                           decay_factor, decay_power, base_constant, 
                                                                           minimum_force)

    # Assert the result against the expected output
    assert result == pytest.approx(expected_output)