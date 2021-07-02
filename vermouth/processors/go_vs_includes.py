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
Add the include statements and the virtual sites for Virtual Site Go model.

The VirtualGoSite model allows to stabilize the ternary structure of Martini
proteins by applying Go potentials maintaining the contacts within the
backbone. The Go potentials are not applied on the backbone beads directly,
instead, they are applied on virtual sites overlapping with the backbone.

The processor defined in this module does not generate the Go potentials.
Instead, they the potential is generated by a third party program. The third
party program generate the interaction matrix for the Go potentials, and the
exclusions as ITP files to be included in the right place in the protein ITP
file. The processor adds an include statement at the end of the `[ exclusions
]` section. Would the third party program need the addition of other include
statements, they can be added by adjusting the `sections` argument of the
processor. To incorporate the include statements, the processor adds the
required lines in the "post_section_lines" meta attribute of the molecules.
This meta attribute is read by :func:`vermouth.gmx.itp.write_molecule_itp`. The
include files are called "<moltype>_<section>_VirtGoSite.itp".

In addition of writing the include statements, the processor adds virtual sites
on top of the backbone beads. The virtual sites are added at the end of the
molecule, they share the residue name, residue id, chain, and position of the
underlying backbone bead. They are also added in the `[ virtual_sitesn ]`
section.
"""

import networkx as nx
from ..molecule import Interaction
from .processor import Processor


class GoVirtIncludes(Processor):
    """
    Add the include statements and the virtual sites for Virtual Site Go model.

    See :mod:`vermouth.processors.go_vs_includes` for more details.

    Every molecule must have a moltype name under the "moltype" key of the
    molecule meta.

    Parameters
    ----------
    sections: collections.abc.Iterable[str], optional
        The sections to which to add an include statement.

    See Also
    --------
    :class:`~vermouth.processors.name_moltype.NameMolType`
        Assign molecule type names to the molecules in a system.
    :func:`add_virtual_sites`
    """
    def __init__(self, sections=('exclusions', )):
        self.sections = sections

    def run_molecule(self, molecule):
        moltype = molecule.meta.get('moltype')
        if not moltype:
            raise ValueError('The molecule does not have a moltype name.')

        add_virtual_sites(molecule, prefix=moltype)

        includes = molecule.meta.get('post_section_lines', {})
        for section in self.sections:
            section_includes = includes.get(section, [])
            section_includes.append('#include "{moltype}_{section}_VirtGoSites.itp"'
                                    .format(moltype=moltype, section=section))
            includes[section] = section_includes
        molecule.meta['post_section_lines'] = includes
        return molecule


def add_virtual_sites(molecule, prefix, backbone='BB', atomname='CA', charge=0):
    """
    Add the virtual sites for GoMartini in the molecule.

    One virtual site is added per backbone bead of the the Martini protein.
    Each virtual site copies the resid, resname, and chain of the backbone
    bead. It also copies the *reference* to the position array, so the virtual
    site position follows if the backbone bead is translated. The virtual sites
    are added *after* all the other atoms of the molecule, each in its own
    charge group, with "CA" as atomname, and a charge of 0. The atomname and
    charge can be set with the `atomname` and `charge` argument, respectively.

    The bead type of the virtual sites is names "<prefix>_<resid>". Where
    `prefix` is provided as an argument of the function, and is expected to be
    the molecule type name.

    Parameters
    ----------
    molecule: vermouth.molecule.Molecule
        The molecule to augment with virtual sites.
    prefix: str
        The prefix to use for bead type names. Usually the molecule type name.
    backbone: str
        The atomname of the backbone beads.
    atomname: str
        The atomname of the virtual sites.
    charge: float or int
        The charge of the virtual sites.
    """
    # If there are no atoms, then there is nothing to do. We can exit early,
    # avoiding to deal with empty iterators.
    if not molecule.nodes:
        return
    virtual_site_nodes = []
    virtual_sites = []
    new_node_id = max(molecule.nodes)
    charge_groups = nx.get_node_attributes(molecule, 'charge_group').values()
    new_charge_group = max(charge_groups) if charge_groups else 0
    for node_id, atom in molecule.nodes(data=True):
        if atom.get('atomname') == backbone:
            new_node_id += 1
            new_charge_group += 1
            virtual_site_nodes.append((new_node_id, {
                'resid': atom['resid'],
                'resname': atom['resname'],
                'atype': '{}_{}'.format(prefix, atom['resid']),
                'charge_group': new_charge_group,
                'chain': atom['chain'],
                'position': atom['position'],
                'atomname': atomname,
                'charge': charge,
            }))
            virtual_sites.append(Interaction(
                atoms=[new_node_id, node_id],
                parameters=['1'],
                meta={'go_vs': True, 'group': 'Virtual go site'},
            ))
    molecule.add_nodes_from(virtual_site_nodes)
    if 'virtual_sitesn' not in molecule.interactions:
        molecule.interactions['virtual_sitesn'] = []
    molecule.interactions['virtual_sitesn'] += virtual_sites
