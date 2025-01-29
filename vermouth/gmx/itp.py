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
Handle the ITP file format from Gromacs.
"""

import copy
import itertools
from collections import defaultdict
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))

__all__ = ['write_molecule_itp', ]


def _attr_has_not_none_attr(obj, attr):
    """
    Raise a Value error is 'obj' does not have an attribute 'attr' or if its
    value is ``None``.
    """
    try:
        value = getattr(obj, attr)
    except AttributeError:
        value = None
    if value is None:
        raise ValueError('{} has no attribute "{}".'.format(obj, attr))


def _interaction_sorting_key(interaction):
    ifdef = interaction.meta.get('ifdef')
    ifndef = interaction.meta.get('ifndef')
    if ifdef is not None and ifndef is not None:
        raise ValueError('An interaction cannot have both an "ifdef" '
                         'and an "ifndef" meta attribute.')
    if ifdef is not None:
        conditional = (ifdef, True)
    elif ifndef is not None:
        conditional = (ifndef, False)
    else:
        conditional = ()

    group = interaction.meta.get('group')
    if group is None:
        group = ''

    return (conditional, group)


def _sort_atoms(atoms, name):
    if name[:4] in ('bond', 'pair'):
        return sorted(atoms)
    elif name.startswith('angle'):
        return atoms if atoms[0] < atoms[-1] else list(reversed(atoms))
    elif name.startswith('dihedral'):
        return atoms if atoms[1] < atoms[2] else list(reversed(atoms))
    return atoms

def _sort_interaction(atoms, name):
    if name.startswith('angle'):
        return (atoms[1], tuple(atoms))
    else: return (min(atoms), tuple(atoms))


def write_molecule_itp(molecule, outfile, header=(), moltype=None,
                       post_section_lines=None, pre_section_lines=None):
    """
    Write a molecule in ITP format.

    The molecule must have a `nrexcl` attribute. Each atom in the molecule must
    have at least the following keys: `atype`, `resid`, `resname`, `atomname`,
    and `charge_group`. Atoms can also have a `charge` and a `mass` key.

    If the `moltype` argument is not provided, then the molecule must have a
    "moltype" meta attribute.

    Parameters
    ----------
    molecule: Molecule
        The molecule to write. See above for the minimal information the
        molecule must contain.
    outfile: io.TextIOBase
        The file in which to write.
    header: collections.abc.Iterable[str]
        List of lines to write as comment at the beginning of the file. The
        comment character and the new line should not be included as they will
        be added in the function.
    moltype: str, optional
        The molecule type. If set to `None` (default), the molecule type is
        read from the "moltype" key of `molecule.meta`.
    post_section_lines: dict[str, collections.abc.Iterable[str]], optional
        List of lines to write at the end of some sections of the file. The
        argument is passed as a dict with the keys being the name of the
        sections, and the values being the lists of lines. If the argument is
        set to `None`, the lines will be read from the "post_section_lines" key
        of `molecule.meta`.
    pre_section_lines: dict[str, collections.abc.Iterable[str]], optional
        List of lines to write at the beginning of some sections, just after
        the section header. The argument is formatted in the same way as
        `post_section_lines`. If the argument is set to `None`, the lines will
        be read from the "post_section_lines" key of `molecule.meta`.

    Raises
    ------
    ValueError
        The molecule is missing required information.
    """
    # Make sure the molecule contains the information required to write the
    # header.
    if moltype is None:
        moltype = molecule.meta.get('moltype')
        if  moltype is None:
            raise ValueError('A molecule must have a moltype to write an '
                             'ITP, provide it with the moltype argument, or '
                             'with the "moltype" meta attribute of the molecule.')
    _attr_has_not_none_attr(molecule, 'nrexcl')

    # Make sure the molecule contains the information required to write the
    # [atoms] section. The charge and mass can be ommited, if so gromacs take
    # them from the [atomtypes] section of the ITP file.
    atoms_with_issues = defaultdict(list)
    for attribute in ('atype', 'resid', 'resname', 'atomname',
                      'charge_group'):
        for _, atom in molecule.atoms:
            if attribute not in atom:
                atoms_with_issues[attribute].append(atom)
    if atoms_with_issues:
        for attr, atoms in atoms_with_issues.items():
            LOGGER.error('The following atoms do not have a {}: {}', attr, atoms)
        raise ValueError("Some atoms are missing required attributes.")

    # Get the maximum length of each atom field so we can align the fields.
    # Atom indexes are written as a consecutive series starting from 1.
    # The maximum index of a 0-based series is `len(x) - 1`; because the
    # series starts at 1, the maximum value is `len(x).
    max_length = {'idx': len(str(len(molecule)))}
    for attribute in ('atype', 'resid', 'resname', 'atomname',
                      'charge_group', 'charge', 'mass'):
        max_length[attribute] = max(len(str(atom.get(attribute, '')))
                                    for _, atom in molecule.atoms)

    # Write the header.
    # We want to follow the header with an empty line, only if there is a
    # header. The `has_header` variable is needed in case `header` is a
    # generator, in which case we cannot know before hand if it contains lines.
    has_header = False
    for line in header:
        outfile.write('; {}\n'.format(line))
        has_header = True
    if has_header:
        outfile.write('\n')

    outfile.write('[ moleculetype ]\n')
    outfile.write('{} {}\n\n'.format(moltype, molecule.nrexcl))

    # Get the post- and pre- section lines. These lines are will be written at
    # the end or at the beginning of the relevant sections.
    if post_section_lines is None:
        post_section_lines = molecule.meta.get('post_section_lines', {})
    if pre_section_lines is None:
        pre_section_lines = molecule.meta.get('pre_section_lines', {})
    seen_sections = set()

    # The atoms in the [atoms] section must be consecutively numbered, yet
    # there is no guarantee that the molecule fulfill that constrain.
    # Therefore we renumber the atoms. The `correspondence` dict allows to
    # keep track of the correspondence between the original and the new
    # numbering so we can apply the renumbering to the interactions.
    # The resid and charge_group should also be consecutive, though this is
    # left as the user responsibility. Make sure residues and charge groups are
    # correctly numbered.
    correspondence = {}
    outfile.write('[ atoms ]\n')
    seen_sections.add('atoms')
    for line in pre_section_lines.get('atoms', []):
        outfile.write(line + '\n')
    # for idx, (original_idx, atom) in enumerate(molecule.atoms, start=1):
    for idx, original_idx in enumerate(molecule.sorted_nodes, start=1):
        atom = molecule.nodes[original_idx]
        correspondence[original_idx] = idx
        new_atom = copy.copy(atom)
        # The charge and the mass can be blank and read from the [atomtypes]
        # section of the ITP file.
        new_atom['charge'] = new_atom.get('charge', '')
        new_atom['mass'] = new_atom.get('mass', '')

        outfile.write('{idx:>{max_length[idx]}} '
                      '{atype:<{max_length[atype]}} '
                      '{resid:>{max_length[resid]}} '
                      '{resname:<{max_length[resname]}} '
                      '{atomname:<{max_length[atomname]}} '
                      '{charge_group:>{max_length[charge_group]}} '
                      '{charge:>{max_length[charge]}} '
                      '{mass:>{max_length[mass]}}\n'
                      .format(idx=idx, max_length=max_length, **new_atom))
    for line in post_section_lines.get('atoms', []):
        outfile.write(line + '\n')
    outfile.write('\n')

    # Write the interactions
    conditional_keys = {True: '#ifdef', False: '#ifndef'}
    for name in molecule.sort_interactions(molecule.interactions):
        interactions = molecule.interactions[name]

        # Improper dihedral angles have their own section in the Molecule
        # object to distinguish them from the proper dihedrals. Yet, they
        # should be written under the [ dihedrals ] section of the ITP file.
        if name == 'impropers':
            section_name = 'dihedrals'
        else:
            section_name = name
        outfile.write('[ {} ]\n'.format(section_name))
        seen_sections.add(name)
        for line in pre_section_lines.get(section_name, []):
            outfile.write(line + '\n')
        interactions_group_sorted = sorted(
            interactions,
            key=_interaction_sorting_key
        )
        interaction_grouped = itertools.groupby(
            interactions_group_sorted,
            key=_interaction_sorting_key
        )
        for (conditional, group), interactions_in_group in interaction_grouped:
            if conditional:
                conditional_key = conditional_keys[conditional[1]]
                outfile.write('{} {}\n'.format(conditional_key, conditional[0]))
            if group:
                outfile.write('; {}\n'.format(group))
            interactions_in_group = sorted(interactions_in_group, key=lambda i: _sort_interaction(_sort_atoms(list(map(correspondence.get, i.atoms)), name), name))
            for interaction in interactions_in_group:
                atoms = ['{atom_idx:>{max_length[idx]}}'
                         .format(atom_idx=x,
                                 max_length=max_length)
                         for x in _sort_atoms(list(map(correspondence.get, interaction.atoms)), name)]
                parameters = ' '.join(str(x) for x in interaction.parameters)
                comment = ''
                if 'comment' in interaction.meta:
                    comment = ' ; ' + interaction.meta['comment']
                if name == 'virtual_sitesn':
                    to_join = [atoms[0], parameters] + atoms[1:]
                else:
                    to_join = atoms + [parameters]
                outfile.write(' '.join(to_join) + comment + '\n')
            if conditional:
                outfile.write('#endif\n')
            for line in post_section_lines.get(name, []):
                outfile.write(line + '\n')
            outfile.write('\n')

    # Some sections may have pre or post lines, but no other content. I that
    # case, we need to write the sections separately.
    remaining_sections = set(pre_section_lines) | set(post_section_lines)
    remaining_sections -= seen_sections
    for name in remaining_sections:
        outfile.write('[ {} ]\n'.format(name))
        for line in pre_section_lines.get(name, []):
            outfile.write(line + '\n')
        for line in post_section_lines.get(name, []):
            outfile.write(line + '\n')
        outfile.write('\n')
