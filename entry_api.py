# -*- coding: utf-8 -*-
"""
High level API for Martinize2
"""

import argparse
from pathlib import Path
import os
import numpy as np
import martinize2 as m2
from martinize2.forcefield import find_force_fields, FORCE_FIELDS
from martinize2 import DATA_PATH
import itertools
import operator
import textwrap


def read_mapping(path):
    """
    Partial reader for Backward mapping files.

    ..warning::

        This parser is a limited proof of concept. It must be replaced! See
        [issue #5](https://github.com/jbarnoud/martinize2/issues/5).

    Read mapping from a Backward mapping file. Not all fields are supported,
    only the "molecule" and the "atoms" fields are read. The origin force field
    is assumed to be "universal", and the destination force field is assumed to
    be "martini22".

    There are no weight computed in case of shared atoms.

    The reader assumes only one molecule per file.

    Parameters
    ----------
    path: str or Path
        Path to the mapping file to read.

    Returns
    -------
    name: str
        The name of the fragment as read in the "molecule" field.
    from_ff: list of str
        A list of force field origins. Each force field is referred by name.
    to_ff: list of str
        A list of force field destinations. Each force field is referred by name.
    mapping: dict
        The mapping. The keys of the dictionary are the atom names in
        the origin force field; the values are lists of atom names in the
        destination force field, the origin atom is assigned to.
    """
    from_ff = ['universal', ]
    to_ff = ['martini22', ]
    mapping = {}
    
    with open(str(path)) as infile:
        for line_number, line in enumerate(infile, start=1):
            cleaned = line.split(';', 1)[0].strip()
            if not cleaned:
                continue
            elif cleaned[0] == '[':
                if cleaned[-1] != ']':
                    raise IOError('Format error at line {}.'.format(line_number))
                context = cleaned[1:-1].strip()
            elif context == 'molecule':
                name = cleaned
            elif context == 'atoms':
                _, from_atom, *to_atoms = line.split()
                mapping[(0, from_atom)] = [(0, to_atom) for to_atom in to_atoms]

    return name, from_ff, to_ff, mapping


def read_mapping_directory(directory):
    """
    Read all the mapping files in a directory.

    The resulting mapping collection is a 3-level dict where the keys are:
    * the name of the origin force field
    * the name of the destination force field
    * the name of the residue

    The values after these 3 levels is a mapping dict where the keys are the
    atom names in the origin force field and the values are lists of names in
    the destination force field.

    Parameters
    ----------
    directory: str or Path
        The path to the directory to search. Files with a '.map' extension will
        be read. There is no recursive search.

    Returns
    -------
    dict
        A collection of mappings.
    """
    directory = Path(directory)
    mappings = {}
    for path in directory.glob('**/*.map'):
        name, all_from_ff, all_to_ff, mapping = read_mapping(path)
        for from_ff in all_from_ff:
            mappings[from_ff] = mappings.get(from_ff, {})
            for to_ff in all_to_ff:
                mappings[from_ff][to_ff] = mappings[from_ff].get(to_ff, {})
                mappings[from_ff][to_ff][name] = mapping
    return mappings


def read_system(path):
    """
    Read a system from a PDB or GRO file.

    This function guesses the file type based on the file extension.

    The resulting system does not have a force field and may not have edges.
    """
    system = m2.System()
    file_extension = path.suffix.upper()[1:]  # We do not keep the dot
    if file_extension in ['PDB', 'ENT']:
        m2.PDBInput().run_system(system, str(path))
    elif file_extension in ['GRO']:
        m2.GROInput().run_system(system, str(path))
    else:
        raise ValueError('Unknown file extension "{}".'.format(file_extension))
    return system


def select_all(node):
    return True


def select_backbone(node):
    return node.get('atomname') == 'BB'


def pdb_to_universal(system, delete_unknown=False):
    """
    Convert a system read from the PDB to a clean canonical atomistic system.
    """
    canonicalized = system.copy()
    canonicalized.force_field = FORCE_FIELDS['universal']
    m2.MakeBonds().run_system(canonicalized)
    m2.RepairGraph(delete_unknown=delete_unknown).run_system(canonicalized)
    m2.CanonizePTMs().run_system(canonicalized)
    return canonicalized


def martinize(system, mappings, to_ff, delete_unknown=False):
    """
    Convert a system from one force field to an other at lower resolution.
    """
    m2.DoMapping(mappings=mappings, to_ff=to_ff, delete_unknown=delete_unknown).run_system(system)
    m2.DoAverageBead().run_system(system)
    m2.ApplyBlocks().run_system(system)
    m2.DoLinks().run_system(system)
    return system


def write_gmx_topology(system, top_path):
    if not system.molecules:
        raise ValueError('No molecule in the system. Nothing to write.')
    # Deduplicate the moleculetypes in order to write each molecule ITP only
    # once.
    molecule_types = [[system.molecules[0], [system.molecules[0]]], ]
    for molecule in system.molecules[1:]:
        for molecule_type, share_moltype in molecule_types:
            if molecule.share_moltype_with(molecule_type):
                share_moltype.append(molecule)
                break
        else:  # no break
            molecule_types.append([molecule, [molecule, ]])
    # Write the ITP files for the moleculetypes.
    for molidx, (molecule_type, _) in enumerate(molecule_types):
        molecule_type.moltype = 'molecule_{}'.format(molidx)
        with open('molecule_{}.itp'.format(molidx), 'w') as outfile:
            m2.gmx.itp.write_molecule_itp(molecule_type, outfile)
    # Reorganize the molecule type assignment to write the top file.
    # The top file "molecules" section lists the molecules in the same order
    # as in the structure and group them. To do the grouping, we associate each
    # molecule to the molecule type (its name actually) instead of associating
    # the molecule types with the molecules as we did above.
    molecule_to_type = {}
    for molecule_type, share_moltype in molecule_types:
        for molecule in share_moltype:
            molecule_to_type[molecule] = molecule_type.moltype
    # Write the top file
    max_name_length = max(len(molecule_type.moltype)
                          for molecule_type, _ in molecule_types)
    template = textwrap.dedent("""\
        #include "martini.itp"
        {includes}

        [ system ]
        Title of the system

        [ molecules ]
        {molecules}
    """)
    include_string = '\n'.join(
        '#include "{}.itp"'.format(molecule_type.moltype)
        for molecule_type, _ in molecule_types
    )
    molecule_groups = itertools.groupby(system.molecules,
                                        key=lambda x: molecule_to_type[x])
    molecule_string = '\n'.join(
        '{mtype:<{length}}    {num}'
        .format(mtype=mtype, num=len(list(group)), length=max_name_length)
        for mtype, group in molecule_groups
    )
    with open(top_path, 'w') as outfile:
        outfile.write(
            textwrap.dedent(
                template.format(
                    includes=include_string,
                    molecules=molecule_string
                )
            )
        )


def entry():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='inpath', required=True, type=Path)
    parser.add_argument('-x', dest='outpath', required=True, type=Path)
    parser.add_argument('-p', dest='posres',
                        choices=('None', 'All', 'Backbone'), default='None')
    parser.add_argument('-pf', dest='posres_fc', type=float, default=500)
    parser.add_argument('-ff', dest='to_ff', default='martini22')
    args = parser.parse_args()

    known_force_fields = m2.forcefield.find_force_fields(Path(DATA_PATH) / 'force_fields')
    known_mappings = read_mapping_directory(Path(DATA_PATH) / 'mappings')

    from_ff = 'universal'
    if args.to_ff not in known_force_fields:
        raise ValueError('Unknown force field "{}".'.format(args.to_ff))
    if from_ff not in known_mappings or args.to_ff not in known_mappings[from_ff]:
        raise ValueError('No mapping known to go from "{}" to "{}".'
                         .format(from_ff, args.to_ff))

    # Reading the input structure.
    # So far, we assume we only go from atomistic to martini. We want the
    # input structure to be a clean universal system.
    # For now at least, we silently delete molecules with unknown blocks.
    system = read_system(args.inpath)
    system = pdb_to_universal(system, delete_unknown=True)

    # Run martinize on the system.
    system = martinize(
        system,
        mappings=known_mappings,
        to_ff=known_force_fields[args.to_ff],
        delete_unknown=True,
    )

    # Apply position restraints if required.
    if args.posres != 'None':
        selectors = {'All': select_all, 'Backbone': select_backbone}
        selector = selectors[args.posres]
        m2.ApplyPosres(selector, args.posres_fc).run_system(system)

    # Write a PDB file.
    m2.pdb.write_pdb(system, str(args.outpath))

    write_gmx_topology(system, Path('topol.top'))


if __name__ == '__main__':
    entry()
