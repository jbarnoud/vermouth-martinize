"""
I/O of topology parameters that are not molecules.
"""
import itertools
from collections import namedtuple
import textwrap
import vermouth
from vermouth.file_writer import deferred_open
from vermouth.citation_parser import citation_formatter
from ..log_helpers import StyleAdapter, get_logger
from .itp import _interaction_sorting_key

LOGGER = StyleAdapter(get_logger(__name__))

Atomtype = namedtuple('Atomtype', 'molecule node sigma epsilon meta')
NonbondParam = namedtuple('NonbondParam', 'atoms sigma epsilon meta')

def _group_by_conditionals(interactions):
    interactions_group_sorted = sorted(interactions,
        key=_interaction_sorting_key
        )
    interaction_grouped = itertools.groupby(
        interactions_group_sorted,
        key=_interaction_sorting_key
        )
    return interaction_grouped

def sigma_epsilon_to_C6_C12(sigma, epsilon):
    """
    Convert the LJ potential from sigma epsilon
    form to C6 C12 form.
    """
    C6 = 4*sigma*epsilon**6
    C12 = 4*sigma*epsilon**12
    return C6, C12

def write_atomtypes(system, itp_path, C6C12=False):
    """
    Writes the [atomtypes] directive to file.
    All atomtypes are defined in system.gmx_topology_params.
    Masses and further information are taken from the molecule
    directly.
    """
    conditional_keys = {True: '#ifdef', False: '#ifndef'}
    with deferred_open(itp_path, "w") as itp_file:
        itp_file.write("[ atomtypes ]\n")
        grouped_types = _group_by_conditionals(system.gmx_topology_params['atomtypes'])
        for (conditional, group), interactions_in_group in grouped_types:
            # conditionals are things like #ifdef; for more details on how this
            # works see the molecule_itp_writer
            if conditional:
                conditional_key = conditional_keys[conditional[1]]
                itp_file.write('{} {}\n'.format(conditional_key, conditional[0]))
            # groups are collections of interactions that are written bunched
            # together and indicated by a comment line
            if group:
                itp_file.write('; {}\n'.format(group))

            for atomtype in interactions_in_group:
                atype = atomtype.molecule.nodes[atomtype.node]['atype']
                charge = atomtype.molecule.nodes[atomtype.node]['charge']
                mass = atomtype.molecule.nodes[atomtype.node]['mass']

                if C6C12:
                    nb1, nb2 = sigma_epsilon_to_C6_C12(atomtype.sigma, atomtype.epsilon)
                else:
                    nb1, nb2 = atomtype.sigma, atomtype.epsilon

                if 'comment' in atomtype.meta:
                    comments = ";" + " ".join(atomtype.meta['comment'])
                else:
                    comments = ""
                itp_file.write(f"{atype} {mass} {charge} A {nb1:3.8F} {nb2:3.8F} {comments}\n")

def write_nonbond_params(system, itp_path, C6C12=False):
    """
    Writes the [nonbond_params] directive to file.
    All atomtypes are defined in system.gmx_topology_params.
    Masses and further information are taken from the molecule
    directly.
    """
    conditional_keys = {True: '#ifdef', False: '#ifndef'}
    with deferred_open(itp_path, "w") as itp_file:
        itp_file.write("[ nonbond_params ]\n")
        grouped_types = _group_by_conditionals(system.gmx_topology_params['nonbond_params'])
        for (conditional, group), interactions_in_group in grouped_types:
            if conditional:
                conditional_key = conditional_keys[conditional[1]]
                itp_file.write('{} {}\n'.format(conditional_key, conditional[0]))
            if group:
                itp_file.write('; {}\n'.format(group))


            for nb_params in interactions_in_group:
                if len(nb_params.atoms) == 2:
                    a1, a2 = nb_params.atoms
                # self interaction
                else:
                    a1 = nb_params.atoms[0]
                    a2 = nb_params.atoms[0]

                if C6C12:
                    nb1, nb2 = sigma_epsilon_to_C6_C12(nb_params.sigma, nb_params.epsilon)
                else:
                    nb1, nb2 = nb_params.sigma, nb_params.epsilon

                if nb_params.meta.get('comment'):
                    comments = ";" + " ".join(nb_params.meta['comment'])
                else:
                    comments = ""
                itp_file.write(f"{a1} {a2} 1 {nb1:3.8F} {nb2:3.8F} {comments}\n")

def write_gmx_topology(system,
                       top_path,
                       itp_paths={"nonbond_params": "extra_nbparams.itp",
                                  "atomtypes": "extra_atomtypes.itp"},
                       C6C12=False,
                       defines=(),
                       header=()):
    """
    Writes a Gromacs .top file for the specified system. Gromacs topology
    files are defined by directives for example `[ atomtypes ]`. However,
    Gromacs supports writing parts of the topology to so called .itp
    files which can be inculded into a toplevel topology file with the
    extension .top using #include statements. The topology writer will
    generate such a toplevel topology file where the different directives
    are written to seperate .itp files and included into the toplevel
    file.

    Parameters
    ----------
    system: vermouth.system.System
    top_path: pathlib.Path
        path for topology file
    itp_paths: dict[str, pathlib.Path]
        list of paths for writing the topology parameters
        like atomtypes with the key being the name of the
        directive.
    C6C12: bool
        write non-bonded interaction parameters using LJ
        C6C12 form
    defines: tuple(str)
        define statments to include in the topology
    header: tuple(str)
        any comment lines to include at the beginning
    """
    if not system.molecules:
        raise ValueError("No molecule in the system. Nothing to write.")

    include_string = ""
    # First we write the atomtypes directive
    if "atomtypes" in system.gmx_topology_params:
        _path = itp_paths['atomtypes']
        write_atomtypes(system, _path, C6C12)
        include_string += f'\n #include "{_path}"'
    # Next we write the nonbond_params directive
    if "nonbond_params" in system.gmx_topology_params:
        _path = itp_paths['nonbond_params']
        write_nonbond_params(system, _path, C6C12)
        include_string += f'\n #include "{_path}"\n'


    # Write the ITP files for the molecule types, and prepare writing the
    # [ molecules ] section of the top file.
    # * We write one ITP file for each different moltype in the system, the
    #   moltype being defined by the name provided under the "moltype" meta of
    #   the molecules. If more than one molecule share the same moltype, we use
    #   the first one to write the ITP file.
    moltype_written = set()
    # * We keep track of the length of the longer moltype name, to align the
    #   [ molecules ] section.
    max_name_length = 0
    # * We keep track of groups of successive molecules with the same moltypes.
    moltype_count = []  # items will be [moltype, number of molecules]

    # Iterate over groups of successive molecules with the same moltypes. We
    # shall *NOT* sort the molecules before hand, as groups of successive
    # molecules with the same moltype can be interupted by other moltypes, and
    # we want to reflect these interuptions in the [ molecules ] section of the
    # top file.
    molecule_groups = itertools.groupby(
        system.molecules, key=lambda x: x.meta["moltype"]
    )
    for moltype, molecules in molecule_groups:
        molecule = next(molecules)
        if moltype not in moltype_written:
            # A given moltype can appear more than once in the sequence of
            # molecules, without being uninterupted by other moltypes. Even in
            # that case, we want to write the ITP only once.
            with deferred_open("{}.itp".format(moltype), "w") as outfile:
                # here we format and merge all citations
                header[-1] = header[-1] + "\n"
                header.append("Please cite the following papers:")
                for citation in molecule.citations:
                    cite_string = citation_formatter(
                        molecule.force_field.citations[citation]
                    )
                    LOGGER.info("Please cite: " + cite_string)
                    header.append(cite_string)
                vermouth.gmx.itp.write_molecule_itp(molecule, outfile, header=header)
            this_moltype_len = len(molecule.meta["moltype"])
            if this_moltype_len > max_name_length:
                max_name_length = this_moltype_len
            moltype_written.add(moltype)
        # We already removed one element from the "molecules" generator, do not
        # forget to count it in the number of molecules in that group.
        moltype_count.append([moltype, 1 + len(list(molecules))])

    # Write the top file
    template = textwrap.dedent(
        """\
        {defines}
        #include "martini.itp"
        {includes}

        [ system ]
        Title of the system

        [ molecules ]
        {molecules}
    """
    )
    include_string = include_string + "\n".join(
        '#include "{}.itp"'.format(molecule_type) for molecule_type, _ in moltype_count
    )
    molecule_string = "\n".join(
        "{mtype:<{length}}    {num}".format(
            mtype=mtype, num=num, length=max_name_length
        )
        for mtype, num in moltype_count
    )
    define_string = "\n".join("#define {}".format(define) for define in defines)
    with deferred_open(str(top_path), "w") as outfile:
        outfile.write(
            textwrap.dedent(
                template.format(
                    includes=include_string,
                    molecules=molecule_string,
                    defines=define_string,
                )
            )
        )
