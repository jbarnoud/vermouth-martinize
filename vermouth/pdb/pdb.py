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
Provides functions for reading and writing PDB files.
"""

import numpy as np
import networkx as nx

from ..file_writer import deferred_open
from ..molecule import Molecule
from ..utils import first_alpha, distance, format_atom_string
from ..parser_utils import LineParser
from ..truncating_formatter import TruncFormatter
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))


class PDBParser(LineParser):
    """
    Parser for PDB files

    Attributes
    ----------
    active_molecule: vermouth.molecule.Molecule
        The molecule/model currently being read.
    molecules: list[vermouth.molecule.Molecule]
        All complete molecules read so far.
    modelidx: int
        Which model to take.

    Parameters
    ----------
    exclude: collections.abc.Container[str]
        Container of residue names. Any atom that has a residue name that is in
        `exclude` will be skipped.
    ignh: bool
        Whether all hydrogen atoms should be skipped
    modelidx: int
        Which model to take.
    """

    def __init__(self, exclude=('SOL',), ignh=False, modelidx=1):
        self.active_molecule = Molecule()
        self.molecules = []
        self._conects = []
        self.exclude = exclude
        self.ignh = ignh
        self.modelidx = modelidx
        self._skipahead = False
        self.cryst = {}
        self._seqres = []

    def dispatch(self, line):
        """
        Returns the appropriate method for parsing `line`. This is determined
        based on the first 6 characters of `line`.

        Parameters
        ----------
        line: str

        Returns
        -------
        collections.abc.Callable[str, int]
            The method to call with the line, and the line number.
        """
        record = line[:6].strip().lower()
        return getattr(self, record, self._unknown_line)

    def parse(self, file_handle):
        # Only PDBParser.finalize should produce a result, namely a list of
        # molecules. This means that mols is a list containing a single list of
        # molecules, which is a little silly.
        outcome = list(super().parse(file_handle))
        assert len(outcome) == 1
        yield from outcome[0]

    @staticmethod
    def _unknown_line(line, lineno):
        """
        Called when a line is unknown. Raises a KeyError with a helpful message.
        """
        raise KeyError("Line {} can not be parsed, since we don't recognize the"
                       " first 6 characters. The line is: {}"
                       "".format(lineno, line))

    @staticmethod
    def _skip(line, lineno=0):
        """
        Does nothing.
        """


    # TODO: Parse some of these, and either do something useful with it, or
    #       propagate it to e.g. the ITP
    # pylint: disable=bad-whitespace
    # TITLE SECTION
    header = _skip
    obslte = _skip
    title  = _skip
    splt   = _skip
    caveat = _skip
    compnd = _skip
    source = _skip
    keywds = _skip
    expdta = _skip
    nummdl = _skip
    mdltyp = _skip
    author = _skip
    revdat = _skip
    sprsde = _skip
    jrnl   = _skip
    remark = _skip

    # PRIMARY STRUCTURE SECTION
    dbref  = _skip
    dbref1 = _skip
    dbref2 = _skip
    seqadv = _skip

    def seqres(self, line, lineno=0):
        """
        Parse a SEQRES record. Store for processing later.

        Parameters
        ----------
        line: str
            The line to parse.

        """
        self._seqres.append(line)

    # seqres = _seqres
    modres = _skip

    # HETEROGEN SECTION
    het    = _skip
    formul = _skip
    hetnam = _skip
    hetsyn = _skip

    # SECONDARY STRUCTURE SECTION
    helix  = _skip
    sheet  = _skip

    # CONNECTIVITY ANNOTATION SECTION
    ssbond = _skip
    link   = _skip
    cispep = _skip

    # MISCELLANEOUS FEATURES SECTION
    site   = _skip

    # CRYSTALLOGRAPHIC AND COORDINATE TRANSFORMATION SECTION
    origx1 = _skip
    origx2 = _skip
    origx3 = _skip
    scale1 = _skip
    scale2 = _skip
    scale3 = _skip
    mtrix1 = _skip
    mtrix2 = _skip
    mtrix3 = _skip

    # COORDINATE SECTION
    # model  = _skip  # Used
    # atom   = _skip  # Used
    anisou = _skip
    # ter    = _skip  # Used
    # hetatm = _skip  # Used
    # endmdl = _skip  # Used

    # CONNECTIVITY SECTION
    # conect = _skip  # Used

    # BOOKKEEPING SECTION
    master = _skip
    # end   = _skip  # Used
    # pylint: enable=bad-whitespace

    def _atom(self, line, lineno=0):
        """
        Parse an ATOM or HETATM record.

        Parameters
        ----------
        line: str
            The line to parse. We do not check whether it starts with either
            "ATOM  " or "HETATM".
        lineno: int
            The line number (not used).
        """
        if self._skipahead:
            return

        fields = [
            ('', str, 6),
            ('atomid', int, 5),
            ('', str, 1),
            ('atomname', str, 4),
            ('altloc', str, 1),
            ('resname', str, 4),
            ('chain', str, 1),
            ('resid', int, 4),
            ('insertion_code', str, 1),
            ('', str, 3),
            ('x', float, 8),
            ('y', float, 8),
            ('z', float, 8),
            ('occupancy', float, 6),
            ('temp_factor', float, 6),
            ('', str, 10),
            ('element', str, 2),
            ('charge', str, 2),
        ]

        start = 0
        field_slices = []
        for name, type_, width in fields:
            if name:
                field_slices.append((name, type_, slice(start, start + width)))
            start += width

        properties = {}
        for name, type_, slice_ in field_slices:
            value = line[slice_].strip()
            if value:
                properties[name] = type_(value)
            else:
                properties[name] = type_()

        # Charge is special, since it's "2-" or "1+", rather than -2 etc. And
        # let's turn it into a number
        charge = properties['charge']
        if charge:
            try:
                charge = float(charge)
            except ValueError:
                charge = float(charge[::-1])
        else:
            charge = 0
        properties['charge'] = charge

        pos = (properties.pop('x'), properties.pop('y'), properties.pop('z'))
        # Coordinates are read in Angstrom, but we want them in nm
        properties['position'] = np.array(pos, dtype=float) / 10

        if not properties['element']:
            atomname = properties['atomname']
            properties['element'] = first_alpha(atomname)
        if properties['altloc'] not in ['', 'A']:
            # TODO: allow selecting alternative conformation for specific
            #       residues.
            LOGGER.warning("There is an alternative conformation for atom {}. "
                           "We use conformation A exclusively",
                           format_atom_string(properties),
                           type='pdb-alternate')
            return
        if (properties['resname'] in self.exclude or
                (self.ignh and properties['element'] == 'H')):
            return
        idx = max(self.active_molecule) + 1 if self.active_molecule else 0
        self.active_molecule.add_node(idx, **properties)

    atom = _atom
    hetatm = _atom

    def cryst1(self, line, lineno=0):
        """
        Parse the CRYST1 record. Crystal structure information are stored with
        the parser object and may be extracted later.
        """
        fields = [
            ('', str, 6),
            ('a', float, 9),
            ('b', float, 9),
            ('c', float, 9),
            ('alpha', float, 7),
            ('beta', float, 7),
            ('gamma', float, 7),
            ('space_group', str, 11),
            ('', str, 1),
            ('z_value', int, 4),
            ]
        start = 0
        field_slices = []
        for name, type_, width in fields:
            if name:
                field_slices.append((name, type_, slice(start, start + width)))
            start += width

        for name, type_, slice_ in field_slices:
            value = line[slice_].strip()
            if value:
                self.cryst[name] = type_(value)
            else:
                LOGGER.warning(f"CRYST1 directive incomplete. Missing entry for {name}.")

    def model(self, line, lineno=0):
        """
        Parse a MODEL record. If the model is not the same as :attr:`modelidx`,
        this model will not be parsed.

        Parameters
        ----------
        line: str
            The line to parse. Should start with "MODEL ", but this is not
            checked.
        lineno: int
            The line number (not used).
        """
        try:
            modelnr = int(line[10:14])
        except ValueError:
            return
        else:
            self._skipahead = modelnr != self.modelidx

    def conect(self, line, lineno=0):
        """
        Parse a CONECT record. The line is stored for later processing.

        Parameters
        ----------
        line: str
            The line to parse. Should start with CONECT, but this is not checked
        lineno: int
            The line number (not used).
        """
        # We can't add edges immediately, since the molecule might not be parsed
        # yet (does the PDB file format mandate anything on the order of
        # records?). Instead, just store the lines for later use.
        self._conects.append(line)

    def _finish_molecule(self, line="", lineno=0):
        """
        Finish parsing the molecule. :attr:`active_molecule` will be appended to
        :attr:`molecules`, and a new :attr:`active_molecule` will be made.
        """
        # We kind of *want* to yield self.active_molecule here, but we can't
        # since there's a very good chance it's CONECT records have not been
        # parsed yet, and the molecule won't have any edges.
        if self.active_molecule:
            if {"a", "b", "c"}.issubset(set(self.cryst.keys())):
                self.active_molecule.box = np.array([self.cryst['a']/10.,
                                                     self.cryst['b']/10.,
                                                     self.cryst['c']/10.])
            self.molecules.append(self.active_molecule)
        self.active_molecule = Molecule()

    endmdl = _finish_molecule
    ter = _finish_molecule
    end = _finish_molecule

    def finalize(self, lineno=0):
        """
        Finish parsing the file. Process all CONECT records found, and returns
        a list of molecules.

        Parameters
        ----------
        lineno: int
            The line number (not used).

        Returns
        -------
        list[vermouth.molecule.Molecule]
            All molecules parsed from this file.
        """
        # TODO: cross reference number of molecules with CMPND records
        self._finish_molecule()
        self.do_conect()
        if len(self._seqres) > 0:
            self.do_seqres()
        return self.molecules

    def do_conect(self):
        """
        Apply connections to molecule based on CONECT records read from PDB file
        """
        id2idxs = [{mol.nodes[idx]['atomid']: idx for idx in mol}
                   for mol in self.molecules]
        for line in self._conects:
            start = 6
            width = 5
            atids = []
            for num in range(start, len(line.rstrip()), width):
                atom = int(line[num:num + width])
                atids.append(atom)
            self._do_single_conect(atids, id2idxs)

    def _do_single_conect(self, conect_record, id2idxs):
        """
        Process a single CONECT line. Adds edges to the molecules in
        :attr:`molecules`.

        Parameters
        ----------
        conect_record: list[int]
        id2idxs: list[dict[int, int]]
            A list of dicts mapping atomids to node keys for every molecule in
            :attr:`molecules`
        """
        atomid0 = conect_record[0]
        # Find the appropriate molecule:
        mol = None
        for mol, id2idx in zip(self.molecules, id2idxs):
            if atomid0 in id2idx:
                atomidx0 = id2idx[atomid0]
                break
        else:  # no break
            # No molecule found with an atom containing atomid atid0.
            # This could be a hydrogen if ignh, or an excluded residue.
            return
        for atomid in conect_record[1:]:
            # Find the second molecule...
            mol2 = None
            for mol2, id2idx in zip(self.molecules, id2idxs):
                if atomid in id2idx:
                    atomidx = id2idx[atomid]
                    break
            else:
                # Two options: a skipped atom
                continue
            if mol is not mol2:
                assert mol is not None and mol2 is not None
                LOGGER.info('Merging two molecules/chains because there is a '
                            'CONECT record between atoms {} and {}',
                            format_atom_string(mol.nodes[atomidx0]),
                            format_atom_string(mol2.nodes[atomidx]))
                # Conect record between two molecules! It's probably a *good*
                # idea to cross reference this with e.g. SSBOND and LINK records
                molidx = self.molecules.index(mol)
                molidx2 = self.molecules.index(mol2)
                del id2idxs[max(molidx, molidx2)]
                del id2idxs[min(molidx, molidx2)]
                self.molecules.remove(mol)
                self.molecules.remove(mol2)
                mol = nx.disjoint_union(mol, mol2)
                mol2 = mol
                self.molecules.append(mol)
                id2idxs.append({mol.nodes[idx]['atomid']: idx for idx in mol})

            dist = distance(mol.nodes[atomidx0]['position'],
                            mol2.nodes[atomidx]['position'])
            mol.add_edge(atomidx0, atomidx, distance=dist)

    def do_seqres(self):
        """
        Get the AA sequences of the chains in the molecules
        """
        chains = list(list(np.unique(list(mol.nodes[idx]['chain'] for idx in mol)))
                      for mol in self.molecules)

        properties = {a: [] for a in [x for xs in chains for x in xs]}

        for line in self._seqres:
            chain = line[11]
            resnames = line[19:].split()
            #some varient of this can check we have protein seqres and not
            #seqres for nucleotides
            #not much use atm but could be helpful for when we have nucleotides here
            # try:
            #     assert list(set(len(i) for i in resnames)) == 3
            # except AssertionError:
            #     #we don't really need to worry about this
            #     #we just need to not use it any further
            #     pass
            properties[chain].append(resnames)

        # this to make a flat list for each chain in the file
        for chain in properties.keys():
            properties[chain] = [x for xs in properties[chain] for x in xs]
        self._check_seqres(properties)


    def _check_seqres(self, properties):
        '''
        In future this could just be done by parsing REMARK 465.
        '''
        LOGGER.info("Checking pdb SEQRES entry for missing residues", type="step")
        for mol in self.molecules:
            resids = np.array([mol.nodes[idx]['resid'] for idx in mol],dtype = int)
            chain = list(set([mol.nodes[idx]['chain'] for idx in mol]))[0]

            missing_res_nos = np.setdiff1d(np.arange(len(properties[chain]))+1,
                                       np.unique(resids)[np.unique(resids) < len(properties[chain])+1])

            #find consecutive sequences of missing residues
            nums = sorted(set(missing_res_nos))
            gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
            edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
            series = list(zip(edges, edges))

            for ser in series:
                LOGGER.warning("SEQRES data suggests residues {}:{} in chain {} are missing",
                                   ser[0], ser[1], chain,
                                   type="pdb-alternate")
                #TODO: something useful with the residues found missing.

def read_pdb(file_name, exclude=('SOL',), ignh=False, modelidx=1):
    """
    Parse a PDB file to create a molecule.

    Parameters
    ----------
    filename: str
        The file to read.
    exclude: collections.abc.Container[str]
        Atoms that have one of these residue names will not be included.
    ignh: bool
        Whether hydrogen atoms should be ignored.
    model: int
        If the PDB file contains multiple models, which one to select.

    Returns
    -------
    list[vermouth.molecule.Molecule]
        The parsed molecules. Will only contain edges if the PDB file has
        CONECT records. Either way, the molecules might be disconnected. Entries
        separated by TER, ENDMDL, and END records will result in separate
        molecules.
    """
    parser = PDBParser(exclude, ignh, modelidx)
    with open(str(file_name)) as file_handle:
        mols = list(parser.parse(file_handle))
    LOGGER.info('Read {} molecules from PDB file {}', len(mols), file_name)
    return mols


def get_not_none(node, attr, default):
    """
    Returns ``node[attr]``. If it doesn't exists or is ``None``, return
    `default`.

    Parameters
    ----------
    node: collections.abc.Mapping
    attr: collections.abc.Hashable
    default
        The value to return if ``node[attr]`` is either ``None``, or does not
        exist.

    Returns
    -------
    object
        The value of ``node[attr]`` if it exists and is not ``None``, else
        `default`.
    """
    value = node.get(attr)
    if value is None:
        value = default
    return value


def write_pdb_string(system, conect=True, omit_charges=True, nan_missing_pos=False):
    """
    Describes `system` as a PDB formatted string. Will create CONECT records
    from the edges in the molecules in `system` iff `conect` is True.

    Parameters
    ----------
    system: vermouth.system.System
        The system to write.
    conect: bool
        Whether to write CONECT records for the edges.
    omit_charges: bool
        Whether charges should be omitted. This is usually a good idea since
        the PDB format can only deal with integer charges.
    nan_missing_pos: bool
        Wether the writing should fail if an atom does not have a position.
        When set to `True`, atoms without coordinates will be written
        with 'nan' as coordinates; this will cause the output file to be
        *invalid* for most uses.

    Returns
    -------
    str
        The system as PDB formatted string.
    """
    out = []

    formatter = TruncFormatter()
#    format_string = 'ATOM  {: >5.5d} {:4.4s}{:1.1s}{:3.3s} {:1.1s}{:4.4d}{:1.1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:2.2s}{:2.2s}'
    format_string = 'ATOM  {: >5dt} {:4st}{:1st}{:3st} {:1st}{:>4dt}{:1st}   {:8.3ft}{:8.3ft}{:8.3ft}{:6.2ft}{:6.2ft}          {:2st}{:2st}'

    nodeidx2atomid = {}
    atomid = 1
    for mol_idx, molecule in enumerate(system.molecules):
        for node_idx in molecule.sorted_nodes:
            # Node indices do not have to be unique across molecules. So store
            # them as (mol_idx, node_idx)
            nodeidx2atomid[(mol_idx, node_idx)] = atomid
            node = molecule.nodes[node_idx]
            atomname = get_not_none(node, 'atomname', '')
            altloc = get_not_none(node, 'altloc', '')
            resname = get_not_none(node, 'resname', '')
            chain = get_not_none(node, 'chain', '')
            resid = get_not_none(node, 'resid', 1)
            insertion_code = get_not_none(node, 'insertion_code', '')
            try:
                # converting from nm to A
                x, y, z = node['position'] * 10  # pylint: disable=invalid-name
            except KeyError:
                if nan_missing_pos:
                    x = y = z = float('nan')  # pylint: disable=invalid-name
                else:
                    raise
            occupancy = get_not_none(node, 'occupancy', 1)
            temp_factor = get_not_none(node, 'temp_factor', 0)
            element = get_not_none(node, 'element', '')
            charge = get_not_none(node, 'charge', 0)
            if charge and not omit_charges:
                charge = '{:+2d}'.format(int(charge))[::-1]
            else:
                charge = ''
            line = formatter.format(format_string, atomid, atomname, altloc,
                                    resname, chain, resid, insertion_code, x,
                                    y, z, occupancy, temp_factor, element,
                                    charge)
            atomid += 1
            out.append(line)
        terline = formatter.format('TER   {: >5dt}      {:3st} {:1st}{: >4dt}{:1st}',
                                   atomid, resname, chain, resid, insertion_code)
        atomid += 1
        out.append(terline)
    if conect:
        number_fmt = '{:>4dt}'
        format_string = 'CONECT '
        for mol_idx, molecule in enumerate(system.molecules):
            for node_idx in molecule:
                todo = sorted(nodeidx2atomid[(mol_idx, n_idx)]
                              for n_idx in molecule[node_idx] if n_idx > node_idx)
                while todo:
                    current, todo = todo[:4], todo[4:]
                    fmt = ['CONECT'] + [number_fmt]*(len(current) + 1)
                    fmt = ' '.join(fmt)
                    line = formatter.format(fmt, nodeidx2atomid[(mol_idx, node_idx)], *current)
                    out.append(line)
    out.append('END   ')
    return '\n'.join(out)


def write_pdb(system, path, conect=True, omit_charges=True, nan_missing_pos=False, defer_writing=True):
    """
    Writes `system` to `path` as a PDB formatted string.

    Parameters
    ----------
    system: vermouth.system.System
        The system to write.
    path: str
        The file to write to.
    conect: bool
        Whether to write CONECT records for the edges.
    omit_charges: bool
        Whether charges should be omitted. This is usually a good idea since
        the PDB format can only deal with integer charges.
    nan_missing_pos: bool
        Whether the writing should fail if an atom does not have a position.
        When set to `True`, atoms without coordinates will be written
        with 'nan' as coordinates; this will cause the output file to be
        *invalid* for most uses.
        for most use.
    defer_writing: bool
        Whether to use :class:`~vermouth.file_writer.DeferredFileWriter` for writing data

    See Also
    --------
    :func:write_pdb_string
    """
    if defer_writing:
        open = deferred_open
    else:
        # This is needed since the variable assignment above declares `open` as
        # a local variable, which means it won't be looked up from the global
        # namespace any more.
        from builtins import open
    with open(path, 'w') as out:
        out.write(write_pdb_string(system, conect, omit_charges, nan_missing_pos))
