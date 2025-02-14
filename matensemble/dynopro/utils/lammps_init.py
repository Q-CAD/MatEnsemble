        
import lammps
import numpy as np
from pymatgen.io.lammps.data import  LammpsData, LammpsBox
from pymatgen.core import Structure

def initilize_from_MD(lmp_snapshot):
        
        lmp = lammps.lammps()
        lmp_box = lmp_snapshot['box_info'] #lmp.extract_box()
        lmp.command('atom_style atomic')
        lmp.command(f'region box_reg prism {lmp_box[0][0]} {lmp_box[1][0]} \
                {lmp_box[0][1]} {lmp_box[1][1]} {lmp_box[0][2]} {lmp_box[1][2]} \
                {lmp_box[2]} {lmp_box[3]} {lmp_box[4]} \
                units box')
        
        N_TYPES = 1 # len(lmp_struc.structure.types_of_specie)--> TO DO. in case ulti-species treatment is important
        N_ATOMS = len(lmp_snapshot['coords'])
        lmp.command(f'create_box {N_TYPES} box_reg')
        lmp.create_atoms(len(N_ATOMS), list(np.arange(1,N_ATOMS+1)),  list(np.arange(1,N_TYPES+1)), x=lmp_snapshot['coords'])

        return lmp


def initialize_from_poscar(structure):

        try:
                struc = Structure.from_file(structure)
        except:
                struc = structure

        lmp_struc = LammpsData.from_structure(struc)

        lmp = lammps.lammps()

        bounds = lmp_struc.box.bounds
        tilts = lmp_struc.box.tilt
        lmp.command('atom_style charge')
        lmp.command(f'region box_reg prism {bounds[0][0]} {bounds[0][1]} \
                {bounds[1][0]} {bounds[1][1]} {bounds[2][0]} {bounds[2][1]} \
                        {tilts[0]} {tilts[1]} {tilts[2]} \
                                units box')
        
        N_TYPES = len(lmp_struc.structure.types_of_specie)
        N_ATOMS = len(lmp_struc.atoms)
        lmp.command(f'create_box {N_TYPES} box_reg')

        lmp.create_atoms(N_ATOMS, lmp_struc.atoms.index.to_list(), lmp_struc.atoms.loc[:,'type'].to_list(), \
                         x=np.array(lmp_struc.atoms.loc[:,['x','y','z']]).flatten('C').tolist())


        return lmp