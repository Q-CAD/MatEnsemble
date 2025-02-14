from utils.lammps_init import initilize_from_MD, initialize_from_poscar
import numpy as np
from lammps import LMP_STYLE_ATOM, LMP_TYPE_ARRAY
import sys
import

def SNAP(cutfac, rfrac0, twojmax, R, w, structure=None,lmp_snapshot=None):

        if structure != None:
                lmp = initialize_from_poscar(structure)
        elif lmp_snapshot != None:
                lmp = initilize_from_MD(lmp_snapshot)
        else:
                raise ValueError("Have to specify input structure either")
        
        lmp.command(f'compute BiSpec all sna/atom {cutfac} {rfrac0} {twojmax} {R} {w}')
    
        try:
                lmp.command('run 0')
        except Exception as e:
                raise RuntimeError(f'LAMMPS: {e}\n')

        BiSpec = lmp.numpy.extract_compute('BiSpec', LMP_STYLE_ATOM, LMP_TYPE_ARRAY)

        return BiSpec


if __name__ == '__main__':

        cutfac = sys.argv[1]
        rfrac0 = sys.argv[2]
        twojmax = sys.argv[3]
        R = sys.argv[4]
        w = sys.argv[5]
        structure = sys.argv[6]

        BiSpec = SNAP(cutfac, rfrac0, twojmax, R, w, structure)
        wd = os.path.dirname(structure)
        np.save('BiSpec.npy', BiSpec)




