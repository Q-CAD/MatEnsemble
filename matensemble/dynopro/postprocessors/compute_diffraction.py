from pymatgen.io.ase import AseAtomsAdaptor
from ovito.io.ase import ovito_to_ase
import pickle

def get_laue_pattern(data, fname):
        
        ase_data = ovito_to_ase(data.data)
        struc = AseAtomsAdaptor.get_structure(ase_data)
        
        import pymatgen.analysis.diffraction.tem as tem
        temcalc = tem.TEMCalculator()
        temcalc.get_plot_2d(struc).write_image(f'{fname}.png')
        temcalc.get_pattern.to_csv(f'{fname}_Laue_Pattern.csv', sep=' ')

        return


def get_xrd_pattern(data, fname):

        ase_data = ovito_to_ase(data.data)
        struc = AseAtomsAdaptor.get_structure(ase_data)

        from pymatgen.analysis.diffraction.xrd import XRDCalculator
        xrd_calc = XRDCalculator()
        pattern = xrd_calc.get_pattern(struc)

        pattern_dict= pattern.as_dict() 
        pattern_dict['2_theta'] = pattern['x']
        pattern_dict['Intensities'] = pattern['y']


        with open(f'{fname}_XRD_Pattern.pkl','wb') as file:
                pickle.dump(pattern_dict, file)

        return












