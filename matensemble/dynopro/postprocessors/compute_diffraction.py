import pickle
from ovito.io.pymatgen import ovito_to_pymatgen

def get_laue_pattern(data, fname):
        
        struc = ovito_to_pymatgen(data.data)
        
        import pymatgen.analysis.diffraction.tem as tem
        temcalc = tem.TEMCalculator()
        temcalc.get_plot_2d(struc).write_image(f'{fname}.png')
        temcalc.get_pattern(struc).to_csv(f'{fname}_Laue_Pattern.csv', sep=' ')

        return


def get_xrd_pattern(data, fname):
  
        struc = ovito_to_pymatgen(data.data)

        from pymatgen.analysis.diffraction.xrd import XRDCalculator
        xrd_calc = XRDCalculator()
        pattern = xrd_calc.get_pattern(struc)

        pattern_dict= {} #pattern.as_dict() 
        pattern_dict['2_theta'] = pattern.x.tolist()
        pattern_dict['Intensities'] = pattern.y.tolist()
        pattern_dict['hkls'] = pattern.hkls


        with open(f'{fname}_XRD_Pattern.pkl','wb') as file:
                pickle.dump(pattern_dict, file)
        return pattern_dict












