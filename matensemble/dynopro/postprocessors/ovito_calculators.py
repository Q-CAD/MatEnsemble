
from ovito.data import DataCollection
from ovito.modifiers import  DislocationAnalysisModifier
from ovito.io import export_file
import numpy as np

class OvitoCalculators():

        def __init__(self, lmp_snapshot):
                self.timestep = lmp_snapshot['timestep']
                self.data = DataCollection()
                particles = self.data.create_particles(count=len(lmp_snapshot['coords']))
                particles.create_property('Position', data=lmp_snapshot['coords'])
                particles.create_property('Particle Type', data=lmp_snapshot['types'])
                particles.create_property('Particle Identifier', data=np.arange(1, 1+len(lmp_snapshot['coords'])))
                self.data.particles['Particle Type'][self.data.particles['Particle Type']==1].name = 'Mo'
                self.data.particles['Particle Type'][self.data.particles['Particle Type']==2].name = 'S'

                # Extract LAMMPS simulation cell geometry and boundary conditions.

                lmp_box = lmp_snapshot['box_info'] #lmp.extract_box()

                cell_matrix = np.empty((3,4))

                cell_matrix[:,0] = (lmp_box[1][0] - lmp_box[0][0],0,0)

                cell_matrix[:,1] = (lmp_box[2], lmp_box[1][1] - lmp_box[0][1], 0)
                cell_matrix[:,2] = (lmp_box[4], lmp_box[3], lmp_box[1][2] - lmp_box[0][2]) #lammps_extract_box --> lo, hi, xy, yz, xz
                cell_matrix[:,3] = (lmp_box[0][0], lmp_box[0][1], lmp_box[0][2])
                pbc_flags = bool(lmp_box[5][0]), bool(lmp_box[5][1]), bool(lmp_box[5][2])
                dimension = lmp_snapshot['dim'] 
                # Create the OVITO simulation cell object.
                cell = self.data.create_cell(cell_matrix, pbc=pbc_flags)
                cell.is2D = (dimension != 3)



        def calculate_dxa(self, export_formats=['ca', 'vtk'], dxa_line_sep=15, lattice='BCC'):
 
                modifier = DislocationAnalysisModifier()
                modifier.input_crystal_structure = DislocationAnalysisModifier.Lattice.BCC
                if lattice == "FCC":
                        modifier.input_crystal_structure = DislocationAnalysisModifier.Lattice.FCC
                elif lattice == "HCP":
                        modifier.input_crystal_structure = DislocationAnalysisModifier.Lattice.HCP

                modifier.line_point_separation = dxa_line_sep
                self.data.apply(modifier)
                self.total_line_length = self.data.attributes['DislocationAnalysis.total_line_length']

                for format in export_formats:
                        if format=='ca':
                                export_file(self.data, f'dxa.{self.timestep}.{dxa_line_sep}.ca', format='ca')
                        elif format=='vtk':
                                export_file(self.data, f'dxa.{self.timestep}.{dxa_line_sep}.vtk', format='vtk/disloc')
                        else:
                                print (f"currently exporting in format: {format} is not yet implmeneted")
                return
             

        
        
        def dump_trajectories(self, export_formats=['lammps/dump'],atom_style='atomic'):
                for format in export_formats:
                        if format=='lammps/dump':
                                export_file(self.data, f'Ovito_dump.{self.timestep}.lmp', format=format, columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z"])
                        elif format=='lammps/data':
                                export_file(self.data, f'Ovito_struc.{self.timestep}.lmp', format=format, atom_style=atom_style)
                        else:
                                print (f"currently exporting in format: {format} is not yet implmeneted")
                return
