units real
dimension 3
boundary p p p
atom_style charge
atom_modify     sort 0 0
neighbor        2.5 bin
neigh_modify    every 1 delay 1 check yes  # Aggressive during melting
processors * * *

read_data       ${structure}

pair_style      reax/c ${control_filename} safezone 1.5 mincap 100
pair_coeff      * * ${ff_filename} Bi Se

fix             1 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c

thermo          100
thermo_style    custom step temp pe epair etotal press pxx pyy pzz lx ly lz vol density

dump            positions all atom 200 filename.lammpstrj
dump            d1 all custom 1000 dump_*.dump id type q x y z
dump_modify     d1 element Bi Se

timestep        1.0
restart         100000 restart.*.res

#---------- ENERGY_MINIMIZATION ---------------
minimize        1.0e-4 1.0e-6 100 1000
write_data      Min.data

#---------- NPT_RELAXATION ---------------------
fix             ensemble all npt temp 300.0 300.0 100.0 iso 0.0 0.0 5000.0
run             25000  
unfix           ensemble
write_restart   npt_relax.res
write_data      data.npt_relax
