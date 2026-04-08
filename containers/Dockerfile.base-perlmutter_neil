FROM docker.io/nersc/base_gpu:26.03
WORKDIR /opt
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install -y libblas-dev liblapack-dev libgsl-dev unzip

RUN python3 -m venv /opt/basic
ENV PATH=$PATH:/opt/basic/bin
RUN ln -sf /opt/basic/bin/pip /usr/local/bin/pip
RUN pip install cuequivariance-ops-torch-cu13 cuequivariance-torch cuequivariance
RUN pip install cupy-cuda13x
RUN pip install cython setuptools pyfftw
RUN pip install mpi4py -i https://pypi.anaconda.org/mpi4py/simple
RUN pip install tblite
RUN pip install --prefer-binary pyscf
RUN pip install gpu4pyscf-cuda11x
RUN pip install tensorflow[and-cuda]
RUN pip install papermill
RUN pip install tqdm
RUN pip install py3DMol
RUN pip install pandas
RUN pip install b2[full]
RUN pip install ply
RUN pip install pyyaml
RUN pip install Cython
ENV PATH=$PATH:/opt/basic/bin
ENV PYTHONPATH=$PYTHONPATH:/opt/basic/lib/python3.12/site-packages



# Install Boost
WORKDIR /opt
RUN wget 'https://sourceforge.net/projects/boost/files/boost/1.87.0/boost_1_87_0.zip'
RUN unzip boost_1_87_0.zip
RUN mv boost_1_87_0 boost
RUN cd /opt/boost                                                           && \
    ./bootstrap.sh                                                          && \
    ./b2 install


#Install HWLOC
WORKDIR /opt
RUN git clone -b v2.11 https://github.com/open-mpi/hwloc.git hwloc          && \
    cd hwloc                                                                && \
    ./autogen.sh                                                            && \
    ./configure                                                             && \
    make -j 16                                                              && \
    make install 


# Install dependencies required by flux-core
RUN apt-get update && apt-get install -y \
    lua5.3 liblua5.3-dev \
    libzmq3-dev \
    libjansson-dev \
    libczmq-dev \
    libsqlite3-dev \
    libarchive-dev \
    libsodium-dev \
    uuid-dev \
    libcurl4-openssl-dev \
    libncurses-dev \
    pkg-config \
    libyaml-cpp-dev \
    libedit-dev

# Install flux-core
WORKDIR /opt
RUN git clone -b v0.69.0 https://github.com/flux-framework/flux-core.git    && \
    cd flux-core                                                            && \
    ./autogen.sh                                                            && \
    ./configure                                                             && \
    make -j 1                                                              && \ 
    make install
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
ENV LD_RUN_PATH=$LD_RUN_PATH:/usr/local/lib


# Install flux-sched
WORKDIR /opt
RUN git clone -b v0.36.0 https://github.com/flux-framework/flux-sched.git   && \
    cd flux-sched                                                           && \
    cmake -B build                                                          && \
    cmake --build build                                                     && \
    cmake --build build -t install                                          
RUN pip install flux-python

# Install Open Babel
WORKDIR /opt
RUN git clone -b openbabel-3-1-1 https://github.com/openbabel/openbabel.git            
RUN git clone https://github.com/conda-forge/openbabel-feedstock.git obf
RUN cd /opt/openbabel                                                       && \
    mkdir build                                                             && \
    cd build                                                                && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/opt/openbabel/install -DBUILD_GUI=ON      \
             -DCMAKE_BUILD_TYPE=Release -DWITH_INCHI=ON                        \
             -DPYTHON_BINDINGS=ON -DRUN_SWIG=ON                                \
             -DCMAKE_CXX_FLAGS="-include ctime"                             && \
    make -j 2                                                               && \
    make install 
ENV PYTHONPATH=$PYTHONPATH:/opt/openbabel
ENV PYTHONPATH=$PYTHONPATH:/opt
ENV PATH=$PATH:/opt/openbabel/install/bin
ENV PATH=$PATH:/opt/openbabel/install/include
ENV PATH=$PATH:/opt/openbabel/install/share
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/openbabel/install/lib
ENV PATH=$PATH:/opt/openbabel/install/include/openbabel3
RUN ln -s /opt/openbabel/install/include/openbabel3 /usr/local/include/openbabel3
RUN ln -s /opt/openbabel/install/lib/libopenbabel.so /usr/local/lib/libopenbabel.so
RUN OB_PY=$(find /opt/openbabel/install -type d -name "openbabel" | head -1) && \
    if [ -n "$OB_PY" ]; then cp -r "$OB_PY" /usr/local/lib/python3.12/dist-packages/; fi


# Install XTB from Grimme-lab
WORKDIR /opt
RUN git clone https://github.com/grimme-lab/xtb.git                         
RUN cd /opt/xtb                                                             && \
    cmake -B build -DCMAKE_C_COMPILER=gcc -DCMAKE_FORTRAN_COMPILER=gfortran    \
    -DCMAKE_INSTALL_PREFIX=/opt/xtb/install -DCMAKE_BUILD_TYPE=Release      && \
    make -C build                                                           && \
    make -C build install
ENV PATH=$PATH:/opt/xtb/install/bin
ENV PATH=$PATH:/opt/xtb/install/include
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/xtb/install/lib


# Install Architector from LANL
WORKDIR /opt
RUN git clone -b Secondary_Solvation_Shell https://github.com/lanl/Architector.git 
RUN cd /opt/Architector                                                     && \
    pip install .                                                         
RUN pip install executorlib


# Install MPI-ping-pong from OLCF
WORKDIR /opt
ENV MPI_HOME "/opt/mpich/install"
RUN git clone https://github.com/olcf-tutorials/MPI_ping_pong.git test_code 
RUN cd /opt/test_code/cpu                                                   && \
    make                                                                    
RUN cd /opt/test_code/cuda_staged                                           && \
    sed -i 's:-arch=sm_70:-arch=sm_80:g' Makefile                           && \
    sed -i 's:OMPI_DIR:MPI_HOME:g' Makefile                                 && \
    sed -i 's:-lmpi_ibm:-lmpi:g' Makefile                                   && \
    make                                                                   
RUN cd /opt/test_code/cuda_aware                                            && \
    sed -i 's:-arch=sm_70:-arch=sm_80:g' Makefile                           && \
    sed -i 's:OMPI_DIR:MPI_HOME:g' Makefile                                 && \
    sed -i 's:-lmpi_ibm:-lmpi:g' Makefile                                   && \
    make
ENV PATH=$PATH:/opt/test_code/cpu
ENV PATH=$PATH:/opt/test_code/cuda_staged
ENV PATH=$PATH:/opt/test_code/cuda_aware

ENV PATH=$PATH:/opt/lammps/build/plumed_build-prefix/bin
ENV PATH=$PATH:/opt/lammps/build/plumed_build-prefix/include
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/lammps/build/plumed_build-prefix/lib
ENV PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/opt/lammps/build/plumed_build-prefix/lib/pkgconfig
ENV PLUMED_KERNEL=/opt/lammps/build/plumed_build-prefix/lib/libplumedKernel.so
ARG BLAS_LIBRARIES=/usr/local/blas/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/fftw/install/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/blas
ENV PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/opt/fftw

#Installing lammps
WORKDIR /opt
RUN git clone https://github.com/brucefan1983/NEP_CPU.git nep_cpu
RUN cp -r /opt/nep_cpu/src/* /opt/nep_cpu/interface/lammps/USER-NEP
RUN git clone -b stable_22Jul2025_update3 --recursive https://github.com/lammps/lammps.git lammps
RUN cp -r /opt/nep_cpu/interface/lammps/USER-NEP /opt/lammps/src
# Fix QUIP build: ML-QUIP.cmake generates arch/Makefile.lammps with F95FLAGS += but
# QUIP's Makefile.rules uses F90FLAGS for .F90 files, so -ffree-line-length-none is never
# applied, causing gfortran to error on long lines in System.F90.
RUN sed -i 's/F95FLAGS += /F90FLAGS += /' /opt/lammps/cmake/Modules/Packages/ML-QUIP.cmake
WORKDIR /opt/lammps
RUN mkdir build
WORKDIR /opt/lammps/build
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/compat
RUN sed -i '/tag;/s/t_int_1d_randomread/t_tagint_1d_randomread/' /opt/lammps/src/KOKKOS/fix_electron_stopping_kokkos.h
RUN cmake -DCMAKE_INSTALL_PREFIX=$PWD/../install -D CMAKE_BUILD_TYPE=Release \
            -D CMAKE_CXX_COMPILER=/opt/lammps/lib/kokkos/bin/nvcc_wrapper -D LAMMPS_EXCEPTIONS=ON \
	    -D PKG_KOKKOS=yes -D Kokkos_ARCH_ZEN3=yes  -D Kokkos_ARCH_AMPERE80=ON -D Kokkos_ENABLE_CUDA=yes \
            -D BUILD_MPI=ON -D BUILD_SHARED_LIBS=on -D CMAKE_CXX_STANDARD=17 -D Kokkos_ENABLE_OPENMP=yes \
            -D BUILD_SHARED_LIBS=ON -D PKG_MANYBODY=ON -D PKG_MOLECULE=ON -D PKG_KSPACE=ON -D PKG_REPLICA=yes \
            -D PKG_GPU=yes -D GPU_API=cuda -D GPU_ARCH=sm_80 -D LAMMPS_SIZES=bigbig \
            -D PKG_RIGID=ON -D PKG_MPIIO=ON -D PKG_ML-SNAP=ON -D PKG_ASPHERE=yes -D PKG_BODY=yes -D PKG_CLASS2=yes \
            -D PKG_COLLOID=yes -D PKG_COMPRESS=yes -D PKG_CORESHELL=yes -D PKG_DIPOLE=yes \
            -D PKG_AMOEBA=yes -D PKG_BPM=yes -D PKG_BROWNIAN=yes -D PKG_CG-DNA=yes -D PKG_CG-SPICA=yes \
            -D PKG_DIELECTRIC=yes -D PKG_DPD-BASIC=yes -D PKG_DPD-MESO=yes -D PKG_DPD-REACT=yes -D PKG_DPD-SMOOTH=yes \
            -D PKG_ELECTRODE=yes -D PKG_EXTRA-COMPUTE=yes -D PKG_EXTRA-MOLECULE=yes -D PKG_INTERLAYER=yes -D PKG_LEPTON=yes \
            -D PKG_MESONT=yes -D PKG_ML-IAP=yes -D MLIAP_ENABLE_PYTHON=yes -D PKG_ML-POD=yes -D PKG_PYTHON=yes \
            -D PKG_ORIENT=yes -D PKG_PLUGIN=yes -D PKG_REACTION=yes \
            -D PKG_AMBER=yes -D PKG_CHARMM=yes -D PKG_COMPASS=yes -D PKG_DREIDING=yes -D PKG_CLASS2=yes \
            -D PKG_GRANULAR=yes -D PKG_EXTRA-DUMP=yes -D PKG_MANYBODY=yes -D PKG_MC=yes -D PKG_ML-PACE=yes \
            -D PKG_MISC=yes -D PKG_MOLECULE=yes -D PKG_MPIIO=yes -D PKG_OPT=yes -D PKG_PERI=yes -D PKG_POEMS=yes \
            -D PKG_QEQ=yes -D PKG_REPLICA=yes -D PKG_RIGID=yes -D PKG_SHOCK=yes -D PKG_SPIN=yes \
            -D PKG_SRD=yes -D PKG_OPENMP=yes -D PKG_H5MD=yes -D PKG_AWPMD=yes -D PKG_BOCS=yes \
            -D PKG_DIFFRACTION=yes -D PKG_DRUDE=yes -D PKG_EFF=yes -D PKG_FEP=yes -D PKG_INTEL=yes \
            -D PKG_MANIFOLD=yes -D PKG_MGPT=yes -D PKG_MISC=yes -D PKG_MOFFF=yes -D PKG_MOLFILE=yes \
            -D PKG_PHONON=yes -D PKG_PTM=yes -D PKG_QMMM=yes -D PKG_QTB=yes -D PKG_SMTBQ=yes -D PKG_SPH=yes \
            -D PKG_TALLY=yes -D PKG_UEF=yes -D PKG_YAFF=yes -D PKG_EXTRA-FIX=yes -D PKG_EXTRA-PAIR=yes \
            -D PKG_REAXFF=yes -D PKG_PLUMED=yes -D DOWNLOAD_PLUMED=yes -D PLUMED_MODE=static -D PKG_UEF=yes \
            -D PKG_MSCG=ON -D DOWNLOAD_MSCG=yes -D PKG_H5MD=yes -D PKG_MEAM=yes \
            -D PKG_SCFACOS=ON -D DOWNLOAD_SCAFACOS=yes \
            -D PKG_ML-QUIP=ON -D DOWNLOAD_QUIP=yes -D PKG_KIM=ON -D DOWNLOAD_KIM=yes -WITH_FFMPEG=yes \
            -D PKG_MACHDYN=ON -D DOWNLOAD_EIGEN3=yes -D PKG_LATTE=ON -D DOWNLOAD_LATTE=yes -D PKG_OPENMP=yes \
            -D LAMMPS_SIZES=BIGBIG -D CMAKE_POSITION_INDEPENDENT_CODE=ON -D CMAKE_EXE_FLAGS="-dynamic" ../cmake
RUN make -j 4
RUN make install
WORKDIR /opt/lammps/python
RUN pip install --no-cache-dir -U build wheel setuptools
RUN pip wheel .
RUN pip install --no-cache-dir --force-reinstall lammps-*.whl
WORKDIR /opt

# Setup entry point for production.
ENV PATH="$PATH:/opt/lammps/install/bin"
ENV PATH=/opt/lammps/install/bin:$PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/lammps/install/lib
ENV PATH=/opt/lammps/build/kim_build-prefix/bin:$PATH
ENV PATH=/opt/lammps/build/kim_build-prefix/include:$PATH
ENV LD_LIBRARY_PATH=/opt/lammps/build/kim_build-prefix/lib:$LD_LIBRARY_PATH
