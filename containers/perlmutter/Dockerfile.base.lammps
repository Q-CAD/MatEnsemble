FROM docker.io/nersc/base_gpu:26.04
WORKDIR /opt
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install -y libblas-dev liblapack-dev libgsl-dev

RUN python3 -m venv /opt/basic
ENV PATH=$PATH:/opt/basic/bin
RUN ln -sf /opt/basic/bin/pip /usr/local/bin/pip
RUN pip install cuequivariance-ops-torch-cu12 cuequivariance-torch cuequivariance
RUN pip install cupy-cuda12x
RUN pip install cython setuptools pyfftw
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip install mace-torch==0.3.14
RUN pip install mpi4py -i https://pypi.anaconda.org/mpi4py/simple
ENV PATH=$PATH:/opt/basic/bin
ENV PYTHONPATH=$PYTHONPATH:/opt/basic/lib/python3.12/site-packages

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
            -D LAMMPS_SIZES=BIGBIG -D CMAKE_POSITION_INDEPENDENT_CODE=ON -D CMAKE_EXE_FLAGS="-dynamic" \
            -D FFT=FFTW3 -D FFT_PACK=array -D FFT_KOKKOS=CUFFT ../cmake
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
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/basic/lib/python3.12/site-packages/torch/lib
