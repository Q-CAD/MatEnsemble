FROM docker.io/nersc/lammps:26.05
WORKDIR /opt
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install -y unzip

RUN pip install tblite
RUN pip install --prefer-binary pyscf
RUN pip install gpu4pyscf-cuda12x
RUN pip install tensorflow[and-cuda]
RUN pip install papermill
RUN pip install tqdm
RUN pip install py3DMol
RUN pip install pandas
RUN pip install b2[full]
RUN pip install ply
RUN pip install pyyaml
RUN pip install Cython


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
    ./configure --with-cuda=/usr/local/cuda                                 && \
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
RUN git clone -b v6.6.1 https://github.com/grimme-lab/xtb.git
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


# Install MPI ping-pong from OLCF
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
