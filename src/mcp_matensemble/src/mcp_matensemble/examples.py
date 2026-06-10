from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

PORTABLE_SYSTEM_GUIDANCE = (
    "These workflows are site-independent MatEnsemble examples. They show the "
    "Python workflow shape; use system-specific examples for launch scripts, "
    "containers, scheduler flags, and science dependency setup."
)

GENERIC_FLUX_COMPATIBLE_SYSTEMS = ("frontier", "perlmutter", "pathfinder", "linux")


@dataclass(frozen=True)
class ExampleSummary:
    name: str
    path: str
    demonstrates: str
    id: str | None = None
    title: str | None = None
    system: str | None = None
    system_title: str | None = None
    compatible_systems: tuple[str, ...] = ()
    agent_guidance: str | None = None


@dataclass(frozen=True)
class ExampleManifestEntry:
    id: str
    name: str
    title: str
    system: str
    system_title: str
    compatible_systems: tuple[str, ...]
    kind: str
    directory: str
    entrypoint: str
    files: tuple[str, ...]
    description: str
    agent_guidance: str
    include_in_mcp: bool = True

    def summary(self) -> ExampleSummary:
        paths = ", ".join(
            f"example_workflows/{self.directory}/{file}" for file in self.files
        )
        demonstrates = f"[{self.system_title}/{self.kind}] {self.description}"
        if self.system == "generic_flux":
            demonstrates = f"{demonstrates} {PORTABLE_SYSTEM_GUIDANCE}"
        return ExampleSummary(
            name=self.name,
            path=paths,
            demonstrates=demonstrates,
            id=self.id,
            title=self.title,
            system=self.system,
            system_title=self.system_title,
            compatible_systems=self.compatible_systems,
            agent_guidance=self.agent_guidance,
        )


EXAMPLES: tuple[ExampleSummary, ...] = (
    ExampleSummary(
        name="chores",
        path="example_workflows/generic_flux/chores/workflow.py",
        demonstrates="Python chores, OutputReference dependencies, and Pipeline.submit().",
    ),
    ExampleSummary(
        name="executable",
        path="example_workflows/generic_flux/executable/workflow.py",
        demonstrates="Pipeline.exec() for argv-style executable chores.",
    ),
    ExampleSummary(
        name="mpi",
        path="example_workflows/generic_flux/mpi/workflow.py",
        demonstrates="MPI-enabled Python chores with num_tasks and mpi=True.",
    ),
    ExampleSummary(
        name="strategy",
        path="example_workflows/generic_flux/strategy/workflow.py",
        demonstrates="Adaptive user strategies that return ChoreSpec objects.",
    ),
    ExampleSummary(
        name="perlmutter_lammps",
        path="example_workflows/perlmutter/run_lammps_mace_calculator.sh",
        demonstrates="Site batch launch pattern for running MatEnsemble inside Flux.",
    ),
)


TEXT_EXAMPLE_EXTENSIONS = {".json", ".lmp", ".md", ".py", ".sh", ".slurm"}

EXAMPLE_NAME_BY_PATH = {
    "generic_flux/chores/workflow.py": "chores",
    "generic_flux/executable/workflow.py": "executable",
    "generic_flux/mpi/workflow.py": "mpi",
    "generic_flux/strategy/workflow.py": "strategy",
    "perlmutter/lammps_mace_calculator/run_lammps_mace_calculator.sh": "perlmutter_lammps",
}


EXAMPLE_SOURCE: dict[str, str] = {
    "chores": '''\
from matensemble.pipeline import Pipeline

pipe = Pipeline()


@pipe.chore()
def factorial(n: int) -> int:
    """Calculate the factorial of a given integer."""

    product = 1
    for i in range(2, n):
        product *= i
    return product


@pipe.chore()
def digit_sum(n) -> int:
    """Calculate the sum of each digit in an integer."""

    total = 0
    for char in str(n):
        total += int(char)
    return total


fact = factorial(100)
total = digit_sum(fact)

pipe.submit(log_delay=1)
print(pipe.results())
''',
    "executable": """\
from matensemble.pipeline import Pipeline

pipe = Pipeline()

for _ in range(10):
    pipe.exec(command=["echo", "Hello, World!"], num_tasks=10)

pipe.submit()
""",
    "mpi": """\
from matensemble.pipeline import Pipeline
from mpi4py import MPI

pipe = Pipeline()


@pipe.chore(num_tasks=10, cores_per_task=1, gpus_per_task=0, mpi=True)
def mpi_hello_world():
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()

    print(f"Hello World! I am process {rank} of {size} on {name}.")


for _ in range(10):
    mpi_hello_world()

pipe.submit(log_delay=1)
""",
    "strategy": """\
from matensemble.model import Resources
from matensemble.pipeline import Pipeline
from matensemble.chore import ChoreSpec

pipe = Pipeline()


@pipe.chore()
def guess(lower: int, upper: int, guess_num: int = 1) -> dict:
    return {
        "guess": ((lower + upper) // 2),
        "low": lower,
        "high": upper,
        "num_guesses": guess_num,
    }


@pipe.strategy(bolo_list=["guess"])
def higher_or_lower(guess_result, ans=42):
    if guess_result["guess"] == ans:
        return None
    if guess_result["guess"] < ans:
        return ChoreSpec(
            args=(
                guess_result["guess"] + 1,
                guess_result["high"],
                guess_result["num_guesses"] + 1,
            ),
            kwargs={},
            resources=Resources(),
            qualname="guess",
        )
    return ChoreSpec(
        args=(
            guess_result["low"],
            guess_result["guess"] - 1,
            guess_result["num_guesses"] + 1,
        ),
        kwargs={},
        resources=Resources(),
        qualname="guess",
    )


guess(1, 100)
future = pipe.submit(log_delay=1)
print(future.result())
""",
    "perlmutter_lammps": """\
#!/usr/bin/bash

# Minimal shape only. Site/account/container details must be reviewed before use.
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH --qos debug
#SBATCH -t 0:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest

IMAGE="ghcr.io/freddude2004/matensemble:perlmutter-dev"
FLUX_CMD="python workflow.py"

srun -N "${SLURM_NNODES}" -n "${SLURM_NNODES}" --mpi=pmi2 \\
  podman-hpc run --rm --network=host --ipc=host -v "$PWD:$PWD" -w "$PWD" \\
  "${IMAGE}" flux start ${FLUX_CMD}
""",
}


EXAMPLE_ALIASES = {
    "chores": "generic_flux.chores",
    "common_chores": "generic_flux.chores",
    "generic_flux_chores": "generic_flux.chores",
    "executable": "generic_flux.executable",
    "common_executable": "generic_flux.executable",
    "generic_flux_executable": "generic_flux.executable",
    "mpi": "generic_flux.mpi",
    "common_mpi": "generic_flux.mpi",
    "generic_flux_mpi": "generic_flux.mpi",
    "strategy": "generic_flux.strategy",
    "common_strategy": "generic_flux.strategy",
    "generic_flux_strategy": "generic_flux.strategy",
    "frontier_lammps_smoke": "frontier.lammps_smoke",
    "perlmutter_lammps_smoke": "perlmutter.lammps_smoke",
    "perlmutter_lammps_mace": "perlmutter.lammps_mace",
    "perlmutter_dependency_campaign": "perlmutter.dependency_campaign",
}


API_GUIDANCE = """\
MatEnsemble workflows are built with matensemble.pipeline.Pipeline.

Core patterns:
- Create a Pipeline with pipe = Pipeline().
- Use @pipe.chore(...) on top-level Python functions to create delayed Python chores.
- Calling a decorated chore appends work and returns an OutputReference.
- Pass OutputReference values to later Python chores to create dependencies.
- Use pipe.exec(command=[...]) for executable chores. Prefer argv lists over shell strings.
- Use pipe.strategy(bolo_list=[...]) for adaptive strategies that can return ChoreSpec.
- Call pipe.submit(...) only after the graph has been constructed.
- Call future.result() or pipe.results() to collect Python chore outputs.

Operational expectations:
- Pipeline.submit() expects a Flux-capable runtime environment.
- For Slurm systems, launch a batch allocation that starts Flux, then run the workflow script.
- Inspect status.json, matensemble_workflow.log, and per-chore stdout/stderr under the generated workflow directory.
"""


CONTAINER_CONTENTS: dict[str, str] = {
    "linux": """\
FROM docker.io/ubuntu:24.04

# Generic MatEnsemble Linux container context:
# - Ubuntu 24.04
# - Python 3.12 managed by uv in /opt/basic
# - MPICH and mpi4py
# - Flux core v0.70.0
# - Flux sched / Fluxion v0.41.0
# - LAMMPS stable_22Jul2025_update4 with manybody, molecule, kspace, replica,
#   ML-SNAP, REAXFF, MPIIO, QEQ, Python, interlayer, and extra compute support
# - flux-python==0.70.0
# - MatEnsemble installed into the Python environment by the matensemble layer
# - Jupyter/ipykernel in the matensemble layer
""",
    "frontier": """\
# MatEnsemble Base
FROM savannah.ornl.gov/olcf-container-images/cpe:25.03_gnu_ubuntu

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /opt

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends               \
    build-essential autoconf automake bash ca-certificates curl git wget unzip \
    build-essential gfortran pkg-config ninja-build cmake libblas-dev libtool  \
    liblapack-dev libgsl-dev libfftw3-dev libopenblas-dev libhdf5-dev          \
    libsqlite3-dev libarchive-dev libcurl4-openssl-dev libzmq3-dev libtool-bin \
    libsodium-dev libjansson-dev libczmq-dev libyaml-cpp-dev libedit-dev       \
    libboost-all-dev uuid-dev lua5.3 liblua5.3-dev python3-venv python3-dev    \
    freeglut3-dev libgl1-mesa-dev libglu1-mesa-dev libglew-dev mesa-utils      \
    && rm -rf /var/lib/apt/lists/*

# Remove system OpenMPI so cmake cannot accidentally pick it up instead of Cray MPICH
RUN apt-get remove -y libopenmpi-dev libopenmpi3 mpi-default-dev mpi-default-bin || true

# Python env
RUN python3 -m venv /opt/basic && \
    ln -sf /opt/basic/bin/pip /usr/local/bin/pip
ENV PATH=/opt/basic/bin:$PATH

RUN python -m pip install --no-cache-dir setuptools wheel cffi \
    PyYAML cython pyarrow pyfftw tqdm papermill py3Dmol ply pyyaml

# Build mpi4py against Cray MPICH
RUN MPICC="hipcc -L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a} -I${MPICH_DIR}/include" \
    pip install --no-cache-dir --no-binary=mpi4py mpi4py

# Flux & Fluxion Build
WORKDIR /opt

RUN git clone -b v0.70.0 https://github.com/flux-framework/flux-core.git && \
    cd flux-core                                                         && \
    PYTHON=/opt/basic/bin/python ./autogen.sh                            && \
    ./configure                                                          && \
    make -j16                                                            && \
    make install                                                         && \
    ldconfig

RUN git clone -b v0.41.0 https://github.com/flux-framework/flux-sched.git && \
    cd flux-sched                                                         && \
    cmake -B build                                                        && \
    cmake --build build -j 16                                             && \
    cmake --build build -t install                                        && \
    ldconfig

RUN python -m pip install --no-cache-dir flux-python==0.70.0
ENV LD_LIBRARY_PATH=${MPICH_DIR}/lib:/usr/local/lib:${LD_LIBRARY_PATH}

# Build symmetrix
RUN git clone --recursive https://github.com/wcwitt/symmetrix && \
    cd symmetrix && \
    git checkout daeda1c1cf5eee633c3f3f409b6bba327f29b4b7 && \
    cd symmetrix && \
    pip install .

# Build LAMMPS
WORKDIR /opt
RUN wget https://github.com/lammps/lammps/archive/refs/tags/patch_10Dec2025.tar.gz && \
    tar -xzvf patch_10Dec2025.tar.gz && \
    rm -rf patch_10Dec2025.tar.gz    && \
    mv lammps-patch_10Dec2025 lammps && \
    mkdir -p /opt/lammps/build

# Patch LAMMPS to use pair_style symmetrix
WORKDIR /opt/symmetrix/pair_symmetrix
RUN ./install.sh /opt/lammps

WORKDIR /opt/lammps/build

ENV LMP_MACH=frontier_gfx90a
ENV MPICH_GPU_SUPPORT_ENABLED=1

RUN cmake ../cmake \
    -D CMAKE_INSTALL_PREFIX=${INSTDIR:-$PWD/install} \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_SHARED_LIBS=ON \
    -D BUILD_MPI=ON \
    -D CMAKE_CXX_COMPILER=${ROCM_PATH}/bin/hipcc \
    -D CMAKE_CXX_STANDARD=20 \
    -D CMAKE_CXX_STANDARD_REQUIRED=ON \
    -D CMAKE_CXX_SCAN_FOR_MODULES=OFF \
    -D CMAKE_CXX_FLAGS="-fdenormal-fp-math=ieee -fgpu-flush-denormals-to-zero -munsafe-fp-atomics -D__HIP_PLATFORM_AMD__=1" \
    -D MPI_CXX_COMPILER=${MPICH_DIR}/bin/mpicxx \
    -D MPI_C_COMPILER=${MPICH_DIR}/bin/mpicc \
    -D MPI_CXX_LIB_NAMES=mpi \
    -D MPI_C_LIB_NAMES=mpi \
    -D MPI_mpi_LIBRARY=${MPICH_DIR}/lib/libmpi.so \
    -D MPI_CXX_HEADER_DIR=${MPICH_DIR}/include \
    -D MPI_C_HEADER_DIR=${MPICH_DIR}/include \
    -D CMAKE_EXE_LINKER_FLAGS="-L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}" \
    -D CMAKE_SHARED_LINKER_FLAGS="-L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}" \
    -D KokkosKernels_ENABLE_ETI=OFF \
    -D KokkosKernels_INST_DOUBLE=OFF \
    -D KokkosKernels_INST_LAYOUTLEFT=OFF \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_HIP=ON \
    -D Kokkos_ARCH_VEGA90A=ON \
    -D Kokkos_ARCH_ZEN3=ON \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS=ON \
    -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
    -D SYMMETRIX_KOKKOS=ON \
    -D BUILD_OMP=ON \
    -D FFT=FFTW3 \
    -D BLAS_LIBRARIES="${LIBSCI_PATH}" \
    -D LAPACK_LIBRARIES="${LIBSCI_PATH}" \
    -D PYTHON_EXECUTABLE=$(which python) \
    -D PKG_MANYBODY=ON \
    -D PKG_MOLECULE=ON \
    -D PKG_KSPACE=ON \
    -D PKG_RIGID=ON \
    -D PKG_REAXFF=ON \
    -D PKG_REPLICA=ON \
    -D PKG_ML-SNAP=ON \
    -D PKG_ML-IAP=ON \
    -D PKG_ML-PACE=ON \
    -D PKG_SPIN=ON \
    -D PKG_PYTHON=ON \
    -D MLIAP_ENABLE_PYTHON=ON

RUN make -j16     && \
    make install  && \
    make install-python

ENV PATH=/opt/lammps/install/bin:${PATH}

RUN pip uninstall -y ovito && \
    pip install -U ovito

# LAMMPS / Kokkos runtime flags
ENV OMP_PROC_BIND=spread
ENV OMP_PLACES=threads
ENV HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
ENV HSA_XNACK=1
ENV MPICH_GPU_SUPPORT_ENABLED=1
ENV SLURM_GPUS_PER_NODE=8
ENV CRAYPE_LINK_TYPE=dynamic

# Sanity checks
RUN python -c "import mpi4py; print('python stack ok')"  && \
    python -c "import lammps; print('lammps python ok')" && \
    python -c "import ovito; print('ovito python ok')"   && \
    python -c "import flux; print('flux python ok')"
""",
    "perlmutter": """\

# lammps:26.05
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




# MatEnsemble Base
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
""",
    "pathfinder": """\

# MatEnsemble Base
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /opt

RUN apt-get update && apt-get install --yes \
    build-essential git wget cmake mpich \
    autoconf automake libtool make pkg-config libc6-dev libzmq3-dev uuid-dev \
    libjansson-dev liblz4-dev libarchive-dev libhwloc-dev libsqlite3-dev lua5.1 \
    liblua5.1-dev lua-posix aspell aspell-en time valgrind jq \
    libboost-dev libboost-graph-dev libedit-dev libyaml-cpp-dev \
    sudo gzip gcc g++ gfortran libucx-dev libibverbs-dev librdmacm-dev libpmix-dev \
    && apt-get clean

# Pull UV into the builder stage
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/
ENV PATH="/root/.local/bin:$PATH"
RUN uv python install 3.12
RUN uv venv /opt/basic --seed --python 3.12
ENV PATH="/opt/basic/bin:$PATH"

# OpenMPI & mpi4py
WORKDIR /opt
ENV OMPI_DIR=/opt/ompi
ENV OMPI_VERSION=4.0.4
ENV OMPI_URL="https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-$OMPI_VERSION.tar.gz"
RUN wget -O openmpi-${OMPI_VERSION}.tar.gz ${OMPI_URL} && tar -xzf openmpi-${OMPI_VERSION}.tar.gz && \
    cd openmpi-${OMPI_VERSION} && \
    ./configure --prefix=${OMPI_DIR} --with-ucx --with-pmix && \
    make install
ENV PATH=${OMPI_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=${OMPI_DIR}/lib:${LD_LIBRARY_PATH}
RUN MPICC=${OMPI_DIR}/bin/mpicc python -m pip install --no-cache-dir mpi4py
RUN uv pip install setuptools wheel cffi ply PyYAML sphinx jsonschema

# Flux-core
WORKDIR /opt
RUN git clone -b v0.70.0 https://github.com/flux-framework/flux-core.git && \
    cd flux-core && \
    PYTHON=/opt/basic/bin/python ./autogen.sh && \
    ./configure && \
    make -j8 && \
    make install && \
    ldconfig

# Flux-sched
RUN git clone -b v0.41.0 https://github.com/flux-framework/flux-sched.git && \
    cd flux-sched                                                          && \
    cmake -B build                                                         && \
    cmake --build build -j1                                                && \
    cmake --build build -t install                                         && \
    ldconfig

# LAMMPS
WORKDIR /opt
RUN wget https://github.com/lammps/lammps/archive/refs/tags/stable_22Jul2025_update4.tar.gz && \
    tar xzf stable_22Jul2025_update4.tar.gz && \
    cmake -S lammps-stable_22Jul2025_update4/cmake \
    -B lammps_build \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/opt/lammps \
    -D BUILD_SHARED_LIBS=yes \
    -D ENABLE_LTO=no \
    -D CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    -D PKG_MANYBODY=ON \
    -D PKG_MC=ON \
    -D PKG_MOLECULE=ON \
    -D PKG_KSPACE=ON \
    -D PKG_REPLICA=ON \
    -D PKG_ASPHERE=ON \
    -D PKG_ML-SNAP=ON \
    -D PKG_REAXFF=ON \
    -D PKG_RIGID=ON \
    -D PKG_MPIIO=ON \
    -D PKG_QEQ=ON \
    -D PKG_PYTHON=On \
    -D PKG_INTERLAYER=ON \
    -D PKG_EXTRA-COMPUTE=ON \
    -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
    -D CMAKE_EXE_FLAGS="-dynamic" \
    -D BUILD_LIB=on && \
    cmake --build lammps_build --target install -j4 && \
    make -C lammps_build install-python && \
    rm -f stable_22Jul2025_update4.tar.gz

RUN python -m pip install --no-cache-dir flux-python==0.70.0

# Final stage
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

# Runtime deps
RUN apt-get update && apt-get install --yes \
    mpich lua5.1 lua-posix liblz4-1 libarchive13 libhwloc15 \
    libzmq5 libjansson4 libsqlite3-0 libyaml-cpp0.8 \
    libedit2 libboost-graph1.83.0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Bring in uv and python3.12 interpreter
COPY --from=builder /bin/uv /bin/uvx /bin/
COPY --from=builder /root/.local/share/uv /root/.local/share/uv

# 3. Copy compiled artifacts
COPY --from=builder /opt/ompi /opt/ompi
COPY --from=builder /opt/basic /opt/basic
COPY --from=builder /opt/lammps /opt/lammps
COPY --from=builder /usr/local /usr/local

ENV PATH="/opt/ompi/bin:/opt/basic/bin:/opt/lammps/bin:$PATH"
ENV LD_LIBRARY_PATH=/opt/ompi/lib:/usr/local/lib:/opt/lammps/lib:${LD_LIBRARY_PATH:-}
ENV PYTHONPATH=/opt/lammps/lib/python3.12/site-packages:${PYTHONPATH:-}
""",
}


def list_examples() -> list[dict[str, object]]:
    manifest = _load_example_manifest()
    if manifest:
        return [entry.summary().__dict__ for entry in manifest if entry.include_in_mcp]
    return [example.__dict__ for example in EXAMPLES]


def get_example_source(name: str) -> str:
    entries = _load_example_manifest()
    manifest = _manifest_lookup(entries)
    key = EXAMPLE_ALIASES.get(name, name)
    if key in manifest:
        root = _repo_root()
        if root is None:
            raise ValueError("repository examples are unavailable in this installation")
        entry = manifest[key]
        parts = _example_header(entry)
        base_dir = root / "example_workflows" / entry.directory
        for rel in entry.files:
            path = base_dir / rel
            parts.extend(
                [
                    f"# --- {path.relative_to(root)} ---",
                    path.read_text(encoding="utf-8"),
                    "",
                ]
            )
        return "\n".join(parts)

    fallback_key = name
    if key in EXAMPLE_SOURCE:
        fallback_key = key
    if fallback_key not in EXAMPLE_SOURCE:
        expected = sorted(set(EXAMPLE_SOURCE) | set(manifest) | set(EXAMPLE_ALIASES))
        raise ValueError(f"unknown example {name!r}; expected one of {expected}")
    return EXAMPLE_SOURCE[fallback_key]


def _example_header(entry: ExampleManifestEntry) -> list[str]:
    parts = [
        f"# Example: {entry.id}",
        f"# Title: {entry.title}",
        f"# System: {entry.system}",
        f"# System title: {entry.system_title}",
        f"# Kind: {entry.kind}",
        f"# Entry point: {entry.entrypoint}",
        f"# Description: {entry.description}",
    ]
    if entry.compatible_systems:
        parts.append(f"# Compatible systems: {', '.join(entry.compatible_systems)}")
    if entry.agent_guidance:
        parts.append(f"# Agent guidance: {entry.agent_guidance}")
    if entry.system == "generic_flux":
        parts.extend(
            [
                "#",
                f"# {PORTABLE_SYSTEM_GUIDANCE}",
                "# This workflow pattern is reusable on Frontier, Perlmutter, Pathfinder,",
                "# Linux containers, and generic Flux runtimes after adapting launch/runtime details.",
                "# Use get_matensemble_system(<site>) or site-specific examples for launch context.",
            ]
        )
    else:
        parts.extend(
            [
                "#",
                "# This is a system-specific example. Use portable generic_flux examples for",
                "# core MatEnsemble workflow structure, and use this example for launch/runtime context.",
            ]
        )
    parts.append("")
    return parts


def _load_example_manifest() -> tuple[ExampleManifestEntry, ...]:
    root = _repo_root()
    if root is None:
        return ()
    manifest_path = root / "example_workflows" / "examples.toml"
    if not manifest_path.exists():
        return ()

    data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    entries = []
    for raw in data.get("example", []):
        files = tuple(str(file) for file in raw.get("files", ()))
        directory = str(raw.get("directory", "")).strip()
        entrypoint = str(raw.get("entrypoint", "")).strip()
        if not files or not directory or not entrypoint:
            continue
        base_dir = root / "example_workflows" / directory
        if not all((base_dir / file).is_file() for file in files):
            continue
        if not (base_dir / entrypoint).is_file():
            continue
        system = str(raw["system"])
        compatible_systems = tuple(
            str(item) for item in raw.get("compatible_systems", ())
        )
        if system == "generic_flux" and not compatible_systems:
            compatible_systems = GENERIC_FLUX_COMPATIBLE_SYSTEMS
        system_title = str(raw.get("system_title") or _default_system_title(system))
        description = str(raw["description"])
        entries.append(
            ExampleManifestEntry(
                id=str(raw.get("id") or raw["name"]),
                name=str(raw["name"]),
                title=str(raw.get("title") or raw["name"]),
                system=system,
                system_title=system_title,
                compatible_systems=compatible_systems,
                kind=str(raw["kind"]),
                directory=directory,
                entrypoint=entrypoint,
                files=files,
                description=description,
                agent_guidance=str(raw.get("agent_guidance") or description),
                include_in_mcp=bool(raw.get("include_in_mcp", True)),
            )
        )
    return tuple(entries)


def _manifest_lookup(
    entries: tuple[ExampleManifestEntry, ...],
) -> dict[str, ExampleManifestEntry]:
    lookup = {}
    for entry in entries:
        lookup[entry.id] = entry
        lookup[entry.name] = entry
    return lookup


def _default_system_title(system: str) -> str:
    if system == "generic_flux":
        return "Portable Flux Workflows"
    return system.replace("_", " ").title()


def _discover_repo_examples() -> tuple[ExampleSummary, ...]:
    root = _repo_root()
    if root is None:
        return ()

    example_root = root / "example_workflows"
    discovered: list[ExampleSummary] = []
    seen: set[str] = set()
    for path in sorted(example_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in TEXT_EXAMPLE_EXTENSIONS:
            continue
        if any(part.startswith("matensemble_workflow-") for part in path.parts):
            continue

        rel = path.relative_to(example_root)
        repo_rel = path.relative_to(root)
        name = _example_name(rel)
        if name in seen:
            continue
        seen.add(name)
        discovered.append(
            ExampleSummary(
                name=name,
                path=str(repo_rel),
                demonstrates=_describe_example(rel),
            )
        )

    return tuple(discovered)


def _repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "example_workflows").is_dir() and (
            parent / "pyproject.toml"
        ).is_file():
            return parent
    return None


def _example_name(relative_path: Path) -> str:
    posix = relative_path.as_posix()
    if posix in EXAMPLE_NAME_BY_PATH:
        return EXAMPLE_NAME_BY_PATH[posix]
    return "__".join(relative_path.with_suffix("").parts).replace("-", "_")


def _describe_example(relative_path: Path) -> str:
    parts = set(relative_path.parts)
    suffix = relative_path.suffix
    if "frontier" in parts and suffix == ".slurm":
        return "Frontier Slurm batch script using the MatEnsemble CLI."
    if "frontier" in parts and suffix == ".sh":
        return "Frontier interactive launch helper using the MatEnsemble CLI."
    if "frontier" in parts and suffix == ".py":
        return "Frontier MatEnsemble workflow example."
    if "perlmutter" in parts and suffix == ".slurm":
        return "Perlmutter Slurm batch script using the MatEnsemble CLI."
    if "perlmutter" in parts and suffix == ".sh":
        return "Perlmutter launch helper using the MatEnsemble CLI or Flux."
    if "perlmutter" in parts and suffix == ".py":
        return "Perlmutter MatEnsemble workflow example."
    if suffix == ".json":
        return "Workflow configuration file."
    if suffix == ".md":
        return "Workflow documentation."
    if suffix == ".lmp":
        return "LAMMPS input/structure artifact used by an example."
    return "MatEnsemble workflow example."


def get_container_contents(name: str) -> str:
    """
    Gets information on the packages that are available in the container for a
    certain system
    """

    key = name.strip().lower().replace("-", "_")
    if key in ("generic", "generic_flux", "conda", "local", "local_conda"):
        return (
            "No MatEnsemble container Dockerfile is required for this environment. "
            "Use the conda/environment.yaml path or an existing Flux-capable Python environment."
        )
    if key not in CONTAINER_CONTENTS:
        raise ValueError(
            f"unknown container {name!r}; expected one of {sorted(CONTAINER_CONTENTS)}"
        )
    return CONTAINER_CONTENTS[key]


def how_to_build_container(name: str) -> str:
    """
    Gives instructions for how to pull or build a container for a specific system
    """
    from mcp_matensemble.systems import render_environment_setup

    return render_environment_setup(name)
