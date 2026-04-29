FROM nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    autoconf automake libtool make pkg-config libc6-dev libzmq3-dev uuid-dev \
    libjansson-dev liblz4-dev libarchive-dev libsqlite3-dev lua5.1 \
    liblua5.1-dev lua-posix python3-dev python3-cffi python3-ply aspell \
    python3-setuptools python3-yaml python3-sphinx aspell-en time jq valgrind \
    libboost-dev libboost-graph-dev libedit-dev libyaml-cpp-dev wget cmake \
    python3-jsonschema \
    && apt-get clean all

WORKDIR /opt
RUN wget https://github.com/open-mpi/hwloc/releases/download/hwloc-2.13.0/hwloc-2.13.0.tar.gz \
    && tar xzf hwloc-2.13.0.tar.gz \
    && rm hwloc-2.13.0.tar.gz \
    && cd hwloc-2.13.0 \
    && ./configure --enable-cuda --enable-nvml --prefix=/opt/hwloc \
    && make \
    && make install
ENV PATH=$PATH:/opt/hwloc/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/hwloc/lib
ENV PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/opt/hwloc/lib/pkgconfig
ENV MANPATH=$MANPATH:/opt/hwloc/share/man

# Install flux-core
WORKDIR /opt
RUN wget https://github.com/flux-framework/flux-core/releases/download/v0.70.0/flux-core-0.70.0.tar.gz \
    && tar xzf flux-core-0.70.0.tar.gz \
    && rm flux-core-0.70.0.tar.gz \
    && cd flux-core-0.70.0 \
    && ./configure \
    && make -j1 \
    && make install
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
ENV LD_RUN_PATH=$LD_RUN_PATH:/usr/local/lib

WORKDIR /opt
RUN wget https://github.com/flux-framework/flux-sched/releases/download/v0.41.0/flux-sched-0.41.0.tar.gz \
    && tar xzf flux-sched-0.41.0.tar.gz \
    && rm flux-sched-0.41.0.tar.gz \
    && cd flux-sched-0.41.0 \
    && cmake -B build \
    && make -C build \
    && make -C build install
