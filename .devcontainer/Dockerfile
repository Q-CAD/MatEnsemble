FROM ghcr.io/freddude2004/matensemble-base:baseline-arm64

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /opt

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv --version

# setup uv to install deps to /opt/basic
ENV UV_PROJECT_ENVIRONMENT=/opt/basic

# clone MatEnsemble
RUN git clone --depth 1 https://github.com/FredDude2004/MatEnsemble.git matensemble && \
    cd matensemble 						       && \
    uv export --format requirements.txt --output-file requirements.txt && \ 
    uv pip install -r requirements.txt

ENV PATH="/opt/basic/bin:${PATH}"
CMD ["flux", "start", "--test-size=10"]
