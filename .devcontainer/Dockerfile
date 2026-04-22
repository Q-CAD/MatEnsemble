FROM ghcr.io/freddude2004/matensemble-base:baseline-arm64

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /opt

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv --version

ENV UV_PROJECT_ENVIRONMENT=/opt/basic
RUN uv pip install IPython jupyter ipykernel
ENV PATH="/opt/basic/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y curl ca-certificates gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null && \
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=arm64 signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list && \
    apt-get update && \
    apt-get install -y gh && \
    rm -rf /var/lib/apt/lists/*