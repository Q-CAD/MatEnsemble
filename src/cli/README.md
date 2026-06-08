# MatEnsemble CLI prototype

Files:

- `install.sh`: remote bootstrap installer.
- `matensemble-frontier`: Frontier/OLCF CLI using Apptainer.
- `matensemble-perlmutter`: Perlmutter/NERSC CLI using podman-hpc and the generated Flux resource config pattern.

Expected GitHub layout:

```text
cli/
  install.sh
  matensemble-frontier
  matensemble-perlmutter
```

Remote install:

```bash
curl -fsSL https://raw.githubusercontent.com/<ORG>/<REPO>/main/cli/install.sh | bash
```

For testing from another branch or fork:

```bash
MATENSEMBLE_CLI_RAW_BASE=https://raw.githubusercontent.com/<ORG>/<REPO>/<BRANCH>/cli \
  bash install.sh
```
