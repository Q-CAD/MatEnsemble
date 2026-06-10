#!/usr/bin/env bash
set -euo pipefail

SYSTEM=""
VERSION=""
WORKSPACE="${SCRATCH:-}/matensemble_campaigns"
if [[ -z "${SCRATCH:-}" ]]; then
  WORKSPACE="$HOME/matensemble_campaigns"
fi
INSTALL_DIR="${HOME}/.local/bin"

usage() {
  cat <<'USAGE'
Usage:
  install-matensemble-agent.sh --system <frontier|perlmutter|pathfinder|linux|conda> [--version X.Y.Z] [--workspace PATH]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system) SYSTEM="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --install-dir) INSTALL_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

if [[ -z "$SYSTEM" ]]; then
  usage >&2
  exit 2
fi

mkdir -p "$INSTALL_DIR"
export PATH="$INSTALL_DIR:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv to $INSTALL_DIR"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$INSTALL_DIR" sh
fi

PACKAGE="mcp-matensemble"
if [[ -n "$VERSION" ]]; then
  PACKAGE="mcp-matensemble==${VERSION#v}"
fi

uv tool install --upgrade "$PACKAGE"
matensemble-agent-install --system "$SYSTEM" --workspace "$WORKSPACE" --install-dir "$INSTALL_DIR"

echo
echo "Open this folder with VS Code Remote:"
echo "  $WORKSPACE"
