#!/usr/bin/env bash
set -euo pipefail

# MatEnsemble CLI installer.
# Intended remote use:
#   curl -fsSL https://raw.githubusercontent.com/<ORG>/<REPO>/main/cli/install.sh | bash

REPO_RAW_BASE="${MATENSEMBLE_CLI_RAW_BASE:-https://raw.githubusercontent.com/Q-CAD/MatEnsemble/main/src/cli}"
INSTALL_DIR="${MATENSEMBLE_INSTALL_DIR:-$HOME/.local/bin}"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/matensemble"
TARGET="$INSTALL_DIR/matensemble"

mkdir -p "$INSTALL_DIR" "$CONFIG_DIR"

printf 'Install MatEnsemble CLI for which system?\n'
printf '  1) Frontier \n'
printf '  2) Perlmutter \n'
printf '  3) Pathfinder \n'
printf 'Choice [1-3]: '

read -r choice </dev/tty

case "$choice" in
1 | frontier | Frontier)
	SYSTEM="frontier"
	SCRIPT_URL="$REPO_RAW_BASE/matensemble-frontier"
	;;
2 | perlmutter | Perlmutter)
	SYSTEM="perlmutter"
	SCRIPT_URL="$REPO_RAW_BASE/matensemble-perlmutter"
	;;
3 | pathfinder | Pathfinder)
	SYSTEM="pathfinder"
	SCRIPT_URL="$REPO_RAW_BASE/matensemble-pathfinder"
	;;
*)
	echo "error: expected 1/frontier, 2/perlmutter or 3/pathfinder" >&2
	exit 2
	;;
esac

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
curl -fsSL "$SCRIPT_URL" -o "$tmp"
install -m 0755 "$tmp" "$TARGET"

cat >"$CONFIG_DIR/system" <<EOF_SYSTEM
$SYSTEM
EOF_SYSTEM

echo "Installed MatEnsemble CLI for $SYSTEM at $TARGET"
case ":$PATH:" in
*":$INSTALL_DIR:"*) ;;
*)
	echo "Add this to your shell rc file if needed:"
	echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
	;;
esac

echo
if [ "$SYSTEM" = "perlmutter" ]; then
	echo "Next: matensemble set-image ghcr.io/q-cad/matensemble:perlmutter-vX.Y.Z"
else
	echo "Next: matensemble set-image /path/to/matensemble.sif"
fi
