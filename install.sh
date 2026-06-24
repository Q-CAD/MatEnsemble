#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${MATENSEMBLE_REPO_URL:-https://github.com/FredDude2004/MatEnsemble.git}"
GHCR_NAMESPACE="${MATENSEMBLE_GHCR_NAMESPACE:-ghcr.io/freddude2004/matensemble}"

err() {
	echo "install.sh: error: $*" >&2
	exit 1
}

prompt_yes_no() {
	local prompt="$1"
	local default="$2"
	local answer
	while true; do
		read -r -p "$prompt" answer
		answer="${answer:-$default}"
		case "$answer" in
		y | Y | yes | YES) return 0 ;;
		n | N | no | NO) return 1 ;;
		*) echo "Please answer y or n." ;;
		esac
	done
}

expand_path() {
	local path="$1"
	case "$path" in
	"~") echo "$HOME" ;;
	"~/"*) echo "$HOME/${path#~/}" ;;
	*) echo "$path" ;;
	esac
}

choose_system() {
	local choice
	echo "Which system are you on?"
	echo "  1. Frontier"
	echo "  2. Perlmutter"
	echo "  3. Pathfinder"
	read -r -p "[1-3]: " choice
	case "$choice" in
	1 | frontier | Frontier) echo "frontier" ;;
	2 | perlmutter | Perlmutter) echo "perlmutter" ;;
	3 | pathfinder | Pathfinder) echo "pathfinder" ;;
	*) err "expected 1/frontier, 2/perlmutter, or 3/pathfinder" ;;
	esac
}

choose_base() {
	local path
	if [[ -n "${SCRATCH:-}" ]]; then
		if prompt_yes_no "Install MatEnsemble in \$SCRATCH (${SCRATCH})? [Y/n] " "Y"; then
			echo "$SCRATCH"
			return
		fi
	else
		if prompt_yes_no "SCRATCH environment variable is not set. Install MatEnsemble in \$PWD (${PWD})? [y/N] " "N"; then
			echo "$PWD"
			return
		fi
	fi

	read -r -p "Provide path to install MatEnsemble: " path
	[[ -n "$path" ]] || err "install path is required"
	echo "$path"
}

clone_or_reuse_repo() {
	local repo_dir="$1"
	mkdir -p "$(dirname "$repo_dir")"
	if [[ -d "$repo_dir/.git" ]]; then
		echo "Using existing MatEnsemble checkout: $repo_dir"
		return
	fi
	if [[ -e "$repo_dir" ]] && [[ -n "$(find "$repo_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
		err "$repo_dir exists and is not an empty git checkout"
	fi
	git clone "$REPO_URL" "$repo_dir"
}

ensure_uv() {
	if command -v uv >/dev/null 2>&1; then
		echo "Found uv: $(command -v uv)"
		return
	fi
	if prompt_yes_no "uv is not installed. Install uv now? [Y/n] " "Y"; then
		curl -LsSf https://astral.sh/uv/install.sh | sh
	else
		err "uv is required"
	fi
	command -v uv >/dev/null 2>&1 || err "uv was not found after installation; restart your shell or add uv to PATH"
}

install_cli() {
	local repo_dir="$1"
	local system="$2"
	local source="$repo_dir/src/cli/matensemble-$system"
	local target_dir="${HOME}/.local/bin"
	local target="$target_dir/matensemble"
	[[ -f "$source" ]] || err "CLI script not found: $source"
	mkdir -p "$target_dir"
	install -m 0755 "$source" "$target"
	echo "Installed MatEnsemble CLI for $system at $target"
	case ":$PATH:" in
	*":$target_dir:"*) ;;
	*) echo "Add this to your shell rc file if needed: export PATH=\"$target_dir:\$PATH\"" ;;
	esac
}

matensemble_version() {
	local repo_dir="$1"
	awk -F '"' '/^version = / { print $2; exit }' "$repo_dir/pyproject.toml"
}

container_command() {
	local install_root="$1"
	local system="$2"
	local version="$3"
	local image="${GHCR_NAMESPACE}:${system}-v${version}"
	if [[ "$system" == "perlmutter" ]]; then
		echo "podman-hpc pull $image"
	else
		printf 'apptainer build %q %q\n' "${install_root}/containers/${system}/matensemble.sif" "docker://${image}"
	fi
}

run_container_command() {
	local install_root="$1"
	local system="$2"
	local version="$3"
	local image="${GHCR_NAMESPACE}:${system}-v${version}"
	if [[ "$system" == "perlmutter" ]]; then
		podman-hpc pull "$image"
	else
		apptainer build "${install_root}/containers/${system}/matensemble.sif" "docker://${image}"
	fi
}

write_configs() {
	local install_root="$1"
	local repo_dir="$2"
	local campaigns_dir="$3"
	local system="$4"
	local codex_dir="$campaigns_dir/.codex"
	local vscode_dir="$campaigns_dir/.vscode"

	mkdir -p "$codex_dir" "$vscode_dir"

	cat >"$codex_dir/config.toml" <<EOF_CODEX
[mcp_servers.matensemble]
command = "uv"
args = [
  "run",
  "--directory",
  "$repo_dir",
  "--package",
  "mcp-matensemble",
  "mcp-matensemble",
  "--system",
  "$system",
]
cwd = "$campaigns_dir"
startup_timeout_sec = 120
EOF_CODEX

	cat >"$campaigns_dir/.mcp.json" <<EOF_CLAUDE
{
  "mcpServers": {
    "matensemble": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "$repo_dir",
        "--package",
        "mcp-matensemble",
        "mcp-matensemble",
        "--system",
        "$system"
      ],
      "cwd": "$campaigns_dir"
    }
  }
}
EOF_CLAUDE

	cat >"$vscode_dir/mcp.json" <<EOF_VSCODE
{
  "servers": {
    "matensemble": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "$repo_dir",
        "--package",
        "mcp-matensemble",
        "mcp-matensemble",
        "--system",
        "$system"
      ],
      "cwd": "$campaigns_dir"
    }
  }
}
EOF_VSCODE

	cat >"$campaigns_dir/README.md" <<EOF_README
# MatEnsemble Campaigns

This workspace is configured for the MatEnsemble MCP server on \`$system\`.

The MatEnsemble checkout lives at:

\`\`\`text
$repo_dir
\`\`\`

The MCP server command is:

\`\`\`bash
uv run --directory "$repo_dir" --package mcp-matensemble mcp-matensemble --system "$system"
\`\`\`
EOF_README

	echo "Wrote MCP configs under $campaigns_dir"
}

main() {
	local system
	local base
	local install_root
	local repo_dir
	local campaigns_dir
	local version
	local command_text

	system="$(choose_system)"
	base="$(choose_base)"
	base="$(expand_path "$base")"
	base="$(cd "$base" 2>/dev/null && pwd || {
		mkdir -p "$base"
		cd "$base"
		pwd
	})"
	install_root="$base/MatEnsemble"
	repo_dir="$install_root/.matensemble"
	campaigns_dir="$install_root/matensemble_campaigns"

	mkdir -p "$install_root" "$campaigns_dir" "$install_root/containers/$system"
	clone_or_reuse_repo "$repo_dir"
	ensure_uv

	version="$(matensemble_version "$repo_dir")"
	[[ -n "$version" ]] || err "could not read MatEnsemble version from $repo_dir/pyproject.toml"

	if prompt_yes_no "Install the MatEnsemble CLI tool? [Y/n] " "Y"; then
		install_cli "$repo_dir" "$system"
	fi

	if prompt_yes_no "Install the MatEnsemble MCP server config files? [Y/n] " "Y"; then
		write_configs "$install_root" "$repo_dir" "$campaigns_dir" "$system"
	fi

	command_text="$(container_command "$install_root" "$system" "$version")"
	if prompt_yes_no "Build or pull the MatEnsemble container now? [Y/n] " "Y"; then
		echo "Running: $command_text"
		run_container_command "$install_root" "$system" "$version"
	else
		echo "Container command for later:"
		echo "  $command_text"
	fi

	echo
	echo "MatEnsemble install root: $install_root"
	echo "Repository checkout: $repo_dir"
	echo "Campaign workspace: $campaigns_dir"
}

main "$@"
