#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ] || [ "$(basename "${BASH:-sh}")" = "sh" ]; then
	echo "install.sh: please run this installer with bash, for example:" >&2
	echo "  curl -fsSL https://raw.githubusercontent.com/Q-CAD/MatEnsemble/refs/heads/main/install.sh | bash" >&2
	exit 2
fi

set -euo pipefail

REPO_URL="${MATENSEMBLE_REPO_URL:-https://github.com/FredDude2004/MatEnsemble.git}"
IMAGE_REPOSITORY="${MATENSEMBLE_IMAGE_REPOSITORY:-ghcr.io/freddude2004/matensemble}"

err() {
	echo "install.sh: error: $*" >&2
	exit 1
}

prompt_read() {
	local prompt="$1"
	local value
	[[ -r /dev/tty ]] || err "interactive prompts require a terminal; run this script from an interactive shell"
	read -r -p "$prompt" value </dev/tty
	printf '%s\n' "$value"
}

prompt_yes_no() {
	local prompt="$1"
	local default_answer="${2:-yes}"
	local answer
	while true; do
		answer="$(prompt_read "$prompt")"
		answer="${answer:-$default_answer}"
		case "$answer" in
		y | Y | yes | YES) return 0 ;;
		n | N | no | NO) return 1 ;;
		*) echo "Please answer y or n." >&2 ;;
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

choose_install_root() {
	local default_parent="${SCRATCH:-$PWD}"
	local path
	path="$(prompt_read "Which directory should contain the MatEnsemble folder? [$default_parent]: ")"
	path="${path:-$default_parent}"
	path="$(expand_path "$path")"

	# Accept an explicitly named MatEnsemble directory without adding it twice.
	if [[ "${path%/}" == */MatEnsemble ]]; then
		echo "${path%/}"
	else
		echo "${path%/}/MatEnsemble"
	fi
}

choose_system() {
	local choice
	while true; do
		echo "Which system are you on?" >&2
		echo "  1. Frontier" >&2
		echo "  2. Pathfinder" >&2
		echo "  3. Perlmutter" >&2
		echo "  4. Linux" >&2
		choice="$(prompt_read "Choose a system [1-4]: ")"
		case "$choice" in
		1 | frontier | Frontier) echo "frontier"; return ;;
		2 | pathfinder | Pathfinder) echo "pathfinder"; return ;;
		3 | perlmutter | Perlmutter) echo "perlmutter"; return ;;
		4 | linux | Linux) echo "linux"; return ;;
		*) echo "Please choose 1, 2, 3, or 4." >&2 ;;
		esac
	done
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
	git clone --depth=1 "$REPO_URL" "$repo_dir"
}

ensure_uv() {
	if command -v uv >/dev/null 2>&1; then
		return
	fi
	echo "uv is required by the MCP server; installing it now."
	command -v curl >/dev/null 2>&1 || err "curl is required to install uv"
	curl -LsSf https://astral.sh/uv/install.sh | sh
	[[ -x "$HOME/.local/bin/uv" ]] || err "uv was not found after installation"
	export PATH="$HOME/.local/bin:$PATH"
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
	awk -F '"' '/^version = / { print $2; exit }' "$1/pyproject.toml"
}

detect_container_engine() {
	local engine
	for engine in apptainer docker podman podman-hpc; do
		if command -v "$engine" >/dev/null 2>&1; then
			echo "$engine"
			return 0
		fi
	done
	return 1
}

pull_container() {
	local repo_dir="$1"
	local install_root="$2"
	local system="$3"
	local version image engine container_dir container_file

	version="$(matensemble_version "$repo_dir")"
	[[ -n "$version" ]] || err "could not read the MatEnsemble version"
	image="$IMAGE_REPOSITORY:${system}-v${version}"

	case "$system" in
	frontier | pathfinder)
		command -v apptainer >/dev/null 2>&1 || err "apptainer is required to install the $system image"
		container_dir="$install_root/containers/$system"
		container_file="$container_dir/matensemble-v${version}.sif"
		mkdir -p "$container_dir"
		if [[ -f "$container_file" ]]; then
			echo "Using existing MatEnsemble image: $container_file"
		else
            echo "Pulling image: $image"
			apptainer pull "$container_file" "docker://$image"
		fi
		install_cli "$repo_dir" "$system"
		"$HOME/.local/bin/matensemble" set-image "$container_file"
		;;
	perlmutter)
		command -v podman-hpc >/dev/null 2>&1 || err "podman-hpc is required to install the Perlmutter image"
        echo "Pulling image: $image"
		podman-hpc pull "$image"
		install_cli "$repo_dir" "$system"
		"$HOME/.local/bin/matensemble" set-image "$image"
		;;
	linux)
		engine="$(detect_container_engine)" || err "no supported container engine found (checked apptainer, docker, podman, podman-hpc)"
		echo "Using detected container engine: $engine"
        echo "Pulling image: $image"
		if [[ "$engine" == "apptainer" ]]; then
			container_dir="$install_root/containers/linux"
			container_file="$container_dir/matensemble-v${version}.sif"
			mkdir -p "$container_dir"
			if [[ -f "$container_file" ]]; then
				echo "Using existing MatEnsemble image: $container_file"
			else
				apptainer pull "$container_file" "docker://$image"
			fi
		else
			"$engine" pull "$image"
		fi
		;;
	esac
}

write_configs() {
	local repo_dir="$1"
	local campaigns_dir="$2"
	local system="$3"
	local uv_command="$4"
	local codex_dir="$campaigns_dir/.codex"
	local copilot_dir="$campaigns_dir/.copilot"
	local gemini_dir="$campaigns_dir/.gemini"
	local vscode_dir="$campaigns_dir/.vscode"
	local args_json

	mkdir -p "$codex_dir" "$copilot_dir" "$gemini_dir" "$vscode_dir"

	cat >"$codex_dir/config.toml" <<EOF_CODEX
[mcp_servers.matensemble]
command = "$uv_command"
args = ["run", "--directory", "$repo_dir", "--package", "mcp-matensemble", "mcp-matensemble", "--system", "$system"]
cwd = "$campaigns_dir"
startup_timeout_sec = 120
EOF_CODEX

	args_json="[\"run\", \"--directory\", \"$repo_dir\", \"--package\", \"mcp-matensemble\", \"mcp-matensemble\", \"--system\", \"$system\"]"
	cat >"$campaigns_dir/.mcp.json" <<EOF_CLAUDE
{"mcpServers":{"matensemble":{"command":"$uv_command","args":$args_json,"cwd":"$campaigns_dir"}}}
EOF_CLAUDE
	cat >"$copilot_dir/mcp-config.json" <<EOF_COPILOT
{"mcpServers":{"matensemble":{"type":"local","command":"$uv_command","args":$args_json,"cwd":"$campaigns_dir","env":{},"tools":["*"]}}}
EOF_COPILOT
	cat >"$gemini_dir/settings.json" <<EOF_GEMINI
{"mcpServers":{"matensemble":{"command":"$uv_command","args":$args_json,"cwd":"$campaigns_dir","env":{},"timeout":120000}}}
EOF_GEMINI
	cat >"$vscode_dir/mcp.json" <<EOF_VSCODE
{"servers":{"matensemble":{"type":"stdio","command":"$uv_command","args":$args_json,"cwd":"$campaigns_dir"}}}
EOF_VSCODE
	cat >"$campaigns_dir/README.md" <<EOF_README
# MatEnsemble Campaigns

This workspace is configured for the MatEnsemble MCP server on \`$system\`.
The MatEnsemble checkout lives at \`$repo_dir\`.
EOF_README
	echo "Wrote MCP configs under $campaigns_dir"
}

main() {
	local install_root repo_dir campaigns_dir system="linux"
	local write_mcp="no" uv_command=""

	install_root="$(choose_install_root)"
	mkdir -p "$install_root"
	install_root="$(cd "$install_root" && pwd)"
	repo_dir="$install_root/.matensemble"
	campaigns_dir="$install_root/matensemble_campaigns"

	command -v git >/dev/null 2>&1 || err "git is required"
	clone_or_reuse_repo "$repo_dir"

	if prompt_yes_no "Would you like to write the MCP configuration files? [Y/n] "; then
		write_mcp="yes"
	fi
	if prompt_yes_no "Would you like to pull the most recent MatEnsemble image? [Y/n] "; then
		system="$(choose_system)"
		pull_container "$repo_dir" "$install_root" "$system"
	fi

	if [[ "$write_mcp" == "yes" ]]; then
		mkdir -p "$campaigns_dir"
		ensure_uv
		uv_command="$(command -v uv)"
		write_configs "$repo_dir" "$campaigns_dir" "$system" "$uv_command"
	fi

	echo
	echo "MatEnsemble install root: $install_root"
	echo "Repository checkout: $repo_dir"
	[[ "$write_mcp" == "yes" ]] && echo "Campaign workspace: $campaigns_dir"
	[[ "$write_mcp" == "yes" ]] && echo "Next: cd \"$campaigns_dir\" and start your preferred agent."
}

main "$@"
