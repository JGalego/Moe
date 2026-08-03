#!/bin/sh
# Install the moe binary. Usage:
#   curl -fsSL https://raw.githubusercontent.com/JGalego/Moe/main/install.sh | sh
# Environment:
#   MOE_VERSION   release tag to install (default: latest)
#   MOE_BIN_DIR   install location (default: ~/.local/bin)
set -eu

repo=JGalego/Moe
version="${MOE_VERSION:-latest}"
bin_dir="${MOE_BIN_DIR:-$HOME/.local/bin}"

os=$(uname -s)
arch=$(uname -m)
case "$os-$arch" in
  Linux-x86_64|Linux-amd64)    target=x86_64-unknown-linux-gnu ;;
  Linux-aarch64|Linux-arm64)   target=aarch64-unknown-linux-gnu ;;
  Darwin-x86_64)               target=x86_64-apple-darwin ;;
  Darwin-arm64)                target=aarch64-apple-darwin ;;
  *) echo "moe: no prebuilt binary for $os-$arch; build with: cargo install --git https://github.com/$repo" >&2; exit 1 ;;
esac

if [ "$version" = latest ]; then
  url="https://github.com/$repo/releases/latest/download/moe-$target.tar.gz"
else
  url="https://github.com/$repo/releases/download/$version/moe-$target.tar.gz"
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
echo "downloading $url"
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$url" -o "$tmp/moe.tar.gz"
else
  wget -qO "$tmp/moe.tar.gz" "$url"
fi
tar xzf "$tmp/moe.tar.gz" -C "$tmp"

mkdir -p "$bin_dir"
install -m 755 "$tmp/moe-$target/moe" "$bin_dir/moe"
echo "installed $("$bin_dir/moe" --version 2>/dev/null || echo moe) to $bin_dir/moe"

case ":$PATH:" in
  *":$bin_dir:"*) ;;
  *) echo; echo "add it to your PATH:"; echo "  export PATH=\"$bin_dir:\$PATH\"" ;;
esac
