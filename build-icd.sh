#!/usr/bin/env bash
# build-icd.sh - build the Honeykrisp (Apple GPU Vulkan) driver used for Strata as a PRIVATE per-process ICD.
#
# The desktop keeps the system Mesa.  A program opts in with
#   VK_DRIVER_FILES=$PREFIX/share/vulkan/icd.d/asahi_icd.aarch64.json
#
# Base: joshuaswarren/mesa-1, branch honeykrisp-omarchy-v3 (the omarchy-mlx maintainer's M1/M1 Max compute set:
# hardware cooperative matrices, bf16, 90% device heap, dependency-tracked CDM barrier, hoisted device loads), pinned
# below, plus this repo's patches/*.patch in order.
#
# Needs: meson ninja clang llvm libclc spirv-llvm-translator spirv-tools glslang python-mako python-packaging python-yaml
set -euo pipefail

BASE_REPO=${BASE_REPO:-https://github.com/joshuaswarren/mesa-1}
BASE_BRANCH=${BASE_BRANCH:-honeykrisp-omarchy-v3}
BASE_SHA=${BASE_SHA:-f144c7949b}
HERE=$(cd "$(dirname "$0")" && pwd)
# The Mesa checkout (cloned on first use; SRC=<path> builds an existing one)
SRC=${SRC:-$HERE/mesa}
NAME=${NAME:-v3}
PREFIX=${PREFIX:-$HOME/.local/lib/honeykrisp-strata/$NAME}

if [ ! -d "$SRC/.git" ]; then
    git clone --depth 50 -b "$BASE_BRANCH" "$BASE_REPO" "$SRC"
fi
cd "$SRC"
# The build resets the checkout's "strata" branch: never over uncommitted work.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "$SRC has uncommitted changes; commit or stash them first" >&2
    exit 1
fi
git fetch --depth 50 origin "$BASE_BRANCH"
git checkout -q -B strata "$BASE_SHA"
shopt -s nullglob
for p in "$HERE"/patches/*.patch; do
    echo "applying $(basename "$p")"
    git am -q --3way "$p"
done

BUILD=build-$NAME
if [ ! -f "$BUILD/build.ninja" ]; then
    meson setup "$BUILD" --prefix="$PREFIX" --buildtype=release -Db_ndebug=true \
        -Dvulkan-drivers=asahi -Dgallium-drivers= -Dplatforms= -Dvulkan-layers= -Dopengl=false \
        -Dgles1=disabled -Dgles2=disabled -Degl=disabled -Dglx=disabled -Dgbm=disabled -Dvideo-codecs= \
        -Dtools= -Dbuild-tests=false
fi
nice ninja -C "$BUILD" -j"$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))"
ninja -C "$BUILD" install >/dev/null
git rev-parse --short=10 HEAD > "$PREFIX/mesa-git-sha"
echo "ICD: $PREFIX/share/vulkan/icd.d/asahi_icd.aarch64.json ($(cat "$PREFIX/mesa-git-sha"))"
