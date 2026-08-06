#!/usr/bin/env bash
#
# Bundle the built Cactus coding agent into the Python package so it ships in the
# pypi wheel and `cactus code` works after `pip install cactus-compute`.
#
# Produces a self-contained payload at python/cactus/code/ with the four built
# packages plus a production-only node_modules (workspace packages copied in via
# --install-links, so it's portable — no symlinks back into the repo).
#
# Requires: node >=22, npm. Run from anywhere.
set -euo pipefail

CC_ROOT="$(cd "$(dirname "$0")/.." && pwd)"        # cactus-code/
OUT="$(cd "$CC_ROOT/.." && pwd)/python/cactus/code" # python/cactus/code/

echo "==> Building cactus-code"
cd "$CC_ROOT"
npm run build

echo "==> Staging payload at $OUT"
rm -rf "$OUT"
mkdir -p "$OUT/packages"
for p in ai agent tui coding-agent; do
	mkdir -p "$OUT/packages/$p"
	cp -R "packages/$p/dist" "$OUT/packages/$p/dist"
	cp "packages/$p/package.json" "$OUT/packages/$p/package.json"
done
# tui ships prebuilt native bindings (not part of dist); copy them so the TUI
# loads its key-modifier addon after a pip install.
if [ -d "packages/tui/native" ]; then
	mkdir -p "$OUT/packages/tui/native"
	cp -R "packages/tui/native" "$OUT/packages/tui/"
fi
cp package.json "$OUT/package.json"
[ -f package-lock.json ] && cp package-lock.json "$OUT/package-lock.json" || true
# Ship the Cactus version so the agent can display it after a pip install.
[ -f "$CC_ROOT/../CACTUS_VERSION" ] && cp "$CC_ROOT/../CACTUS_VERSION" "$OUT/CACTUS_VERSION" || true

echo "==> Installing production dependencies"
cd "$OUT"
# --install-links copies workspace packages as real dirs; --omit=dev drops
# typescript/tsgo/etc. Optional deps (the native clipboard module used by /copy
# and image paste) are kept so those work; the wheel is platform-specific anyway
# since the Cactus engine ships native libraries per platform.
# --ignore-scripts: the bundled root package.json carries dev lifecycle hooks
# (e.g. `prepare: husky`) that must not run when installing the wheel payload.
npm install --omit=dev --install-links --no-audit --no-fund --ignore-scripts

echo "==> Making payload wheel-safe (dereference symlinks, drop .bin)"
# Workspace packages get symlinked by npm; wheels don't carry symlinks reliably,
# so replace them with real directory copies.
if [ -d "$OUT/node_modules/@earendil-works" ]; then
	for d in "$OUT/node_modules/@earendil-works"/*; do
		if [ -L "$d" ]; then
			real="$(cd "$d" && pwd -P)"
			rm "$d"
			cp -R "$real" "$d"
		fi
	done
fi
# .bin holds only symlinks to dev tools; not needed at runtime.
rm -rf "$OUT/node_modules/.bin"
# Drop any other stray symlinks so the wheel is fully self-contained.
find "$OUT" -type l -delete 2>/dev/null || true

echo "==> Done. Bundled agent at: $OUT"
echo "    entry: $OUT/packages/coding-agent/dist/cli.js"
