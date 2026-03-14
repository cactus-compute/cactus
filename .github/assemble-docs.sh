#!/usr/bin/env bash
# Assembles site_docs/ from scattered source files and rewrites internal links.
# Used by the release.yml workflow.
set -euo pipefail

# Optional: pass a version tag (e.g., "v1.11") to inject a version note into quickstart.
# Usage: bash .github/assemble-docs.sh [version]
# If no version is passed (dev builds), no note is injected.
DOCS_VERSION="${1:-}"

# macOS sed requires '' for in-place, GNU sed does not
if sed --version >/dev/null 2>&1; then
  sedi() { sed -i "$@"; }
else
  sedi() { sed -i '' "$@"; }
fi

rm -rf site_docs
mkdir -p site_docs/docs site_docs/python site_docs/apple site_docs/android \
         site_docs/flutter site_docs/rust site_docs/blog site_docs/assets

# Copy assets
cp -r assets/* site_docs/assets/

# Copy custom stylesheets
mkdir -p site_docs/stylesheets
cp .github/docs-overrides/stylesheets/custom.css site_docs/stylesheets/custom.css

# Copy supporting files
cp CONTRIBUTING.md site_docs/CONTRIBUTING.md

# Copy docs (includes quickstart.md, choose-sdk.md, etc.)
cp docs/*.md site_docs/docs/

# Copy SDK READMEs
cp python/README.md site_docs/python/README.md
cp apple/README.md site_docs/apple/README.md
cp android/README.md site_docs/android/README.md
cp flutter/README.md site_docs/flutter/README.md

# Fetch React Native README from external repo
mkdir -p site_docs/react-native
if curl -sfL "https://raw.githubusercontent.com/cactus-compute/cactus-react-native/main/README.md" -o site_docs/react-native/README.md; then
  # Prepend version compatibility note
  {
    echo '!!! info "Independent release cycle"'
    echo '    The React Native SDK releases independently from the Cactus engine.'
    echo '    Check the [releases page](https://github.com/cactus-compute/cactus-react-native/releases) for the latest compatible version.'
    echo ''
    cat site_docs/react-native/README.md
  } > site_docs/react-native/README.tmp && mv site_docs/react-native/README.tmp site_docs/react-native/README.md
  # Fetch logo referenced in the README
  mkdir -p site_docs/react-native/assets
  curl -sfL "https://raw.githubusercontent.com/cactus-compute/cactus-react-native/main/assets/logo.png" -o site_docs/react-native/assets/logo.png 2>/dev/null || true
  echo "Fetched React Native README"
else
  echo "# React Native SDK" > site_docs/react-native/README.md
  echo "" >> site_docs/react-native/README.md
  echo "See [cactus-react on GitHub](https://github.com/cactus-compute/cactus-react) for full documentation." >> site_docs/react-native/README.md
  echo "Warning: Could not fetch React Native README, using fallback"
fi

# Rust may not exist in older versions
if [ -f rust/README.md ]; then
  cp rust/README.md site_docs/rust/README.md
fi

# Blog may not exist in older versions
if [ -d blog ] && ls blog/*.md >/dev/null 2>&1; then
  cp blog/*.md site_docs/blog/
fi

# Use README.md as index (single source of truth)
{
  echo '---'
  echo 'title: "Cactus"'
  echo 'description: "Energy-efficient AI inference engine for phones, wearables, Macs, and ARM devices."'
  echo '---'
  echo ''
  cat README.md
} > site_docs/index.md

# Remove the H1 title from index (banner replaces it)
sedi 's/^# Cactus$//' site_docs/index.md

# --- Fix links in index.md (README links -> site_docs structure) ---
sedi 's|(cactus_engine\.md)|(docs/cactus_engine.md)|g' site_docs/index.md
sedi 's|(cactus_graph\.md)|(docs/cactus_graph.md)|g' site_docs/index.md
sedi 's|(cactus_index\.md)|(docs/cactus_index.md)|g' site_docs/index.md
sedi 's|(finetuning\.md)|(docs/finetuning.md)|g' site_docs/index.md
sedi 's|(compatibility\.md)|(docs/compatibility.md)|g' site_docs/index.md
sedi 's|(/CONTRIBUTING\.md)|(CONTRIBUTING.md)|g' site_docs/index.md
sedi 's|(/python/)|(python/README.md)|g' site_docs/index.md
sedi 's|(/apple/)|(apple/README.md)|g' site_docs/index.md
sedi 's|(/android/)|(android/README.md)|g' site_docs/index.md
sedi 's|(/flutter/)|(flutter/README.md)|g' site_docs/index.md
sedi 's|(/rust/)|(rust/README.md)|g' site_docs/index.md
sedi 's|(/blog/hybrid_transcription\.md)|(blog/hybrid_transcription.md)|g' site_docs/index.md
sedi 's|(/blog/lfm2_24b_a2b\.md)|(blog/lfm2_24b_a2b.md)|g' site_docs/index.md
sedi 's|(quickstart\.md)|(docs/quickstart.md)|g' site_docs/index.md
sedi 's|(choose-sdk\.md)|(docs/choose-sdk.md)|g' site_docs/index.md

# --- Fix links in docs/ files (absolute /docs/ -> relative same dir, absolute /sdk/ -> ../sdk/) ---
for f in site_docs/docs/*.md; do
  sedi 's|(/docs/cactus_engine\.md)|(cactus_engine.md)|g' "$f"
  sedi 's|(/docs/cactus_graph\.md)|(cactus_graph.md)|g' "$f"
  sedi 's|(/docs/cactus_index\.md)|(cactus_index.md)|g' "$f"
  sedi 's|(/docs/finetuning\.md)|(finetuning.md)|g' "$f"
  sedi 's|(/docs/compatibility\.md)|(compatibility.md)|g' "$f"
  sedi 's|(/docs/quickstart\.md)|(quickstart.md)|g' "$f"
  sedi 's|(/docs/choose-sdk\.md)|(choose-sdk.md)|g' "$f"
  sedi 's|(/docs/index\.md)|(../index.md)|g' "$f"
  sedi 's|(/CONTRIBUTING\.md)|(../CONTRIBUTING.md)|g' "$f"
  sedi 's|(/python/)|(../python/README.md)|g' "$f"
  sedi 's|(/apple/)|(../apple/README.md)|g' "$f"
  sedi 's|(/android/)|(../android/README.md)|g' "$f"
  sedi 's|(/flutter/)|(../flutter/README.md)|g' "$f"
  sedi 's|(/rust/)|(../rust/README.md)|g' "$f"
done

# --- Fix links in SDK READMEs (one level deep -> ../docs/, ../sdk/) ---
for f in site_docs/python/README.md site_docs/apple/README.md site_docs/android/README.md site_docs/flutter/README.md; do
  sedi 's|(/docs/cactus_engine\.md)|(../docs/cactus_engine.md)|g' "$f"
  sedi 's|(/docs/cactus_graph\.md)|(../docs/cactus_graph.md)|g' "$f"
  sedi 's|(/docs/cactus_index\.md)|(../docs/cactus_index.md)|g' "$f"
  sedi 's|(/docs/finetuning\.md)|(../docs/finetuning.md)|g' "$f"
  sedi 's|(/docs/compatibility\.md)|(../docs/compatibility.md)|g' "$f"
  sedi 's|(/docs/quickstart\.md)|(../docs/quickstart.md)|g' "$f"
  sedi 's|(/docs/choose-sdk\.md)|(../docs/choose-sdk.md)|g' "$f"
  sedi 's|(/python/)|(../python/README.md)|g' "$f"
  sedi 's|(/apple/)|(../apple/README.md)|g' "$f"
  sedi 's|(/android/)|(../android/README.md)|g' "$f"
  sedi 's|(/flutter/)|(../flutter/README.md)|g' "$f"
  sedi 's|(/rust/)|(../rust/README.md)|g' "$f"
  sedi 's|(\.\.\/README\.md)|(../index.md)|g' "$f"
done

# Rust README (may not exist)
if [ -f site_docs/rust/README.md ]; then
  for pattern in \
    's|(/docs/cactus_engine\.md)|(../docs/cactus_engine.md)|g' \
    's|(/docs/cactus_graph\.md)|(../docs/cactus_graph.md)|g' \
    's|(/docs/cactus_index\.md)|(../docs/cactus_index.md)|g' \
    's|(/docs/finetuning\.md)|(../docs/finetuning.md)|g' \
    's|(/docs/compatibility\.md)|(../docs/compatibility.md)|g' \
    's|(/docs/quickstart\.md)|(../docs/quickstart.md)|g' \
    's|(/docs/choose-sdk\.md)|(../docs/choose-sdk.md)|g' \
    's|(/python/)|(../python/README.md)|g' \
    's|(/apple/)|(../apple/README.md)|g' \
    's|(/android/)|(../android/README.md)|g' \
    's|(/flutter/)|(../flutter/README.md)|g' \
    's|(/rust/)|(../rust/README.md)|g' \
    's|(\.\.\/README\.md)|(../index.md)|g'; do
    sedi "$pattern" site_docs/rust/README.md
  done
fi

# --- Fix links in blog posts (one level deep) ---
if ls site_docs/blog/*.md >/dev/null 2>&1; then
  for f in site_docs/blog/*.md; do
    sedi 's|(/docs/cactus_engine\.md)|(../docs/cactus_engine.md)|g' "$f"
    sedi 's|(/docs/cactus_graph\.md)|(../docs/cactus_graph.md)|g' "$f"
    sedi 's|(/docs/cactus_index\.md)|(../docs/cactus_index.md)|g' "$f"
    sedi 's|(/docs/finetuning\.md)|(../docs/finetuning.md)|g' "$f"
    sedi 's|(/docs/compatibility\.md)|(../docs/compatibility.md)|g' "$f"
    sedi 's|(/python/)|(../python/README.md)|g' "$f"
    sedi 's|(/apple/)|(../apple/README.md)|g' "$f"
    sedi 's|(/android/)|(../android/README.md)|g' "$f"
    sedi 's|(/flutter/)|(../flutter/README.md)|g' "$f"
    sedi 's|(/blog/hybrid_transcription\.md)|(hybrid_transcription.md)|g' "$f"
    sedi 's|(/blog/lfm2_24b_a2b\.md)|(lfm2_24b_a2b.md)|g' "$f"
  done
fi

# --- Fix links in CONTRIBUTING.md (root level) ---
sedi 's|(/docs/cactus_engine\.md)|(docs/cactus_engine.md)|g' site_docs/CONTRIBUTING.md
sedi 's|(/docs/cactus_graph\.md)|(docs/cactus_graph.md)|g' site_docs/CONTRIBUTING.md
sedi 's|(/docs/cactus_index\.md)|(docs/cactus_index.md)|g' site_docs/CONTRIBUTING.md
sedi 's|(/docs/index\.md)|(index.md)|g' site_docs/CONTRIBUTING.md

# --- Inject version note into quickstart for stable releases ---
if [ -n "$DOCS_VERSION" ]; then
  {
    echo "!!! note \"Version ${DOCS_VERSION}\""
    echo "    You're viewing docs for **${DOCS_VERSION}**. If you are cloning the repository, make sure to check out this release: \`git checkout ${DOCS_VERSION}\`"
    echo ""
    cat site_docs/docs/quickstart.md
  } > site_docs/docs/quickstart.tmp && mv site_docs/docs/quickstart.tmp site_docs/docs/quickstart.md
fi
