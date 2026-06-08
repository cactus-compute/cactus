#!/usr/bin/env bash
# Assembles site_docs/ from scattered source files and rewrites internal links.
# Used by the release.yml workflow.
set -euo pipefail

DOCS_VERSION="${1:-}"

if sed --version >/dev/null 2>&1; then
  sedi() { sed -i "$@"; }
else
  sedi() { sed -i '' "$@"; }
fi

rm -rf site_docs
mkdir -p site_docs/docs site_docs/python site_docs/apple site_docs/android \
         site_docs/flutter site_docs/rust site_docs/swift site_docs/kotlin \
         site_docs/react-native site_docs/blog site_docs/assets

cp -r assets/* site_docs/assets/

echo "docs.cactuscompute.com" > site_docs/CNAME

mkdir -p site_docs/stylesheets site_docs/javascripts
cp .github/docs-overrides/stylesheets/custom.css site_docs/stylesheets/custom.css
cp .github/docs-overrides/javascripts/mathjax.js site_docs/javascripts/mathjax.js

[ -f CONTRIBUTING.md ] && cp CONTRIBUTING.md site_docs/CONTRIBUTING.md
[ -f DCO.md ] && cp DCO.md site_docs/DCO.md

cp docs/*.md site_docs/docs/

cp python/README.md site_docs/python/README.md
cp apple/README.md site_docs/apple/README.md
cp android/README.md site_docs/android/README.md
cp bindings/flutter/README.md site_docs/flutter/README.md
cp bindings/swift/README.md site_docs/swift/README.md
cp bindings/kotlin/README.md site_docs/kotlin/README.md
cp bindings/react-native/README.md site_docs/react-native/README.md
cp bindings/rust/README.md site_docs/rust/README.md

if [ -d blog ] && ls blog/*.md >/dev/null 2>&1; then
  cp blog/*.md site_docs/blog/
fi

{
  echo '---'
  echo 'title: "Cactus"'
  echo 'description: "Energy-efficient AI inference engine for phones, wearables, Macs, and ARM devices."'
  echo '---'
  echo ''
  cat README.md
} > site_docs/index.md

sedi 's/^# Cactus$//' site_docs/index.md

sedi 's|(/docs/cactus_engine\.md)|(docs/cactus_engine.md)|g' site_docs/index.md
sedi 's|(/docs/cactus_graph\.md)|(docs/cactus_graph.md)|g' site_docs/index.md
sedi 's|(/docs/cactus_index\.md)|(docs/cactus_index.md)|g' site_docs/index.md
sedi 's|(/docs/cactus_kernels\.md)|(docs/cactus_kernels.md)|g' site_docs/index.md
sedi 's|(/docs/cactus_quants\.md)|(docs/cactus_quants.md)|g' site_docs/index.md
sedi 's|(/docs/cactus_transpiler\.md)|(docs/cactus_transpiler.md)|g' site_docs/index.md
sedi 's|(/docs/cactus_hybrid\.md)|(docs/cactus_hybrid.md)|g' site_docs/index.md
sedi 's|(/docs/finetuning\.md)|(docs/finetuning.md)|g' site_docs/index.md
sedi 's|(/docs/compatibility\.md)|(docs/compatibility.md)|g' site_docs/index.md
sedi 's|(/docs/quickstart\.md)|(docs/quickstart.md)|g' site_docs/index.md
sedi 's|(/docs/choose-bindings\.md)|(docs/choose-bindings.md)|g' site_docs/index.md
sedi 's|(/CONTRIBUTING\.md)|(CONTRIBUTING.md)|g' site_docs/index.md
sedi 's|(/bindings/swift/)|(swift/README.md)|g' site_docs/index.md
sedi 's|(/bindings/kotlin/)|(kotlin/README.md)|g' site_docs/index.md
sedi 's|(/bindings/python/)|(python/README.md)|g' site_docs/index.md
sedi 's|(/bindings/react-native/)|(react-native/README.md)|g' site_docs/index.md
sedi 's|(/bindings/flutter/)|(flutter/README.md)|g' site_docs/index.md
sedi 's|(/bindings/rust/)|(rust/README.md)|g' site_docs/index.md
sedi 's|(/python/)|(python/README.md)|g' site_docs/index.md
sedi 's|(/apple/)|(apple/README.md)|g' site_docs/index.md
sedi 's|(/android/)|(android/README.md)|g' site_docs/index.md
sedi 's|(/blog/hybrid_transcription\.md)|(blog/hybrid_transcription.md)|g' site_docs/index.md
sedi 's|(/blog/lfm2_24b_a2b\.md)|(blog/lfm2_24b_a2b.md)|g' site_docs/index.md
sedi 's|(/blog/parakeet\.md)|(blog/parakeet.md)|g' site_docs/index.md
sedi 's|(/blog/lfm2\.5_350m\.md)|(blog/lfm2.5_350m.md)|g' site_docs/index.md
sedi 's|(/blog/gemma4\.md)|(blog/gemma4.md)|g' site_docs/index.md
sedi 's|(/blog/turboquant-h\.md)|(blog/turboquant-h.md)|g' site_docs/index.md

for f in site_docs/docs/*.md; do
  sedi 's|(/docs/cactus_engine\.md)|(cactus_engine.md)|g' "$f"
  sedi 's|(/docs/cactus_graph\.md)|(cactus_graph.md)|g' "$f"
  sedi 's|(/docs/cactus_index\.md)|(cactus_index.md)|g' "$f"
  sedi 's|(/docs/cactus_kernels\.md)|(cactus_kernels.md)|g' "$f"
  sedi 's|(/docs/cactus_quants\.md)|(cactus_quants.md)|g' "$f"
  sedi 's|(/docs/cactus_transpiler\.md)|(cactus_transpiler.md)|g' "$f"
  sedi 's|(/docs/cactus_hybrid\.md)|(cactus_hybrid.md)|g' "$f"
  sedi 's|(/docs/finetuning\.md)|(finetuning.md)|g' "$f"
  sedi 's|(/docs/compatibility\.md)|(compatibility.md)|g' "$f"
  sedi 's|(/docs/quickstart\.md)|(quickstart.md)|g' "$f"
  sedi 's|(/docs/choose-bindings\.md)|(choose-bindings.md)|g' "$f"
  sedi 's|(/docs/index\.md)|(../index.md)|g' "$f"
  sedi 's|(/blog/hybrid_transcription\.md)|(../blog/hybrid_transcription.md)|g' "$f"
  sedi 's|(/blog/lfm2_24b_a2b\.md)|(../blog/lfm2_24b_a2b.md)|g' "$f"
  sedi 's|(/blog/parakeet\.md)|(../blog/parakeet.md)|g' "$f"
  sedi 's|(/blog/lfm2\.5_350m\.md)|(../blog/lfm2.5_350m.md)|g' "$f"
  sedi 's|(/blog/gemma4\.md)|(../blog/gemma4.md)|g' "$f"
  sedi 's|(/blog/turboquant-h\.md)|(../blog/turboquant-h.md)|g' "$f"
  sedi 's|(/CONTRIBUTING\.md)|(../CONTRIBUTING.md)|g' "$f"
  sedi 's|(/bindings/swift/)|(../swift/README.md)|g' "$f"
  sedi 's|(/bindings/kotlin/)|(../kotlin/README.md)|g' "$f"
  sedi 's|(/bindings/python/)|(../python/README.md)|g' "$f"
  sedi 's|(/bindings/react-native/)|(../react-native/README.md)|g' "$f"
  sedi 's|(/bindings/flutter/)|(../flutter/README.md)|g' "$f"
  sedi 's|(/bindings/rust/)|(../rust/README.md)|g' "$f"
  sedi 's|(/python/)|(../python/README.md)|g' "$f"
  sedi 's|(/apple/)|(../apple/README.md)|g' "$f"
  sedi 's|(/android/)|(../android/README.md)|g' "$f"
done

for f in site_docs/python/README.md site_docs/apple/README.md site_docs/android/README.md site_docs/flutter/README.md site_docs/swift/README.md site_docs/kotlin/README.md site_docs/react-native/README.md site_docs/rust/README.md; do
  sedi 's|(/docs/cactus_engine\.md)|(../docs/cactus_engine.md)|g' "$f"
  sedi 's|(/docs/cactus_graph\.md)|(../docs/cactus_graph.md)|g' "$f"
  sedi 's|(/docs/cactus_index\.md)|(../docs/cactus_index.md)|g' "$f"
  sedi 's|(/docs/cactus_kernels\.md)|(../docs/cactus_kernels.md)|g' "$f"
  sedi 's|(/docs/cactus_quants\.md)|(../docs/cactus_quants.md)|g' "$f"
  sedi 's|(/docs/cactus_transpiler\.md)|(../docs/cactus_transpiler.md)|g' "$f"
  sedi 's|(/docs/cactus_hybrid\.md)|(../docs/cactus_hybrid.md)|g' "$f"
  sedi 's|(/docs/finetuning\.md)|(../docs/finetuning.md)|g' "$f"
  sedi 's|(/docs/compatibility\.md)|(../docs/compatibility.md)|g' "$f"
  sedi 's|(/docs/quickstart\.md)|(../docs/quickstart.md)|g' "$f"
  sedi 's|(/docs/choose-bindings\.md)|(../docs/choose-bindings.md)|g' "$f"
  sedi 's|(/docs/index\.md)|(../index.md)|g' "$f"
  sedi 's|(/blog/hybrid_transcription\.md)|(../blog/hybrid_transcription.md)|g' "$f"
  sedi 's|(/blog/lfm2_24b_a2b\.md)|(../blog/lfm2_24b_a2b.md)|g' "$f"
  sedi 's|(/blog/parakeet\.md)|(../blog/parakeet.md)|g' "$f"
  sedi 's|(/blog/lfm2\.5_350m\.md)|(../blog/lfm2.5_350m.md)|g' "$f"
  sedi 's|(/blog/gemma4\.md)|(../blog/gemma4.md)|g' "$f"
  sedi 's|(/blog/turboquant-h\.md)|(../blog/turboquant-h.md)|g' "$f"
  sedi 's|(/CONTRIBUTING\.md)|(../CONTRIBUTING.md)|g' "$f"
  sedi 's|(/bindings/swift/)|(../swift/README.md)|g' "$f"
  sedi 's|(/bindings/kotlin/)|(../kotlin/README.md)|g' "$f"
  sedi 's|(/bindings/python/)|(../python/README.md)|g' "$f"
  sedi 's|(/bindings/react-native/)|(../react-native/README.md)|g' "$f"
  sedi 's|(/bindings/flutter/)|(../flutter/README.md)|g' "$f"
  sedi 's|(/bindings/rust/)|(../rust/README.md)|g' "$f"
  sedi 's|(/python/)|(../python/README.md)|g' "$f"
  sedi 's|(/apple/)|(../apple/README.md)|g' "$f"
  sedi 's|(/android/)|(../android/README.md)|g' "$f"
  sedi 's|(\.\.\/README\.md)|(../index.md)|g' "$f"
done

if ls site_docs/blog/*.md >/dev/null 2>&1; then
  for f in site_docs/blog/*.md; do
    sedi 's|(/docs/cactus_engine\.md)|(../docs/cactus_engine.md)|g' "$f"
    sedi 's|(/docs/cactus_graph\.md)|(../docs/cactus_graph.md)|g' "$f"
    sedi 's|(/docs/cactus_index\.md)|(../docs/cactus_index.md)|g' "$f"
    sedi 's|(/docs/cactus_kernels\.md)|(../docs/cactus_kernels.md)|g' "$f"
    sedi 's|(/docs/cactus_quants\.md)|(../docs/cactus_quants.md)|g' "$f"
    sedi 's|(/docs/cactus_transpiler\.md)|(../docs/cactus_transpiler.md)|g' "$f"
    sedi 's|(/docs/cactus_hybrid\.md)|(../docs/cactus_hybrid.md)|g' "$f"
    sedi 's|(/docs/finetuning\.md)|(../docs/finetuning.md)|g' "$f"
    sedi 's|(/docs/compatibility\.md)|(../docs/compatibility.md)|g' "$f"
    sedi 's|(/docs/quickstart\.md)|(../docs/quickstart.md)|g' "$f"
    sedi 's|(/docs/choose-bindings\.md)|(../docs/choose-bindings.md)|g' "$f"
    sedi 's|(/docs/index\.md)|(../index.md)|g' "$f"
    sedi 's|(/CONTRIBUTING\.md)|(../CONTRIBUTING.md)|g' "$f"
    sedi 's|(/bindings/swift/)|(../swift/README.md)|g' "$f"
    sedi 's|(/bindings/kotlin/)|(../kotlin/README.md)|g' "$f"
    sedi 's|(/bindings/python/)|(../python/README.md)|g' "$f"
    sedi 's|(/bindings/react-native/)|(../react-native/README.md)|g' "$f"
    sedi 's|(/bindings/flutter/)|(../flutter/README.md)|g' "$f"
    sedi 's|(/bindings/rust/)|(../rust/README.md)|g' "$f"
    sedi 's|(/python/)|(../python/README.md)|g' "$f"
    sedi 's|(/apple/)|(../apple/README.md)|g' "$f"
    sedi 's|(/android/)|(../android/README.md)|g' "$f"
    sedi 's|(/blog/hybrid_transcription\.md)|(hybrid_transcription.md)|g' "$f"
    sedi 's|(/blog/lfm2_24b_a2b\.md)|(lfm2_24b_a2b.md)|g' "$f"
    sedi 's|(/blog/parakeet\.md)|(parakeet.md)|g' "$f"
    sedi 's|(/blog/lfm2\.5_350m\.md)|(lfm2.5_350m.md)|g' "$f"
    sedi 's|(/blog/gemma4\.md)|(gemma4.md)|g' "$f"
    sedi 's|(/blog/turboquant-h\.md)|(turboquant-h.md)|g' "$f"
  done
fi

if [ -f site_docs/CONTRIBUTING.md ]; then
  sedi 's|(/docs/cactus_engine\.md)|(docs/cactus_engine.md)|g' site_docs/CONTRIBUTING.md
  sedi 's|(/docs/cactus_graph\.md)|(docs/cactus_graph.md)|g' site_docs/CONTRIBUTING.md
  sedi 's|(/docs/cactus_index\.md)|(docs/cactus_index.md)|g' site_docs/CONTRIBUTING.md
  sedi 's|(/docs/cactus_kernels\.md)|(docs/cactus_kernels.md)|g' site_docs/CONTRIBUTING.md
  sedi 's|(/docs/cactus_quants\.md)|(docs/cactus_quants.md)|g' site_docs/CONTRIBUTING.md
  sedi 's|(/docs/cactus_transpiler\.md)|(docs/cactus_transpiler.md)|g' site_docs/CONTRIBUTING.md
  sedi 's|(/docs/cactus_hybrid\.md)|(docs/cactus_hybrid.md)|g' site_docs/CONTRIBUTING.md
  sedi 's|(/docs/index\.md)|(index.md)|g' site_docs/CONTRIBUTING.md
fi

if [ -n "$DOCS_VERSION" ]; then
  {
    echo "!!! note \"Version ${DOCS_VERSION}\""
    echo "    You're viewing docs for **${DOCS_VERSION}**. If you are cloning the repository, make sure to check out this release: \`git checkout ${DOCS_VERSION}\`"
    echo ""
    cat site_docs/docs/quickstart.md
  } > site_docs/docs/quickstart.tmp && mv site_docs/docs/quickstart.tmp site_docs/docs/quickstart.md
fi

for nav_path in \
  "python/README.md" \
  "swift/README.md" \
  "kotlin/README.md" \
  "flutter/README.md" \
  "react-native/README.md" \
  "rust/README.md" \
  "blog/README.md" \
  "blog/hybrid_transcription.md" \
  "blog/lfm2_24b_a2b.md" \
  "blog/parakeet.md" \
  "blog/lfm2.5_350m.md" \
  "blog/gemma4.md" \
  "blog/turboquant-h.md" \
  "CONTRIBUTING.md" \
  "docs/compatibility.md"; do
  if [ ! -f "site_docs/$nav_path" ]; then
    grep -vF "$nav_path" mkdocs.yml > mkdocs.yml.tmp && mv mkdocs.yml.tmp mkdocs.yml
  fi
done

if ! ls site_docs/blog/*.md >/dev/null 2>&1; then
  grep -v "^  - Blog:" mkdocs.yml > mkdocs.yml.tmp && mv mkdocs.yml.tmp mkdocs.yml
fi
