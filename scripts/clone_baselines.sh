#!/bin/sh
# Clone CPU-CAST baseline / tool repositories (shallow) into baselines/.
# Tool libraries installed via pip are recorded in manifests/python_env.txt instead.
set -e
cd "$(dirname "$0")/.."
mkdir -p baselines
cd baselines

# Algorithm baselines (must-reproduce)
for repo in \
  ibrahimjamadi98/GAINER \
  alexlimh/CEC-Ranking \
  castorini/earlyexiting-monobert \
  veneres/SEE-SIGIR25 \
  xpmir/mice \
  lightonai/fast-plaid
do
  dir=$(basename "$repo")
  if [ -d "$dir/.git" ]; then
    echo "cached: $dir"
  else
    echo "cloning $repo"
    git clone --depth 1 "https://github.com/$repo.git" "$dir"
  fi
done

# Tool repo needed for BM25 candidate runs (requires Java at runtime; not installed in this sandbox)
if [ ! -d pyserini/.git ]; then
  git clone --depth 1 https://github.com/castorini/pyserini.git pyserini
fi

# Record pinned commit for reproducibility
: > ../manifests/baselines_manifest.txt
for d in */; do
  sha=$(git -C "$d" rev-parse HEAD)
  url=$(git -C "$d" remote get-url origin)
  printf "%s\t%s\t%s\n" "${d%/}" "$url" "$sha" >> ../manifests/baselines_manifest.txt
done
cat ../manifests/baselines_manifest.txt
