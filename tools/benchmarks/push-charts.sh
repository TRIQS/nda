#!/bin/sh
# push-charts.sh <results-dir>: upload the benchmark-*.png charts to refs/benchmarks/pr-<CHANGE_ID>
# and point <results-dir>/charts.md at them.
set -eu
R=TRIQS/nda
cd "$1"

entry() {  # entry <file>: upload the file as a blob and print its tree entry
  sha=$(printf '{"encoding":"base64","content":"%s"}' "$(base64 -w0 "$1")" |
    gh api -X POST "repos/$R/git/blobs" --input - --jq .sha)
  printf '{"path":"%s","mode":"100644","type":"blob","sha":"%s"}' "$1" "$sha"
}
entries=
for png in benchmark-*.png; do
  entries="$entries${entries:+,}$(entry "$png")"
done
tree=$(printf '{"tree":[%s]}' "$entries" |
  gh api -X POST "repos/$R/git/trees" --input - --jq .sha)
head=$(printf '{"message":"benchmark charts PR #%s","tree":"%s","parents":[]}' "$CHANGE_ID" "$tree" |
  gh api -X POST "repos/$R/git/commits" --input - --jq .sha)
ref=benchmarks/pr-$CHANGE_ID
gh api -X PATCH "repos/$R/git/refs/$ref" -F sha="$head" -F force=true > /dev/null 2>&1 ||
  gh api -X POST "repos/$R/git/refs" -f ref="refs/$ref" -f sha="$head" > /dev/null
sed -i "s|](benchmark-|](https://raw.githubusercontent.com/$R/$head/benchmark-|" charts.md
