#!/bin/sh
# post-comment.sh <name> <body-file>: post <body-file> as the PR's <name> comment, editing it on later runs.
set -eu
R=TRIQS/nda
marker="<!-- nda benchmarks: $1 -->"  # invisible on GitHub; tells our two comments apart

id=$(gh api "repos/$R/issues/$CHANGE_ID/comments" --paginate \
  --jq ".[] | select(.body | startswith(\"$marker\")) | .id" | tail -n 1)
if [ -n "$id" ]; then method=PATCH path=issues/comments/$id; else method=POST path=issues/$CHANGE_ID/comments; fi
{ echo "$marker"; cat "$2"; } | gh api -X "$method" "repos/$R/$path" -F body=@- > /dev/null
