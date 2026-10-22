#!/bin/bash
# Recreate the lean fork: upstream llama.cpp at the pinned PR #28696 head + our six patches (branch v41-thor).
set -euo pipefail
cd "$(dirname "$0")"
PIN=2076412598e99127ea2554467b2b7530856895d3
[ -d llama.cpp/.git ] || git clone --filter=blob:none https://github.com/ggml-org/llama.cpp llama.cpp
cd llama.cpp
git cat-file -e $PIN^{commit} 2>/dev/null || git fetch origin pull/28696/head:pr-28696
if git rev-parse -q --verify v41-thor >/dev/null; then
  echo "branch v41-thor exists at $(git rev-parse --short v41-thor); delete it to re-apply"; exit 0
fi
git checkout -q -b v41-thor $PIN
git -c user.name="${GIT_AUTHOR_NAME:-s09}" -c user.email="${GIT_AUTHOR_EMAIL:-s09@localhost}" am -q ../patches/*.patch
git log --oneline -4
