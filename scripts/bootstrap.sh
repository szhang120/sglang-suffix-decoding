#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -d sglang/.git ]]; then
  git clone --branch v0.5.21 --depth 1 https://github.com/sgl-project/sglang.git sglang
  git -C sglang switch -c codex/suffix-decoding
fi
base=e00930c5489053f26d86b179cee0d087f846acbb
if [[ "$(git -C sglang rev-parse HEAD)" != "$base" ]]; then
  echo 'Existing SGLang HEAD differs from source lock; preserving checkout.' >&2
  exit 1
fi
if git -C sglang apply --reverse --check ../patches/sglang-suffix.patch 2>/dev/null; then
  echo 'Suffix patch is already present.'
elif [[ -z "$(git -C sglang status --porcelain)" ]]; then
  git -C sglang apply --check ../patches/sglang-suffix.patch
  git -C sglang apply ../patches/sglang-suffix.patch
else
  echo 'Existing SGLang changes found; preserving checkout. Inspect patch manually.' >&2
  exit 1
fi
if [[ ! -d reference/ArcticInference/.git ]]; then
  git clone https://github.com/snowflakedb/ArcticInference.git reference/ArcticInference
  git -C reference/ArcticInference checkout --detach aca5d9a8a62474035c15d114d40a01abc8c94b51
fi
if [[ "$(git -C reference/ArcticInference rev-parse HEAD)" != aca5d9a8a62474035c15d114d40a01abc8c94b51 ]]; then
  echo 'Author checkout differs from source lock; preserving it.' >&2
  exit 1
fi
