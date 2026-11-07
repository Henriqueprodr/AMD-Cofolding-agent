#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Serve the chat model with vLLM on a LUMI-G node. session.sbatch starts it.
#
#   VLLM_MODEL         gemma (google/gemma-4-31B-it, default), gptoss
#                      (openai/gpt-oss-120b) or qwen (Qwen/Qwen3.8-27B)
#   VLLM_GPUS          GCDs to use (default 0,1,2,3; 4-7 are left for REINVENT and Boltz)
#   VLLM_PORT          OpenAI-compatible endpoint on the node (default 8001; dd-web uses 8000)
#   VLLM_HOST          address to bind (default 127.0.0.1). The supervisor tier has
#                      to be 0.0.0.0: its clients are jobs on other nodes.
#   VLLM_SIF           vLLM image (default services/lumi/vllm.sif, made by build.sbatch)
#   HF_HOME            model weights (default data/hf-cache, filled by build.sbatch)
#   VLLM_LONG_CONTEXT  Qwen only: 1 (default) for 1M context through YaRN, 0 for the native 262,144
#   VLLM_SPEC_TOKENS   Qwen only: tokens drafted per step by its MTP head (default 2, 0 = off)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DATA_DIR=${DATA_DIR:-$REPO/data}
SIF=${VLLM_SIF:-$HERE/vllm.sif}
WHICH=${VLLM_MODEL:-gemma}
GPUS=${VLLM_GPUS:-0,1,2,3}
PORT=${VLLM_PORT:-8001}
TP=$(awk -F, '{print NF}' <<<"$GPUS")
SPEC=${VLLM_SPEC_TOKENS:-2}

case "$WHICH" in
qwen) MODEL=Qwen/Qwen3.8-27B ;;
gemma) MODEL=google/gemma-4-31B-it ;;
gptoss) MODEL=openai/gpt-oss-120b ;;
*)
    echo "VLLM_MODEL must be gemma, gptoss or qwen, got '$WHICH'" >&2
    exit 2
    ;;
esac

# Decoding and memory are the same for both models so answers feel the same either way.
ARGS=(
    --host "${VLLM_HOST:-127.0.0.1}" --port "$PORT"
    --served-model-name "$MODEL"
    --tensor-parallel-size "$TP"
    --dtype bfloat16
    --gpu-memory-utilization 0.90
    --max-num-seqs 8
    --enable-prefix-caching
    --enable-auto-tool-choice
    --limit-mm-per-prompt '{"image": 0, "video": 0}'
)

if [[ "$WHICH" == gptoss ]]; then
    # gpt-oss ships MXFP4 weights and gfx90a has no MXFP4 silicon, so the MoE
    # has to be dequantized per matmul. The weights stay 4-bit in VRAM either
    # way, which is the whole reason the model fits two GCDs instead of the
    # eight a BF16 upcast would need.
    #
    # The backend is named explicitly because ROCm's automatic selection does
    # not offer triton_unfused for gpt-oss -- it is commented out of the
    # priority list over an MTP-drafting bug that does not apply here, since
    # this checkpoint has no MTP head -- and would otherwise fall through to
    # the slower emulation path.
    #
    # No --tool-call-parser or --reasoning-parser: gpt-oss speaks harmony, which
    # vLLM parses natively, and naming a parser here overrides that with one
    # that cannot read it. Verified on gfx90a: the reply carries a populated
    # `reasoning` field with neither flag set.
    ARGS+=(
        --moe-backend "${VLLM_MOE_BACKEND:-triton_unfused}"
        --max-model-len "${VLLM_MAX_MODEL_LEN:-131072}"
    )
    # AITER's kernels are built for gfx942 and newer; on gfx90a enabling it
    # only produces dispatch errors.
    export VLLM_ROCM_USE_AITER=0
elif [[ "$WHICH" == qwen ]]; then
    ARGS+=(--tool-call-parser qwen3_coder --reasoning-parser qwen3)
    if [[ "${VLLM_LONG_CONTEXT:-1}" == 1 ]]; then
        export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
        ARGS+=(
            --max-model-len 1000000
            --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
        )
    else
        ARGS+=(--max-model-len 262144)
    fi
    (( SPEC > 0 )) && ARGS+=(--speculative-config "{\"method\": \"qwen3_5_mtp\", \"num_speculative_tokens\": $SPEC}")
else
    # 262,144 is Gemma 4's native context; the -it checkpoint ships no MTP head, so no drafting.
    ARGS+=(--tool-call-parser gemma4 --reasoning-parser gemma4 --max-model-len 262144)
fi

# Per-port, not shared: two tiers start at the same moment, and both the AITER
# copy below and Triton's compile cache are written by whichever gets there
# first. One scratch each costs a second JIT compile and removes the race.
TMP=$DATA_DIR/tmp/vllm-$PORT
# Gemma's RMS norm goes through AITER, whose JIT copies itself into $HOME on first use;
# the tensor-parallel workers race that copy and the engine dies. Seed one copy first.
export AITER_JIT_DIR=$TMP/aiter-jit
if [[ ! -f $AITER_JIT_DIR/core.py ]]; then
    mkdir -p "$AITER_JIT_DIR"
    # A few root-owned cache blobs in the image are unreadable; AITER rebuilds those.
    singularity exec -B /pfs,/scratch "$SIF" \
        bash -c "cp -r /usr/local/lib/python3.12/dist-packages/aiter/jit/. '$AITER_JIT_DIR/' 2>/dev/null; true"
fi

export HF_HOME=${HF_HOME:-$DATA_DIR/hf-cache}
export HF_HUB_OFFLINE=1
export HIP_VISIBLE_DEVICES=$GPUS
export VLLM_CACHE_ROOT=$TMP/vllm-cache
export TRITON_CACHE_DIR=$TMP/triton-cache
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

# VLLM_PORT is ours above but also vLLM's own: it is the base it derives the
# tensor-parallel workers' rendezvous port from, so two tiers started from
# here would both reach for the same one and the second engine dies with
# EADDRINUSE. Dropped so vLLM picks a free port per instance; the rest go with
# it only to keep "Unknown vLLM environment variable" out of the log.
unset VLLM_PORT VLLM_MODEL VLLM_GPUS VLLM_HOST VLLM_SIF \
      VLLM_MAX_MODEL_LEN VLLM_MOE_BACKEND VLLM_SPEC_TOKENS VLLM_LONG_CONTEXT

exec singularity exec -B /pfs,/scratch,/project,/appl "$SIF" \
    vllm serve "$MODEL" "${ARGS[@]}"
