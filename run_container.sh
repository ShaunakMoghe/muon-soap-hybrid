#!/bin/bash
# Run a command inside PACE's NGC PyTorch container (pytorch/25.10).
# PACE-specific fixes applied here:
# - TRITON_LIBCUDA_PATH: triton looks up libcuda via a stale ldconfig cache;
#   point it at apptainer's --nv bind dir instead (torch.compile breaks without this).
# - --cleanenv: scrub host env (XALT/COMPILER_PATH/LIBRARY_PATH/CPATH/LD_PRELOAD
#   from the module system poison the container's gcc); APPTAINERENV_* below
#   are re-applied after cleaning.
# - caches redirected to the project dir (home quota is only 20GB).
set -u
SIF=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-25.10.sif
PROJ=/storage/project/ps-moereasoning-0/smoghe6
export APPTAINERENV_TRITON_LIBCUDA_PATH=/.singularity.d/libs
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR=$PROJ/.cache/inductor
export APPTAINERENV_TRITON_CACHE_DIR=$PROJ/.cache/triton
export APPTAINERENV_HF_HOME=$PROJ/.cache/hf
export APPTAINERENV_HF_HUB_VERBOSITY=warning
export APPTAINERENV_PYTHONPATH=$PROJ # so training scripts can import research_stack/
export APPTAINER_CACHEDIR=$PROJ/.cache/apptainer
mkdir -p "$PROJ/.cache/inductor" "$PROJ/.cache/triton" "$PROJ/.cache/hf" "$PROJ/.cache/apptainer"
apptainer exec --cleanenv --nv "$SIF" "$@"
