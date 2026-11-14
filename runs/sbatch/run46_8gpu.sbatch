#!/bin/bash
#SBATCH -J run46_8gpu
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:8
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/run46_8gpu_%j.out
#SBATCH --exclude=atl1-1-03-020-6-0
# Run 46 8-GPU: COSMOS r32 QR/4 seed1, 3250 steps, saving OFF (ckpt asserts world_size 1).
# Node atl1-1-03-020-6-0 excluded at submit: bad GPU slot (RUNS.md incidents).
PROJ=/storage/project/ps-moereasoning-0/smoghe6
# GPU preflight. On 2026-11-11 one bad GPU slot failed 14 queued jobs in 3.5
# minutes, each fast failure handing the node to the next job. If any of the
# 8 GPUs cannot create a CUDA context (exit 42): exclude this node from my
# pending GPU jobs, resubmit this run once, and exit before the run dir is
# touched. Any other preflight failure falls through to the run unchanged.
$PROJ/run_container.sh python -c '
import sys, torch
try:
    n = torch.cuda.device_count()
    if n != 8:
        raise RuntimeError(f"device_count={n}")
    for i in range(n):
        torch.zeros(8, device=f"cuda:{i}").sum().item()
except Exception as e:
    print("GPU_PREFLIGHT_FAIL", repr(e))
    sys.exit(42)
print("GPU_PREFLIGHT_OK", n)
'
if [ $? -eq 42 ]; then
    NODE=${SLURMD_NODENAME:-$(hostname -s)}
    echo "GPU_PREFLIGHT_FAIL node=$NODE job=$SLURM_JOB_ID"
    nvidia-smi --query-gpu=index,uuid,compute_mode,memory.used --format=csv
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
    for J in $(squeue -u "$(id -un)" -h -t PD -p gpu-rtxpro-blackwell -o %i); do
        CUR=$(squeue -j "$J" -h -o %x)
        case ",$CUR," in *",$NODE,"*) continue ;; esac
        case "$CUR" in *,*) echo "job $J already excludes 2 nodes; left alone"; continue ;; esac
        scontrol update JobId="$J" ExcNodeList="${CUR:+$CUR,}$NODE" && echo "excluded $NODE for job $J"
    done
    OWN=$(squeue -j "$SLURM_JOB_ID" -h -o %x)
    case "$OWN" in
        *,*) echo "this job already excluded 2 nodes; not resubmitting" ;;
        *) [ "${GUARD_ATTEMPT:-1}" -lt 2 ] && sbatch -A "$SLURM_JOB_ACCOUNT" --exclude="${OWN:+$OWN,}$NODE" --export=ALL,GUARD_ATTEMPT=2 "$PROJ/runs/sbatch/run46_8gpu.sbatch" ;;
    esac
    exit 1
fi
cd $PROJ/runs/run46_cosmos_qrk4_s1 || exit 1
APPTAINERENV_CKPT_EVERY=1000000 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=8 $PROJ/variants/train_gpt_cosmos_r32_qrk4_seed1.py 1 > stdout.log 2>&1 || { echo RUN46_FAIL; tail -50 stdout.log; exit 1; }
echo RUN46_8GPU_DONE
