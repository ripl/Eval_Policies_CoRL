#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --exclude=g0,g7
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=120G
#SBATCH -t 08:00:00
#SBATCH --no-requeue
#SBATCH -J se_abcde_dex
#SBATCH -o /home-nfs/tianchong/simplerenv_protocol_abcde_full_%x_%j.out
#SBATCH -e /home-nfs/tianchong/simplerenv_protocol_abcde_full_%x_%j.err

set -Eeuo pipefail
SUPPORT_ROOT="${SUPPORT_ROOT:?SUPPORT_ROOT required}"
exec bash "${SUPPORT_ROOT}/run_dexbotic_condition.sh"
