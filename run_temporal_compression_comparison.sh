#!/usr/bin/env bash
# This script has been superseded by two purpose-specific scripts:
#
#   run_tc_training_sweep.sh  — 8 training tasks (20h, wandb=True, all_people_one_expression)
#   run_tc_compile_test.sh    — 10 compile benchmark tasks (2.5h, no wandb, local results)
#
# To submit the training sweep:
#   sbatch run_tc_training_sweep.sh
#
# To submit the compile benchmark:
#   sbatch run_tc_compile_test.sh
#
# To run only a subset (e.g. tasks 1-3 of the training sweep):
#   sbatch --array=1-3 run_tc_training_sweep.sh
#
# To do a dry run of a specific training task locally:
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=2 bash run_tc_training_sweep.sh
echo "This script has been replaced. See the comment at the top of this file."
echo "  Training sweep : sbatch run_tc_training_sweep.sh"
echo "  Compile test   : sbatch run_tc_compile_test.sh"
