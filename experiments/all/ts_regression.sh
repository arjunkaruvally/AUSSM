#!/bin/bash

#####################
EXPERIMENT_ID="ts_regression"

## activate the waveRNN environment
conda activate ${CONDA_ENV}

# go to the experiments root dir
cd ${EXPERIMENT_SOURCE_ROOT} || exit

# the 2 variables below are usually set to 0 by default
# change them if the script needs to be rerun with more variations of inputs or after interruptions
START_GROUP_ID=0
((START_JOB_ID=START_GROUP_ID*5))

IN_SLURM_ARRAY=0;
# simulated SLURM environment variables
SLURM_ARRAY_TASK_ID=0;
SLURM_ARRAY_TASK_COUNT=10;
EXPERIMENT_ID="${EXPERIMENT_ID}_debug"

##################

JOB_ID=0  # This is the start of the jobid (identifies each job) mainly required when script is rerun
GROUP_ID=0   # this is the start of the group id (group jobs by seed for averages)

echo "Initializations Complete"

for SEED in {1..20}  # 5 - 2430 ## seed averaging in the outer loop - averages get better over time
do
  GROUP_ID=0
    for LEARNING_RATE in 0.00001 0.0001 0.001 # 3 - 162
    do
      for D_STATE in 16 64 128 # 3 - 54
      do
        for D_MODEL in 16 64 128 # 3 - 18
        do
          for WEIGHT_DECAY in 0.0 0.001  # 2 - 6
          do
            for LAYERS in "m|a" "a|m|m|a" "a|m|m|a|m|a"  # 3 - 3
            do
              echo "$SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT $JOB_ID / $LAST_JOB_ID"
              if [ $(( $JOB_ID % $SLURM_ARRAY_TASK_COUNT )) -eq $SLURM_ARRAY_TASK_ID  ]
              then
                if [ $JOB_ID -lt $START_JOB_ID ]
                then
                  echo "Skipped group: $GROUP_ID job: $JOB_ID"
                  ((JOB_ID=JOB_ID+1))
                  continue
                fi

                SAVE_DIR="${EXPERIMENT_OUTPUT_ROOT}/${EXPERIMENT_ID}_experiment/${JOB_ID}"

                # TODO: resume from checkpoint logic https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#save-a-checkpoint

                ## Have to find a better way to set the vocab_size
                python ts_regression_runner.py fit -c all/weather.yaml \
                                     --seed_everything $SEED \
                                     --data.root_path $DATA_ROOT \
                                     --data.batch_size 64 \
                                     --model.optimizer.weight_decay $WEIGHT_DECAY \
                                     --model.optimizer.lr $LEARNING_RATE --model.network.d_model $D_MODEL \
                                     --model.network.d_state $D_STATE \
                                     --model.network.layers $LAYERS \
                                     --trainer.logger.group experiment_$GROUP_ID \
                                     --trainer.default_root_dir $SAVE_DIR \
                                     --trainer.logger.save_dir $SAVE_DIR \
                                     --trainer.logger.project exp_$EXPERIMENT_ID \
                                     --trainer.deterministic true

                if [ $IN_SLURM_ARRAY -eq 0 ]  # if not in slurm array, exit after one execution
                then
                  echo "BASH: Exiting because the task is not run in slurm"
                  exit;
                fi
              fi
              ((JOB_ID=JOB_ID+1))
              ((GROUP_ID=GROUP_ID+1))
            done
          done
        done
      done
    done
  done
