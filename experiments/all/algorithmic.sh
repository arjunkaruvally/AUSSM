#####################
EXPERIMENT_ID="algorithmic"

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

for D_STATE in 8 16 32 # 1
do
  for D_MODEL in 8 16 32 64 # 4 - 24
  do
    for WEIGHT_DECAY in 0.01 0.001 0.0 # 3 -6
    do
    for LEARNING_RATE in 0.01 0.001 0.0001 # 1
    do
      for LAYERS in "a|m" "m|a" "a|a" "m|m"
      do
        for TASK_NAME in bucketsort repetition modarithmeticwobraces cyclenav modarithmetic solveequation parity majoritycount majority ;
        do
          for SEED in 1 2 3 4 5  # 2
          do
            echo "$SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT $JOB_ID"
            if [ $(( $JOB_ID % $SLURM_ARRAY_TASK_COUNT )) -eq $SLURM_ARRAY_TASK_ID  ]
            then
              if [ $JOB_ID -lt $START_JOB_ID ]
              then
                echo "Skipped group: $GROUP_ID job: $JOB_ID"
                ((JOB_ID=JOB_ID+1))
                continue
              fi

              T_0=1000
              BATCH_SIZE=256

              SAVE_DIR="${EXPERIMENT_OUTPUT_ROOT}/${EXPERIMENT_ID}_experiment/${JOB_ID}"

              # TODO: resume from checkpoint logic https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#save-a-checkpoint

               ## Have to find a better way to set the vocab_size
              python fl_seq2seq_runner.py fit -c all/fl_base.yaml --seed_everything $SEED \
                                   --data.batch_size $BATCH_SIZE \
                                   --data.task_name $TASK_NAME \
                                   --model.optimizer.weight_decay $WEIGHT_DECAY \
                                   --model.optimizer.lr $LEARNING_RATE --model.network.d_model $D_MODEL \
                                   --model.network.d_state $D_STATE \
                                   --model.network.layers $LAYERS \
                                   --model.scheduler.T_0 $T_0 \
                                   --trainer.logger.group experiment_$GROUP_ID \
                                   --trainer.default_root_dir $SAVE_DIR \
                                   --trainer.logger.save_dir $SAVE_DIR \
                                   --trainer.logger.project exp_$EXPERIMENT_ID \
                                   --trainer.deterministic true
            fi
            ((JOB_ID=JOB_ID+1))
          done
          ((GROUP_ID=GROUP_ID+1))
        done
      done
    done
  done
done

