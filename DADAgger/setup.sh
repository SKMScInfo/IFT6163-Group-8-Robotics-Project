export DMC_VIS_BENCH_DIR=~/IFT6163-Group-8-Robotics-Project/dmc_vision_benchmark
export DATA_DIR=/tmp/dmc_vision_bench_data
export MUJOCO_GL=osmesa

mkdir -p $DATA_DIR/dmc_vision_benchmark/dmc_vision_benchmark
gcloud storage cp -r gs://dmc_vision_benchmark/{LICENSE.txt,README.md,kubric_movi-d} $DATA_DIR/dmc_vision_benchmark

export ANTMAZE_DATA=antmaze_random_hidden_goal  # Options: antmaze_random_visible_goal, antmaze_random_hidden_goal
export BEHAVIORAL_POLICY=expert  # Options: expert, medium

export ANTMAZE_TASK=easy7x7a  # Options: empty7x7, easy7x7a, easy7x7b, easy7x7c, medium7x7a, medium7x7b, medium7x7c
mkdir -p $DATA_DIR/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK
gcloud storage cp -r gs://dmc_vision_benchmark/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK/$BEHAVIORAL_POLICY $DATA_DIR/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK

# export ANTMAZE_TASK=easy7x7a  # Options: empty7x7, easy7x7a, easy7x7b, easy7x7c, medium7x7a, medium7x7b, medium7x7c
# mkdir -p $DATA_DIR/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK
# gcloud storage cp -r gs://dmc_vision_benchmark/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK/$BEHAVIORAL_POLICY $DATA_DIR/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK

# export ANTMAZE_TASK=easy7x7b  # Options: empty7x7, easy7x7a, easy7x7b, easy7x7c, medium7x7a, medium7x7b, medium7x7c
# mkdir -p $DATA_DIR/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK
# gcloud storage cp -r gs://dmc_vision_benchmark/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK/$BEHAVIORAL_POLICY $DATA_DIR/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK

# export ANTMAZE_TASK=easy7x7c  # Options: empty7x7, easy7x7a, easy7x7b, easy7x7c, medium7x7a, medium7x7b, medium7x7c
# mkdir -p $DATA_DIR/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK
# gcloud storage cp -r gs://dmc_vision_benchmark/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK/$BEHAVIORAL_POLICY $DATA_DIR/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK

# export ANTMAZE_TASK=medium7x7a  # Options: empty7x7, easy7x7a, easy7x7b, easy7x7c, medium7x7a, medium7x7b, medium7x7c
# mkdir -p $DATA_DIR/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK
# gcloud storage cp -r gs://dmc_vision_benchmark/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK/$BEHAVIORAL_POLICY $DATA_DIR/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK

# export ANTMAZE_TASK=medium7x7b  # Options: empty7x7, easy7x7a, easy7x7b, easy7x7c, medium7x7a, medium7x7b, medium7x7c
# mkdir -p $DATA_DIR/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK
# gcloud storage cp -r gs://dmc_vision_benchmark/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK/$BEHAVIORAL_POLICY $DATA_DIR/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK

# export ANTMAZE_TASK=medium7x7c  # Options: empty7x7, easy7x7a, easy7x7b, easy7x7c, medium7x7a, medium7x7b, medium7x7c
# mkdir -p $DATA_DIR/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK
# gcloud storage cp -r gs://dmc_vision_benchmark/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK/$BEHAVIORAL_POLICY $DATA_DIR/dmc_vision_benchmark/$ANTMAZE_DATA/$ANTMAZE_TASK




# export ANTMAZE_DATA=antmaze_random_visible_goal  # Options: antmaze_random_visible_goal, antmaze_random_hidden_goal, antmaze_fixed_hidden_goal
# gcloud storage cp -r gs://dmc_vision_benchmark/dmc_vision_benchmark/$ANTMAZE_DATA $DATA_DIR/dmc_vision_benchmark/dmc_vision_benchmark

# gcloud storage cp -r gs://dmc_vision_benchmark/dmc_vision_benchmark/locomotion/ $DATA_DIR/dmc_vision_benchmark/dmc_vision_benchmark
