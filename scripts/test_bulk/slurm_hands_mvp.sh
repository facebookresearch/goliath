#!/bin/bash

###-----------------------------------------------------------------
### Configuration variables
QOS='urgent_deadline'

###-----------------------------------------------------------------
CONFIG_FILE=config/hand_mvp_example.yml

CONDA_ENV="/uca/conda-envs/dgxenv-2024-08-16-07-36-10-x7977-centos9-py310-pt231/bin/activate"
TIME='1-00:00:00'


###-----------------------------------------------------------------
ROOT_DIR=/home/julietamartinez/rsc/goliath/

declare -a SIDS=(
    "AXE977"
    "XKT970"
    "QVC422"
    "QZX685"
)
declare -a DATA_ROOTS=(
    "/uca/julieta/oss/goliath/m--20230306--0839--AXE977--pilot--ProjectGoliath--Hands--"
    "/uca/julieta/oss/goliath/m--20230524--1054--XKT970--pilot--ProjectGoliath--Hands--"
    "/uca/julieta/oss/goliath/m--20230714--1019--QVC422--pilot--ProjectGoliath--Hands--"
    "/uca/julieta/oss/goliath/m--20230317--1130--QZX685--pilot--ProjectGoliath--Hands--"
)
declare -a HAND_SIDES=("left" "right")

for (( i=0; i<"${#SIDS[@]}"; i++ )); do
for (( j=0; j<"${#HAND_SIDES[@]}"; j++ )); do

###-----------------------------------------------------------------
# Assign variables from loop
SID="${SIDS[i]}"
HAND_SIDE="${HAND_SIDES[j]}"

echo "meow $i"
echo $HAND_SIDE

DATA_ROOT="${DATA_ROOTS[i]}"${HAND_SIDE}

SHARED_ASSETS=/uca/julieta/oss/goliath/shared/static_assets_hand_${HAND_SIDE}.pt

###-----------------------------------------------------------------
JOB_NAME=TEST_HAND_MVP_${SID}_${HAND_SIDE}
CKPT_DIR=/checkpoint/avatar/julietamartinez/goliath/hand_mvp/${SID}/${HAND_SIDE}

###-----------------------------------------------------------------
# Create a temporary script file
SCRIPT=$(mktemp)

# Write the SLURM script to the temporary file
cat > $SCRIPT <<EOL
#!/bin/bash
#SBATCH --partition=learn
#SBATCH --array=1-4
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=220G
#SBATCH --output=${CKPT_DIR}/slurm/%A_%a.out
#SBATCH --error=${CKPT_DIR}/slurm/%A_%a.err
#SBATCH --qos=${QOS}  # QOS

cd $ROOT_DIR
source ${CONDA_ENV}

mkdir -p ${CKPT_DIR}/slurm/

# Run testing
srun python -m ca_code.scripts.run_test \
    ${CONFIG_FILE} \
    sid=${SID} \
    data.root_path=${DATA_ROOT} \
    data.shared_assets_path=${SHARED_ASSETS} \
    test_path=${CKPT_DIR}/"\${SLURM_ARRAY_TASK_ID}"

# # Run viz
# srun python -m ca_code.scripts.run_vis_relight \
#     ${CONFIG_FILE} \
#     sid=${SID} \
#     data.shared_assets_path=${SHARED_ASSETS} \
#     data.root_path=${DATA_ROOT} \
#     train.run_dir=${CKPT_DIR}/"\${SLURM_ARRAY_TASK_ID}"

EOL

###-----------------------------------------------------------------
# Print the script content in green
echo $SCRIPT
echo -e "\033[0;32m"
cat $SCRIPT
echo -e "\033[0m"

###-----------------------------------------------------------------
# Submit the job
sbatch $SCRIPT

###Optionally, remove the temporary script file
rm -f $SCRIPT

done  # loop over HAND_SIDES
done  # loop over SIDS
