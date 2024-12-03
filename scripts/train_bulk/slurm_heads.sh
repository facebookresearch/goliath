#!/bin/bash

###-----------------------------------------------------------------
### Configuration variables
QOS='urgent_deadline'

###-----------------------------------------------------------------
CONFIG_FILE=config/rgca_example.yml

# CONDA_ENV="/uca/conda-envs/dgxenv-2024-08-16-07-36-10-x7977-centos9-py310-pt231/bin/activate"
CONDA_ENV="/uca/conda-envs/dgxenv-2024-12-02-16-01-45-x9539-centos9-py310-pt251cf/bin/activate"  # gsplat 1.0
TIME='7-00:00:00'


###-----------------------------------------------------------------

ROOT_DIR=/home/julietamartinez/rsc/goliath/

declare -a SIDS=(
    "AXE977"
    # "XKT970"
    # "QVC422"
    # "QZX685"
)
declare -a DATA_ROOTS=(
    "/uca/julieta/oss/goliath/m--20230306--0707--AXE977--pilot--ProjectGoliath--Head/"
    # "/uca/julieta/oss/goliath/m--20230524--0942--XKT970--pilot--ProjectGoliath--Head/"
    # "/uca/julieta/oss/goliath/m--20230714--0903--QVC422--pilot--ProjectGoliath--Head/"
    # "/uca/julieta/oss/goliath/m--20230317--1011--QZX685--pilot--ProjectGoliath--Head/"
)

###-----------------------------------------------------------------
# Loop over for creation of runs
for (( i=0; i<"${#SIDS[@]}"; i++ )); do

SID="${SIDS[i]}"
DATA_ROOT="${DATA_ROOTS[i]}"

JOB_NAME=RGCA_${SID}
CKPT_DIR=/checkpoint/avatar/julietamartinez/goliath/RGCA/${SID}/

###-----------------------------------------------------------------
# Create a temporary script file
SCRIPT=$(mktemp)

# Write the SLURM script to the temporary file
cat > $SCRIPT <<EOL
#!/bin/bash
#SBATCH --partition=learn
#SBATCH --array=5-5
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

# Run training
srun python -m ca_code.scripts.run_train \
    ${CONFIG_FILE} \
    sid=${SID} \
    data.root_path=${DATA_ROOT} \
    test.test_path=${CKPT_DIR} \
    train.run_dir=${CKPT_DIR}/"\${SLURM_ARRAY_TASK_ID}"

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

done
