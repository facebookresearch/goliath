#!/bin/bash

###-----------------------------------------------------------------
### Configuration variables
QOS='urgent_deadline'

###-----------------------------------------------------------------
CONFIG_FILE=config/mesh_vae_example.yml

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
    "/uca/julieta/oss/goliath/s--20230306--1330--AXE977--pilot--ProjectGoliath--MinimalBody"
    "/uca/julieta/oss/goliath/s--20230306--1509--AXE977--pilot--ProjectGoliath--ClothedBody"
    "/uca/julieta/oss/goliath/s--20230524--1235--XKT970--pilot--ProjectGoliath--MinimalBody"
    "/uca/julieta/oss/goliath/s--20230524--1356--XKT970--pilot--ProjectGoliath--ClothedBody"
    "/uca/julieta/oss/goliath/s--20230714--1233--QVC422--pilot--ProjectGoliath--MinimalBody"
    "/uca/julieta/oss/goliath/s--20230714--1345--QVC422--pilot--ProjectGoliath--ClothedBody"
    "/uca/julieta/oss/goliath/s--20230317--1336--QZX685--pilot--ProjectGoliath--MinimalBody"
    "/uca/julieta/oss/goliath/s--20230317--1516--QZX685--pilot--ProjectGoliath--ClothedBody"
)
declare -a BODY_TYPES=("MinimalBody" "ClothedBody")

k=0  # Data root index
for (( i=0; i<"${#SIDS[@]}"; i++ )); do
for (( j=0; j<"${#BODY_TYPES[@]}"; j++ )); do

###-----------------------------------------------------------------
# Assign variables from loop
SID="${SIDS[i]}"
BODY_TYPE="${BODY_TYPES[j]}"

echo "meow $i"
echo $BODY_TYPE

DATA_ROOT="${DATA_ROOTS[k]}"--134cams

SHARED_ASSETS=/uca/julieta/oss/goliath/shared/static_assets_body.pt

###-----------------------------------------------------------------
JOB_NAME=TEST_BODY_MESH_VAE_${SID}_${BODY_TYPE}
CKPT_DIR=/checkpoint/avatar/julietamartinez/goliath/body/${SID}/${BODY_TYPE}

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

export TORCH_HOME=/home/julietamartinez/rsc/TORCH_HOME

mkdir -p ${CKPT_DIR}/slurm/

# Run training
srun python -m ca_code.scripts.run_test \
    ${CONFIG_FILE} \
    sid=${SID} \
    data.root_path=${DATA_ROOT} \
    test_path=${CKPT_DIR}/"\${SLURM_ARRAY_TASK_ID}"
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

let k++
done  # loop over BODY_TYPES
done  # loop over SIDS
