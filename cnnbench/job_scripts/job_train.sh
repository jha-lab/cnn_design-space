#!/bin/bash

# Script to train a given model

# Author : Shikhar Tuli

autotune="0"
model_hash=""
model_dir=""
config_file=""
neighbor_file=""
dataset=""
graphlib_file=""

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
ENDC='\033[0m'

Help()
{
   # Display Help
   echo -e "${CYAN}Script to train a given model${ENDC}"
   echo
   echo -e "Syntax: source ${CYAN}./job_scripts/job_train.sh${ENDC} [${YELLOW}flags${ENDC}]"
   echo "Flags:"
   echo -e "${YELLOW}-a${ENDC} | ${YELLOW}--autotune${ENDC} [default = ${GREEN}\"0\"${ENDC}] \t\t To autotune the given model"
   echo -e "${YELLOW}-m${ENDC} | ${YELLOW}--model_hash${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Model hash"
   echo -e "${YELLOW}-d${ENDC} | ${YELLOW}--model_dir${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Directory to save the model"
   echo -e "${YELLOW}-f${ENDC} | ${YELLOW}--config_file${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Path to the config file"
   echo -e "${YELLOW}-n${ENDC} | ${YELLOW}--neighbor_file${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Path to the chosen neighbor"
   echo -e "${YELLOW}-s${ENDC} | ${YELLOW}--dataset${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t\t Dataset name"
   echo -e "${YELLOW}-g${ENDC} | ${YELLOW}--graphlib_file${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Path to the graphlib dataset"
   echo -e "${YELLOW}-h${ENDC} | ${YELLOW}--help${ENDC} \t\t\t\t\t Call this help message"
   echo
}

while [[ $# -gt 0 ]]
do
case "$1" in
    -a | --autotune)
        shift
        autotune=$1
        shift
        ;;
    -m | --model_hash)
        shift
        model_hash=$1
        shift
        ;;
    -d | --model_dir)
        shift
        model_dir=$1
        shift
        ;;
    -f | --config_file)
        shift
        config_file=$1
        shift
        ;;
    -n | --neighbor_file)
        shift
        neighbor_file=$1
        shift
        ;;
    -s | --dataset)
        shift
        dataset=$1
        shift
        ;;
    -g | --graphlib_file)
        shift
        graphlib_file=$1
        shift
        ;;
    -h| --help)
       Help
       return 1;
       ;;
    *)
       echo "Unrecognized flag $1"
       return 1;
       ;;
esac
done  


job_file="./job_${model_hash}.slurm"
mkdir -p "./job_scripts/${dataset}/"

cd "./job_scripts/${dataset}/"

# Create SLURM job script to train surrogate model
echo "#!/bin/bash" >> $job_file
echo "#SBATCH --job-name=cnnbench_${dataset}_${model_hash}   # create a short name for your job" >> $job_file
echo "#SBATCH --nodes=1                                      # node count" >> $job_file
echo "#SBATCH --ntasks=1                                     # total number of tasks across all nodes" >> $job_file
echo "#SBATCH --cpus-per-task=20                             # cpu-cores per task (>1 if multi-threaded tasks)" >> $job_file
echo "#SBATCH --mem=128G                                     # memory per cpu-core (4G is default)" >> $job_file
# echo "#SBATCH --gres=gpu:1" >> $job_file
echo "#SBATCH --time=144:00:00                               # total run time limit (HH:MM:SS)" >> $job_file
echo "" >> $job_file
echo "module purge" >> $job_file
echo "module load anaconda3/2020.7" >> $job_file
echo "conda activate cnnbench" >> $job_file
echo "" >> $job_file
echo "cd ../.." >> $job_file
echo "" >> $job_file
echo "python model_trainer.py --config_file ${config_file} \
  --graphlib_file ${graphlib_file} \
  --neighbor_file ${neighbor_file} \
  --model_dir ${model_dir} \
  --model_hash ${model_hash} \
  --autotune ${autotune}" >> $job_file
# echo "export MKL_SERVICE_FORCE_INTEL=1" >> $job_file
# echo "python -c \"import time, torch, random, os, numpy; \
#     os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'; \
#     acc = random.random(); \
#     ckpt = {'train_accuracies': [acc], 'val_accuracies': [acc], 'test_accuracies': [acc]}; \
#     os.makedirs('${model_dir}'); \
#     torch.save(ckpt, os.path.join('${model_dir}', 'model.pt'))\"" >> $job_file

sbatch $job_file