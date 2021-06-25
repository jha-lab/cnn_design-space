#!/bin/bash

# Sript to run manually defined models in the CNN design space
# Author : Shikhar Tuli

manual_models=("lenet" "alexnet" "vgg11" "vgg13" "vgg16" "vgg19" "resnet18" "resnet34" "resnet50" "resnet101" "resnet152" "shufflenet" \
	"mobilenet" "googlenet" "xception")
cluster="della"
id="stuli"
nodes="ee"
tune=1

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
ENDC='\033[0m'

Help()
{
	# Display Help
	echo -e "${CYAN}Sript to load GLUE dataset into cache using load_glue_datset.py" 
	echo -e "and then create a job script that trains a surrogate model for the"
	echo -e "given task.${ENDC}"
	echo
	echo -e "Syntax: source ${CYAN}job_creator_script.sh${ENDC} [${YELLOW}flags${ENDC}]"
	echo "Flags:"
	echo -e "${YELLOW}-c${ENDC} | ${YELLOW}--cluster${ENDC} [default = ${GREEN}\"della\"${ENDC}] \t Selected cluster - adroit, tiger to della"
	echo -e "${YELLOW}-t${ENDC} | ${YELLOW}--tune${ENDC} [default = ${GREEN}\"0\"${ENDC}] \t\t To tune training recipe"
	echo -e "${YELLOW}-n${ENDC} | ${YELLOW}--nodes${ENDC} [default = ${GREEN}\"ee\"${ENDC}] \t\t Selected nodes for della in [\"ee\", \"all\"]"
	echo -e "${YELLOW}-i${ENDC} | ${YELLOW}--id${ENDC} [default = ${GREEN}\"stuli\"${ENDC}] \t\t Selected PU-NetID to email slurm updates"
	echo -e "${YELLOW}-h${ENDC} | ${YELLOW}--help${ENDC} \t\t\t\t Call this help message"
	echo
}

while [[ $# -gt 0 ]]
do
	case "$1" in
		-c | --cluster)
			shift
			cluster=$1
			shift
			;;
		-t | --tune)
			shift
			tune=$1
			shift
			;;
		-i | --id)
			shift
			id=$1
			shift
			;;
		-n | --nodes)
			shift
			nodes=$1
			shift
			;;
		-h | --help)
			Help
			return 1;
			;;
		*)
		echo "Unrecognized flag $1"
			return 1;
			;;
	esac
done  

if [[ $tune == 0 ]]
then
	if [[ $cluster == "adroit" ]]
	then
		cluster_gpu="gpu:tesla_v100:1"
	elif [[ $cluster == "tiger" ]]
	then
		cluster_gpu="gpu:1"
	elif [[ $cluster == "della" ]]
	then
		cluster_gpu="gpu:1"
	else
		echo "Unrecognized cluster"
		return 1
	fi
else
	if [[ $cluster == "adroit" ]]
	then
		cluster_gpu="gpu:tesla_v100:2"
	elif [[ $cluster == "tiger" ]]
	then
		cluster_gpu="gpu:2"
	elif [[ $cluster == "della" ]]
	then
		cluster_gpu="gpu:2"
	else
		echo "Unrecognized cluster"
		return 1
	fi
fi

counter=1
max_ee=4

if [[ $tune == 1 ]]
then
	max_ee=2
fi

for model in ${manual_models[*]}
do
	job_file="job_${model}.slurm"

	rm -rf $job_file

	# Create SLURM job script to train surrogate model
	echo "#!/bin/bash" >> $job_file
	echo "#SBATCH --job-name=cnnbench_${model}        # create a short name for your job" >> $job_file
	if [[ $cluster == "della" ]]
	then
		if [[ $counter -le $max_ee || $nodes == "ee" ]]
		then
			echo "#SBATCH --partition gpu-ee				  # prioritize della-i12g[1-2] nodes" >> $job_file
		fi
	fi
	echo "#SBATCH --nodes=1                           # node count" >> $job_file
	echo "#SBATCH --ntasks=1                          # total number of tasks across all nodes" >> $job_file
	echo "#SBATCH --cpus-per-task=64                  # cpu-cores per task (>1 if multi-threaded tasks)" >> $job_file
	echo "#SBATCH --mem-per-cpu=2G                    # memory per cpu-core (4G is default)" >> $job_file
	echo "#SBATCH --gres=${cluster_gpu}               # number of gpus per node" >> $job_file
	echo "#SBATCH --time=48:00:00                     # total run time limit (HH:MM:SS)" >> $job_file
	echo "#SBATCH --mail-type=all                     # send email" >> $job_file
	echo "#SBATCH --mail-user=${id}@princeton.edu" >> $job_file
	echo "" >> $job_file
	echo "module purge" >> $job_file
	echo "module load anaconda3/2020.7" >> $job_file
	echo "conda activate cnnbench" >> $job_file
	echo "" >> $job_file
	echo "cd .." >> $job_file
	echo "" >> $job_file
	if [[ $tune == 0 ]]
	then
		echo "python manual_models.py --model_name ${model}" >> $job_file
	else
		echo "python manual_models.py --model_name ${model} --model_dir /scratch/gpfs/stuli/cnn_design-space/results/manual_models/${model}_tune --auto_tune" >> $job_file
	fi

	((counter++))

	sbatch $job_file
done