#!/bin/bash

numNodes=1
numTasks=2
cluster="adroit"
cluster_gpu="gpu:tesla_v100:2"
numVertices=2
numOps=3
baseOps="conv3x3-bn-relu,conv1x1-bn-relu,maxpool3x3"
numRepeats=3
numEpochs=4

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
ENDC='\033[0m'

Help()
{
   # Display Help
   echo "Flags for this script"
   echo
   echo -e "Syntax: source ${CYAN}job_creator_script.sh${ENDC} [${YELLOW}flags${ENDC}]"
   echo "Flags:"
   echo -e "${YELLOW}-n${ENDC} | ${YELLOW}--numNodes${ENDC} [default = ${GREEN}1${ENDC}] \t\t Number of nodes to use in cluster"
   echo -e "${YELLOW}-t${ENDC} | ${YELLOW}--numTasks${ENDC} [default = ${GREEN}2${ENDC}] \t\t Number of tasks across all nodes"
   echo -e "${YELLOW}-c${ENDC} | ${YELLOW}--cluster${ENDC} [default = ${GREEN}\"adroit\"${ENDC}] \t Selected cluster - adroit or tiger"
   echo -e "${YELLOW}-v${ENDC} | ${YELLOW}--numVertices${ENDC} [default = ${GREEN}2${ENDC}] \t Number of vertices per module in NASBench"
   echo -e "${YELLOW}-r${ENDC} | ${YELLOW}--numRepeats${ENDC} [default = ${GREEN}3${ENDC}] \t Number of training repeats for each model"
   echo -e "${YELLOW}-e${ENDC} | ${YELLOW}--numEpochs${ENDC} [default = ${GREEN}4${ENDC}] \t\t Number of training epochs"
   echo -e "${YELLOW}-o${ENDC} | ${YELLOW}--numOps${ENDC} [default = ${GREEN}3${ENDC}] \t\t Number of operations in every module"
   echo -e "${YELLOW}-b${ENDC} | ${YELLOW}--baseOps${ENDC} \t\t\t\t Available base operations"
   echo -e "${YELLOW}-h${ENDC} | ${YELLOW}--help${ENDC} \t\t\t\t Call this help message"
   echo
}

while [[ $# -gt 0 ]]
do
case "$1" in
    -n | --numNodes)
        shift
        numNodes=$1
        shift
        ;;
    -t | --numTasks)
        shift
        numTasks=$1
        shift
        ;;
    -c | --cluster)
        shift
        cluster=$1
        shift
        ;;
    -v | --numVertices)
        shift
        numVertices=$1
        shift
        ;;
    -r | --numRepeats)
        shift
        numRepeats=$1
        shift
        ;;
    -e | --numEpochs)
        shift
        numEpochs=$1
        shift
        ;; 
    -o | --numOps)
        shift
        numOps=$1
        shift
        ;;
    -b | --baseOps)
        shift
        baseOps=$1
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

if [[ $cluster == "adroit" ]]
then
  if [[ $numNodes -gt 1 ]]
	then
    cluster_gpu="gpu:tesla_v100:4"
  else
    cluster_gpu="gpu:tesla_v100:"+$numTasks
  fi
elif [[ $cluster == "tiger" ]]
then
  if [[ $numNodes -gt 1 ]]
  then
    cluster_gpu="gpu:4"
  else
    cluster_gpu="gpu:"+$numTasks
  fi
else
	echo "Unrecognized cluster"
	return 1
fi

numTask_end=$(($numTasks-1))

modelDir="../results/vertices_${numVertices}"

job_file="job_cnnbench_n${numNodes}_t${numTasks}_v${numVertices}_n${numOps}_r${numRepeats}_e${numEpochs}.slurm"

echo "#!/bin/bash
#SBATCH --job-name=cnnbench_multi           # create a short name for your job
#SBATCH --nodes=${numNodes}                 # node count
#SBATCH --ntasks=${numTasks}                # total number of tasks across all nodes
#SBATCH --cpus-per-task=8                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G                    # memory per cpu-core (4G is default)
#SBATCH --gres=${cluster_gpu}               # number of gpus per node
#SBATCH --time=10:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all                     # send email
#SBATCH --mail-user=stuli@princeton.edu

module purge
module load anaconda3/2020.7
conda activate cnnbench

python generate_graphs_script.py --module_vertices ${numVertices} \
--num_ops ${numOps} --output_file '${modelDir}/generated_graphs.json'

for i in {0..${numTask_end}}
do
  python run_evaluation_script.py --worker_id \$i --total_workers ${numTasks} --module_vertices ${numVertices} \
  --available_ops ${baseOps} \
  --num_repeats ${numRepeats} \
  --models_file '${modelDir}/generated_graphs.json' \
  --output_dir '${modelDir}/evaluation' &
done

wait

python cleanup_script.py --cleanup_dir '${modelDir}/evaluation'

python generate_dataset_script.py --model_dir '${modelDir}' --available_ops ${baseOps}" > $job_file

sbatch $job_file