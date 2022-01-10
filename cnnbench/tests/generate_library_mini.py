# Graph library generating all possible graphs
# within the design space

# Author : Shikhar Tuli

import argparse
import sys
from os import path

sys.path.append('../')

from library import GraphLib, Graph
from utils import print_util as pu
import manual_models
import yaml

CREATE_GRAPHS = True


def main():
    parser = argparse.ArgumentParser(
        description='Input parameters for generation of dataset library',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_file',
        metavar='', 
        type=str, 
        help='path to yaml file for the design space',
        default='./config_all-ops.yaml')
    parser.add_argument('--dataset_file',
        metavar='',
        type=str,
        help='path to store the dataset',
        default='../dataset/dataset_mini.json')
    parser.add_argument('--kernel',
        metavar='',
        type=str,
        help='kernel for graph similarity computation',
        default='GraphEditDistance')
    parser.add_argument('--algo',
        metavar='',
        type=str,
        help='algorithm for training embeddings',
        default='GD')
    parser.add_argument('--embedding_size',
        metavar='',
        type=int,
        help='dimensionality of embedding',
        default=16)
    parser.add_argument('--num_neighbors',
        metavar='',
        type=int,
        help='number of neighbors to save for each graph',
        default=100)
    parser.add_argument('--n_jobs',
        metavar='',
        type=int,
        help='number of parallel jobs',
        default=8)

    args = parser.parse_args()

    if not path.exists(args.dataset_file):
        # Create an empty graph library with the hyper-parameter ranges
        # given in the design_space file
        graphLib = GraphLib(config=args.config_file)

        # Show generated library
        print(f'{pu.bcolors.OKGREEN}Generated empty library{pu.bcolors.ENDC}')
        print(graphLib)
        print()

        with open(args.config_file) as config_file:
            try:
                config = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                print(exc)

        # Manual models in the mini dataset
        models_mini = ['vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 
                       'resnet50', 'resnet101', 'resnet152', 'shufflenet', 'mobilenet', 'googlenet',
                       'inception', 'xception']

        # Generating graph library
        for i in range(len(models_mini)):
            print(f'{pu.bcolors.HEADER}Adding model: {models_mini[i]} with currently {len(graphLib.library)} models in graphLib...{pu.bcolors.ENDC}')
            model_1 = manual_models.get_manual_graph(config, model_name=models_mini[i])
            graphLib.library.append(model_1)
            for j in range(i+1, len(models_mini)):
                print(f'{pu.bcolors.HEADER}Interpolating between {models_mini[i]} and {models_mini[j]}...{pu.bcolors.ENDC}')
                model_2 = manual_models.get_manual_graph(config, model_name=models_mini[j])
                graphLib.library.extend(graphLib.get_interpolants(model_1, model_2, 1, 1, check_isomorphism=False))

        graphLib.modules_per_stack = 1

        # Simple test to check isomorphisms, rather than comparing hash for every new graph
        hashes = [graph.hash for graph in graphLib.library]
        if len(hashes) == len(set(hashes)):
            print(f'{pu.bcolors.OKGREEN}No isomorphisms detected!{pu.bcolors.ENDC}')
        else:
            print(f'{pu.bcolors.WARNING}Graphs with the same hash encountered!{pu.bcolors.ENDC}')

            hashes = set()
            library_new = []
            for graph in graphLib.library:
                if graph.hash not in hashes:
                    hashes.add(graph.hash)
                    library_new.append(graph)
            graphLib.library = library_new
        print()

        # Save dataset without embeddings to dataset_file
        graphLib.save_dataset(args.dataset_file)
        print()
    else:
        # Load dataset without embeddings from dataset_file
        graphLib = GraphLib.load_from_dataset(args.dataset_file)

        print(f'{pu.bcolors.OKGREEN}Dataset without embeddings loaded from:{pu.bcolors.ENDC}' \
            + f' {args.dataset_file}')
        print()

    # Build embeddings
    if CREATE_GRAPHS:
        graphLib.build_embeddings(embedding_size=args.embedding_size, algo=args.algo,
            kernel=args.kernel, neighbors=args.num_neighbors, n_jobs=args.n_jobs)
        print()

    # Save dataset
    graphLib.save_dataset(args.dataset_file)


if __name__ == '__main__':
    main()




