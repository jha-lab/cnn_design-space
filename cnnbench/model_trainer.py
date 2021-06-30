# Builds PyTorch model for the given Graph Object and trains
# the given model. Runs automatic hyper-parameter tuning to
# optimizer performance for the CNN architecture.

# Author : Shikhar Tuli


import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, MedianStoppingRule
from ray.tune.suggest.hebo import HEBOSearch
os.environ['TUNE_GLOBAL_CHECKPOINT_S'] = "20"
# os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = "1"

import numpy as np
from inspect import getmembers
from functools import partial
from copy import deepcopy
import shutil
import hashlib
import json
import time

import multiprocessing as mp
mp.set_start_method('forkserver', force=True)

from input_pipeline import get_loader
from model_builder import CNNBenchModel

from library import GraphLib, Graph
from utils import print_util as pu

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt


LOG_INTERVAL = 10
NUM_SAMPLES = 64 
KEEP_TRIALS = False
HP_SCHDLR = 'ASHA' # One in ['ASHA', 'MSR']
HP_ALGO = 'RAND' # One in ['RAND', 'HEBO']
LIN_SS = False # Needs to be True for HEBO
EARLY_STOP = True


def worker(config: dict, 
           graphObject: 'Graph', 
           device: torch.device = None, 
           model_dir: str = None, 
           auto_tune = False,
           ckpt_interval = 1,
           save_fig = True):
    """Trains or tunes a CNN model with different training recipes
    
    Args:
        config (dict): dictionary of configuration
        graphObject (Graph): Graph object
        device (torch.device, optional): cuda device
        model_dir (str, optional): directory to store the model and metrics
        auto_tune (bool, optional): to use ray-tune for automatic tuning of hyper-parameter,
            else defaults to the first training recipe in config
        ckpt_interval (int, optional): checkpointing interval. If "-1", only last checkpoint is
            saved
        save_fig (bool, optional): to save the learning curves
    """
    torch.manual_seed(0)

    if ckpt_interval == -1: ckpt_interval = config['epochs']

    if model_dir is None:
        if not os.path.exists(os.path.join(config['models_dir'], config['dataset'], graphObject.hash)):
            os.makedirs(os.path.join(config['models_dir'], config['dataset'], graphObject.hash))
    else:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    if not auto_tune:
        hp_config = {}
        
        if 'optimizer' in config.keys():
            hp_config['optimizer'] = config['optimizer']
        
        if 'scheduler' in config.keys():
            hp_config['scheduler'] = config['scheduler']

        train(hp_config, config, graphObject, device, model_dir, auto_tune=False, 
            checkpointing=True, ckpt_interval=ckpt_interval)

        checkpoints = os.listdir(model_dir)
        checkpoints = [c for c in checkpoints if c.startswith('model')]

        sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))

        # TODO: implement saving model from checkpoints with best validation accuracy

        best_checkpoint = torch.load(os.path.join(model_dir, sorted_checkpoints[-1]))

        if save_fig: plot_metrics(best_checkpoint, model_dir)

    else:
        # Implementing a basic hyper-parameter search space
        if not LIN_SS:
            hp_config = {'optimizer':
                            tune.choice(
                            [{'Adam': 
                                {'lr': tune.loguniform(1e-5, 1e-2),
                                'betas': [tune.uniform(0.8, 0.95), tune.uniform(0.9, 0.999)],
                                'weight_decay': tune.loguniform(1e-5, 1e-3)}},
                            {'AdamW': 
                                {'lr': tune.loguniform(1e-5, 1e-2),
                                'betas': [tune.uniform(0.8, 0.95), tune.uniform(0.9, 0.999)],
                                'weight_decay': tune.loguniform(1e-5, 1e-3)}}]),    
                         'scheduler':
                            tune.choice(
                            [{'CosineAnnealingLR':
                                {'T_max': config['epochs']}},
                             {'ExponentialLR':
                                {'gamma': tune.uniform(0.8, 0.99)}},
                             {'CosineAnnealingWarmRestarts':
                                {'T_0': tune.choice([10, 20]),
                                 'T_mult': tune.choice([1, 2, 4])}}])}
        else:
            # Implement linearized search space
            hp_config = {'optimizer': tune.choice(['Adam', 'AdamW']),
                         'lr': tune.uniform(1e-5, 1e-2),
                         'beta1': tune.uniform(0.8, 0.95),
                         'beta2': tune.uniform(0.9, 0.999),
                         'weight_decay': tune.loguniform(1e-5, 1e-3),
                         'scheduler': tune.choice(['CosineAnnealingLR', 'ExponentialLR', 'CosineAnnealingWarmRestarts']),
                         'T_max': config['epochs'],
                         'gamma': tune.uniform(0.8, 0.99),
                         'T_0': tune.choice([10, 20]),
                         'T_mult': tune.choice([1, 2, 4])}

        if HP_SCHDLR == 'ASHA':
            # Implement Asynchronous Successive Halving scheduler
            scheduler = ASHAScheduler(
                time_attr="training_iteration",
                metric="val_loss",
                mode="min",
                max_t=config['epochs'],
                grace_period=10,
                reduction_factor=2)

            if HP_ALGO == 'HEBO':
                search_alg = HEBOSearch(
                    metric="val_loss",
                    mode="min",
                    max_concurrent=4)
            else:
                search_alg = None
        elif HP_SCHDLR == 'MSR':
            # Implement Median Stopping Rule
            scheduler = MedianStoppingRule(
                time_attr="training_iteration",
                metric="val_loss",
                mode="min",
                grace_period=10)

            if HP_ALGO == 'HEBO':
                search_alg = HEBOSearch(
                    metric="val_loss",
                    mode="min",
                    max_concurrent=4)
            else:
                search_alg = None

        if EARLY_STOP:
            # Implement early stopping.  
            stopper = tune.stopper.ExperimentPlateauStopper(
                metric="val_loss", 
                top=4, 
                std=1e-4,
                mode="min", 
                patience=30)
        else:
            stopper = None

        assert ckpt_interval == 1, 'Checkpoint interval should be 1 for early stopping'

        reporter = CLIReporter(parameter_columns=['optimizer', 'scheduler'],
            metric_columns=["val_loss", "val_accuracy", "training_iteration"])

        assert torch.cuda.device_count() > 1, 'More than one GPU is required for automatic tuning'

        # Implement trial runs to see if model fits in half the GPU memory. 
        # Gives robust batch sizes for automatic tuning.
        small_batch_config = run_trial(config, graphObject, model_dir, gpuFrac=0.4)
        
        print(f'{pu.bcolors.OKGREEN}Selected batch size:{pu.bcolors.ENDC} {small_batch_config["train_batch_size"]}')

        print(f'{pu.bcolors.OKBLUE}Running automatic hyper-parameter tuning{pu.bcolors.ENDC}')

        print(f'{pu.bcolors.OKBLUE}Using scheduler: {HP_SCHDLR} w/ search algo: {HP_ALGO} ' \
            + f'{"w/" if EARLY_STOP else "w/o"} early stopping{pu.bcolors.ENDC}')

        # Run automatic hyper-parameter tuning
        result = tune.run(
            partial(train, 
                main_config=small_batch_config, 
                graphObject=graphObject, 
                device=device, 
                model_dir=model_dir,
                auto_tune=True,
                checkpointing=True,
                ckpt_interval=ckpt_interval,
                gpuFrac=None),
            local_dir=model_dir,
            name='auto_tune',
            trial_name_creator=get_trial_name,
            trial_dirname_creator=get_trial_name,
            resources_per_trial={'gpu': 0.5},
            config=hp_config,
            num_samples=NUM_SAMPLES,
            keep_checkpoints_num=1, # Stores only best checkpoint to save space
            checkpoint_score_attr='min-val_loss', # Stores checkpoint with minimum loss
            stop=stopper, # Early stopping
            scheduler=scheduler,
            search_alg=search_alg,
            progress_reporter=reporter)

        best_trial = result.get_best_trial(metric="val_loss", mode="min", scope="last")
        best_hp_config = best_trial.config

        print(f'{pu.bcolors.OKGREEN}Best hyper-parameter set:{pu.bcolors.ENDC}\n{best_hp_config}')

        # Get path to directory with best model
        best_model_dir = os.path.join(best_trial.checkpoint.value)
        
        # Save best model path to a file in model_dir
        with open(os.path.join(model_dir, 'best_model_dir.txt'), 'w+') as f:
            f.write(best_model_dir)

        with open(os.path.join(model_dir, 'best_model_dir.txt'), 'r') as f:
            best_model_dir = f.read()

        # Copy the checkpoint of the best model to model_dir
        for ckpt in os.listdir(best_model_dir):
            if ckpt.startswith('model'):
                full_file_name = os.path.join(best_model_dir, ckpt)
                shutil.copy(full_file_name, os.path.join(model_dir, 'model.pt'))

        best_checkpoint = torch.load(os.path.join(model_dir, 'model.pt'))

        print(f'{pu.bcolors.OKGREEN}Best model\'s performance:{pu.bcolors.ENDC}')
        print(f'Train Accuracy: {best_checkpoint["train_accuracies"][-1] : 0.02f}%')
        print(f'Validation Accuracy: {best_checkpoint["val_accuracies"][-1] : 0.02f}%')
        print(f'Validation Loss: {best_checkpoint["val_losses"][-1] : 0.02f}')
        print(f'Test Accuracy: {best_checkpoint["test_accuracies"][-1] : 0.2f}%')

        if save_fig: 
            plot_metrics(best_checkpoint, model_dir)

            # Obtain a trial dataframe from all run trials
            dfs = result.trial_dataframes

            ax = None
            for d in dfs.values():
                ax = d.val_loss.plot(ax=ax, legend=False)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Validation Loss')
            ax.figure.savefig(os.path.join(model_dir, 'all_curves.png'))

        # Remove trials if KEEP_TRIALS is False
        if not KEEP_TRIALS:
            shutil.rmtree(os.path.join(model_dir, 'auto_tune'))
            os.remove(os.path.join(model_dir, 'best_model_dir.txt'))


def run_trial(config: dict, graphObject, model_dir, gpuFrac):

    trial_config = deepcopy(config)
    trial_config['epochs'] = 1
    trial_hp_config = {}

    if 'optimizer' in config.keys():
        trial_hp_config['optimizer'] = config['optimizer']
    if 'scheduler' in config.keys():
        trial_hp_config['scheduler'] = config['scheduler']

    print(f'{pu.bcolors.OKBLUE}Trial run to test batch size{pu.bcolors.ENDC}')

    # Robust batch sizing for different model sizes
    while (trial_config['train_batch_size'] >= 1) and (trial_config['test_batch_size'] >= 1):
        print(f'{pu.bcolors.OKBLUE}Running trial with batch size:{pu.bcolors.ENDC} {trial_config["train_batch_size"]}')
        try:
            # Do a trial run
            trial_device = torch.device('cuda:0')
            p = mp.Process(target=train, args=(trial_hp_config, trial_config, graphObject, trial_device, model_dir),
                kwargs={'auto_tune': False, 'checkpointing': False, 'gpuFrac': gpuFrac})
            p.start()
            p.join()
            if p.exitcode != 0:
                raise RuntimeError
        except RuntimeError as e:
            if (trial_config['train_batch_size'] > 1) and (trial_config['test_batch_size'] > 1):
                trial_config['train_batch_size'] = trial_config['train_batch_size']//2
                trial_config['test_batch_size'] = trial_config['test_batch_size']//2
            else:
                raise ValueError('RuntimeError faced even with batch size of 1')
        except Exception as e:
            raise e
        else:
            break

    trial_config['epochs'] = config['epochs']

    return trial_config


def get_trial_name(trial: tune.trial.Trial):
    """Returns the hash for a given hyper-parameter configuration
    
    Args:
        trial (tune.trial.Trial): ray-tune trial
    
    Returns:
        trial_name: string with trial name consisting of the optimizer, 
            scheduler and a unique hash for the hyper-parameter configuration.
    """
    if not LIN_SS:
        opt = list(trial.config['optimizer'].keys())[0]
        schdlr = list(trial.config['scheduler'].keys())[0]
    else:
        opt = trial.config['optimizer']
        schdlr = trial.config['scheduler']
    return '_'.join([opt,
                     schdlr,
                     hashlib.shake_128(str(trial.config).encode('utf-8')).hexdigest(5)])


def plot_metrics(checkpoint, model_dir):
    """Plot and save metrics for the given checkpoint
    """
    epochs = checkpoint['epochs']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    train_accuracies = checkpoint['train_accuracies']
    val_accuracies = checkpoint['val_accuracies']
    test_accuracies = checkpoint['test_accuracies']
    model_params = checkpoint['model_params']
    model_name = checkpoint['model_name']
    times = checkpoint['times']

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    train_loss, = ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    val_loss, = ax1.plot(epochs, val_losses, 'b--', label='Validation Loss')
    train_acc, = ax2.plot(epochs, train_accuracies, 'r-', label='Train Accuracy')
    val_acc, = ax2.plot(epochs, val_accuracies, 'r-.', label='Validation Accuracy')
    test_acc, = ax2.plot(epochs, test_accuracies, 'r--', label='Test Accuracy')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy (%)')
    ax1.yaxis.label.set_color(train_loss.get_color())
    ax2.yaxis.label.set_color(train_acc.get_color())
    ax1.tick_params(axis='y', colors=train_loss.get_color())
    ax2.tick_params(axis='y', colors=train_acc.get_color())
    ax1.legend(handles=[train_loss, val_loss, train_acc, val_acc, test_acc], loc='center right')

    plt.title(f'Model: {model_name}. Params: {pu.human_format(model_params)}. Time: {times[-1]/3600 : 0.2f}h')

    plt.savefig(os.path.join(model_dir, 'model_curves.png'))
    plt.clf()
    

def train(config, 
          main_config, 
          graphObject, 
          device, 
          model_dir, 
          auto_tune, 
          checkpointing = True,
          ckpt_interval = 1,
          checkpoint_dir = None,
          gpuFrac = None):
    """Trains a CNN model on a given device
    
    Args:
        config (dict): dictionary of hyper-parameters of the training recipe
        main_config (dict): dictionary of main configuration
        graphObject (Graph): Graph object
        device (torch.device, optional): cuda device
        model_dir (str, optional): directory to store the model and metrics
        auto_tune (bool): if ray-tune is running
        checkpointing (bool, optional): to checkpoint trained models
        ckpt_interval (int, optional): checkpointing interval
        checkpoint_dir (None, optional): directory where checkpoint is stored
        gpuFrac (float): fraction of GPU memory to be alloted for training given model
    
    Raises:
        ValueError: if no GPU is found
    """
    print(f'{pu.bcolors.OKBLUE}Given hyper-parameters:{pu.bcolors.ENDC}\n{config}')

    model = CNNBenchModel(main_config, graphObject)

    model_params = model.get_params()

    train_loader, val_loader, test_loader, total_size, val_size = get_loader(main_config)
    train_size, test_size = int(total_size - val_size), len(test_loader.dataset)

    if gpuFrac:
        # GPU memory fraction only to be defined during trial run on a single GPU
        assert device is not None and device.index is not None
        torch.cuda.set_per_process_memory_fraction(float(gpuFrac), device.index)

    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
        else:
            raise ValueError('No GPU device found!') 

    model.to(device)

    if 'manual_models' in model_dir:
        model_name = model_dir.split('/')[-1]
    else:
        model_name = graphObject.hash

    optims = [opt[0] for opt in getmembers(optim)]

    if 'optimizer' in config.keys():
        if not LIN_SS:
            opt = list(config['optimizer'].keys())[0]
            if opt not in optims:
                raise ValueError(f'Optimizer {opt} not supported in PyTorch')
            for hp in config['optimizer'][opt].keys():
                if type(config['optimizer'][opt][hp]) not in [float, int]:
                    if type(config['optimizer'][opt][hp]) is list:
                        for i in range(len(config['optimizer'][opt][hp])):
                            if type(config['optimizer'][opt][hp][i]) not in [float, int]:
                                config['optimizer'][opt][hp][i] = config['optimizer'][opt][hp][i].sample()
                    else:
                        config['optimizer'][opt][hp] = config['optimizer'][opt][hp].sample()
            optimizer = eval(f'optim.{opt}(model.parameters(), **config["optimizer"][opt])')
        else:
            opt = config['optimizer']
            if opt not in optims:
                raise ValueError(f'Optimizer {opt} not supported in PyTorch')
            optimizer = eval(f'optim.{opt}(model.parameters(), lr={config["lr"]}, ' \
                + f'betas=(config["beta1"], config["beta2"]), weight_decay={config["weight_decay"]})')
    else:
        opt = 'AdamW'
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)

    shdlrs = [sh[0] for sh in getmembers(optim.lr_scheduler)]

    if 'scheduler' in config.keys():
        if not LIN_SS:
            schdlr = list(config['scheduler'].keys())[0]
            if schdlr not in shdlrs:
                raise ValueError(f'Scheduler {schdlr} not supported in PyTorch')
            for hp in config['scheduler'][schdlr].keys():
                if type(config['scheduler'][schdlr][hp]) is not float and type(config['scheduler'][schdlr][hp]) is not int:
                    config['scheduler'][schdlr][hp] = config['scheduler'][schdlr][hp].sample()
            scheduler = eval(f'optim.lr_scheduler.{schdlr}(optimizer, **config["scheduler"][schdlr])')
        else:
            schdlr = config['scheduler']
            if schdlr not in shdlrs:
                raise ValueError(f'Scheduler {schdlr} not supported in PyTorch')
            if schdlr == 'CosineAnnealingLR':
                schdlr_args = {'T_max': config['T_max']}
            elif schdlr == 'ExponentialLR':
                schdlr_args = {'gamma': config['gamma']}
            elif schdlr == 'CosineAnnealingWarmRestarts':
                schdlr_args = {'T_0': config['T_0'], 'T_mult': config['T_mult']}
            else:
                raise ValueError(f'Scheduler {schdlr} not supported in search space yet')
            scheduler = eval(f'optim.lr_scheduler.{schdlr}(optimizer, **schdlr_args)')
    else:
        schdlr = 'CosineAnnealingLR'
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(main_config['epochs']))

    train_losses = []
    val_losses = []
    epochs = []
    learning_rates = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    times = []

    start_time = time.time()

    last_epoch = 0
    batch_size = 0

    if checkpoint_dir:
        print(f'{pu.bcolors.OKBLUE}Loading from checkpoint.{pu.bcolors.ENDC}')
        for ckpt in os.listdir(checkpoint_dir):
            if ckpt.startswith('model'):
                full_file_name = os.path.join(checkpoint_dir, ckpt)
        checkpoint = torch.load(full_file_name)
        epochs = checkpoint['epochs']
        last_epoch = checkpoint['epochs'][-1]
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        learning_rates = checkpoint['learning_rates']
        train_accuracies = checkpoint['train_accuracies']
        test_accuracies = checkpoint['test_accuracies']
        times = checkpoint['times'] # Can lead to non-monotonic times
        
    for epoch in range(last_epoch + 1, main_config['epochs'] + 1):
        # Run training
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            if epoch == 1 and batch_idx == 0:
                batch_size = len(data)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if schdlr == 'CosineAnnealingWarmRestarts':
                scheduler.step(epoch + batch_idx / train_size)
            
            if batch_idx % LOG_INTERVAL == 0:
                print('Train Epoch: {} [{:6d}/{:6d} ({:.0f}%)]\tLearning Rate: {:.6f}\tLoss: {:.6f}'.format(
                    epoch, batch_idx * batch_size, train_size,
                    100. * batch_idx / len(train_loader), optimizer.param_groups[0]['lr'], loss.item()))
            
            if gpuFrac:
                # Dry run complete
                break

        # Run validation
        model.eval()
        with torch.no_grad():
            train_loss = 0
            correct = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

            train_loss /= train_size

            print('\nTrain set:\tAverage loss: {:.4f},\tAccuracy: {}/{} ({:.0f}%)'.format(
                train_loss, correct, train_size,
                100. * correct / train_size))

            train_accuracies.append(100. * correct / train_size)

            val_loss = 0
            correct = 0
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

            val_loss /= val_size

            print('Val set:\tAverage loss: {:.4f},\tAccuracy: {}/{} ({:.0f}%)'.format(
                val_loss, correct, val_size,
                100. * correct / val_size))

            val_accuracies.append(100. * correct / val_size)

            test_loss = 0
            correct = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= test_size

            print('Test set:\tAverage loss: {:.4f},\tAccuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, test_size,
                100. * correct / test_size))

            test_accuracies.append(100. * correct / test_size)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epochs.append(epoch)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        times.append(time.time() - start_time)

        if checkpointing and epoch % ckpt_interval == 0:
            if auto_tune:
                with tune.checkpoint_dir(step=epoch) as tune_checkpoint_dir:
                    checkpoint_dir = tune_checkpoint_dir
            else:
                checkpoint_dir = model_dir

            if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
            
            ckpt_path = os.path.join(checkpoint_dir, f'model_{epoch}.pt')

            torch.save({'config': main_config,
                        'hp_config': config,
                        'model_name': model_name,
                        'model_params': model_params,
                        'graphObject': graphObject,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epochs': epochs,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'learning_rates': learning_rates,
                        'train_accuracies': train_accuracies,
                        'val_accuracies': val_accuracies,
                        'test_accuracies': test_accuracies,
                        'times': times}, ckpt_path)

            print(f'{pu.bcolors.OKGREEN}Saved checkpoint to:{pu.bcolors.ENDC} {ckpt_path}')

        if auto_tune:
            tune.report(val_loss=val_losses[-1], val_accuracy=val_accuracies[-1])

        if schdlr == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        elif schdlr != 'CosineAnnealingWarmRestarts':
            scheduler.step()
