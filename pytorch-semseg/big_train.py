import os
import yaml
import shutil
import random
import argparse
import numpy as np

from train import train
from ptsemseg.utils import get_logger
from tensorboardX import SummaryWriter

# For anyone reading this, this code is full of side effects. I didn't have
# time to properly implement a class with variables so I just used globals


def setup_logging(name, cfg):
    # Create argparser and logging object to send to train()
    parser = argparse.ArgumentParser()
    # Pass name to the argparser
    parser.add_argument(
        "--name",
        nargs="?",
        type=str,
        default=name,
        help="argparse.SUPPRESS"
    )
    # train() needs this arg but we wont be using it here
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='argparse.SUPPRESS'
    )
    args = parser.parse_args()
    # Define the experiment data directory
    logdir = os.path.join(
        'runs',
        cfg['training']['loss']['name'],
        args.name
    )
    writer = SummaryWriter(log_dir=logdir)
    print('RUNDIR: {}'.format(logdir))
    shutil.copy(config, logdir)

    logger_old = get_logger(logdir)
    logger_old.info('Let the games begin')
    return args, writer, logger_old


def run_experiment(lr_exp, wd_exp):
    # Send input parameters to the cfg to be sent to train()
    wd = 10. ** - wd_exp
    cfg['training']['optimizer']['weight_decay'] = wd
    lr = 10. ** - lr_exp
    cfg['training']['optimizer']['lr'] = lr
    # Define name for directory to save experiment data under
    name = 'sp_' + str(sp_level) + '_lr_' + str(lr_exp) + '_wd_' + str(wd_exp)
    # Set up all datalogging that train() needs
    args, writer, logger_old = setup_logging(name, cfg)
    # Run the experiment
    iou = train(cfg, writer, logger_old, args)
    return iou


def searching(exp_min, exp_max, exp_const, lr=False, wd=False):
    # Flag to signal when line search is finished
    searching = True
    while searching:
        # Define which axis to search
        if lr:
            line = search_grid[exp_min:exp_max+1, exp_const]
        elif wd:
            line = search_grid[exp_const, exp_min:exp_max+1]
        print('line:', line)
        # If max iou is not at end of line, search complete
        if 0 < np.argmax(line) < len(line) - 1:
            exp_var = np.argmax(line) + exp_min
            searching = False
            continue
        # If max iou is at end of line, continue in that direction
        elif np.argmax(line) == 0:
            exp_min -= 1
            exp_var = exp_min
        elif np.argmax(line) == len(line) - 1:
            exp_max += 1
            exp_var = exp_max
        # More defining which line to search
        if lr:
            exp_1 = exp_var
            exp_2 = exp_const
        elif wd:
            exp_1 = exp_const
            exp_2 = exp_var
        print('sp: {}, lr: {}, wd: {}'.format(sp_level, exp_1, exp_2))
        # If experiment has not been run already, run it
        if search_grid[exp_1, exp_2] == np.inf:
            iou = run_experiment(lr_exp=exp_1, wd_exp=exp_2)
            search_grid[exp_1, exp_2] = iou
            # Save checkpoint in case it is needed
            np.save(chkpnt_path, search_grid)
            print('mean iou: {}'.format(search_grid[exp_1, exp_2]))
        else:
            print('mean iou: {}'.format(search_grid[exp_1, exp_2]))
    return exp_var


def find_best_exp(exp_var_init, exp_const, lr=False, wd=False):
    # Run experiments 1 above and below input variable
    exp_min = exp_var_init - 1
    exp_max = exp_var_init + 1
    # Define which axis to run experiments on
    for exp_var in range(exp_min, exp_max+1):
        if lr:
            exp_1 = exp_var
            exp_2 = exp_const
        elif wd:
            exp_1 = exp_const
            exp_2 = exp_var
        print('sp: {}, lr: {}, wd: {}'.format(sp_level, exp_1, exp_2))
        # If experiment has not been run already, run it
        if search_grid[exp_1, exp_2] == np.inf:
            iou = run_experiment(lr_exp=exp_1, wd_exp=exp_2)
            search_grid[exp_1, exp_2] = iou
            # Save checkpoint in case it is needed
            np.save(chkpnt_path, search_grid)
            print('mean iou: {}'.format(search_grid[exp_1, exp_2]))
        else:
            print('mean iou: {}'.format(search_grid[exp_1, exp_2]))
    # Search to see if max is in middle of these 3 experiments
    # (it probably isn't)
    lr_exp = searching(exp_min, exp_max, exp_const, lr, wd)
    return lr_exp


if __name__ == "__main__":
    run_id = random.randint(1, 100000)
    configs = ['ce', 'macro', 'micro', 'ziou']
    sp_levels = [100, 1000, 10000, None]

    # for testing
    sp_levels = [100]

    for config in configs:
        config = os.path.join('./configs', config + '_adam_alexnet.yml')

        with open(config) as fp:
            cfg = yaml.load(fp)

        for sp_level in sp_levels:
            # Log which experiment is happening
            if sp_level is None:
                print('Loss function: {}, Pixel level'.format(
                    cfg['training']['loss']['name'])
                )
            else:
                print('Loss function: {}, Superpixel level: {}'.format(
                    cfg['training']['loss']['name'], sp_level)
                )
            # Set current superpixel level
            cfg['training']['loss']['superpixels'] = sp_level
            # Set initial adam parameters
            lr_exp = 4
            lr = 10. ** - lr_exp
            if sp_level is not None:
                # The more superpixels, the less regularisation is needed
                wd_exp = int(8-np.log10(sp_level))
                wd = 10. ** - wd_exp
            else:
                wd_exp = 3
                wd = 10. ** - wd_exp
            # Initialise/load the search grid for cross validation
            search_grid = np.full((10, 10), np.inf)
            chkpnt_path = os.path.join(
                'runs',
                cfg['training']['loss']['name'],
                'cross_validation_checkpoint_{}_{}.npy'.format(
                    cfg['training']['loss']['name'],
                    sp_level
                )
            )
            if os.path.isfile(chkpnt_path):
                search_grid = np.load(chkpnt_path)
            # Define conditions that will eventially end the experiment
            neighbours = search_grid[lr_exp-1:lr_exp+2, wd_exp-1:wd_exp+2]
            unsampled_neighbours = (neighbours == np.inf).sum()
            # If any neighbours are inf and centre isn't max, keep going
            while (unsampled_neighbours > 0 or
                   neighbours[1, 1] != np.max(neighbours)):
                # Learning rate line search
                lr_exp = find_best_exp(
                    exp_var_init=lr_exp,
                    exp_const=wd_exp,
                    lr=True
                )
                # Current best learning rate
                lr = 10. ** - lr_exp
                cfg['training']['optimizer']['lr'] = lr
                # Update the while conditions
                neighbours = search_grid[lr_exp-1:lr_exp+2, wd_exp-1:wd_exp+2]
                unsampled_neighbours = (neighbours == np.inf).sum()
                print('neighbours:\n', neighbours)
                print('unsampled neighbours:', unsampled_neighbours)
                # Weight decay line search
                wd_exp = find_best_exp(
                    exp_var_init=wd_exp,
                    exp_const=lr_exp,
                    wd=True
                )
                # Current best weight decay
                wd = 10. ** - wd_exp
                cfg['training']['optimizer']['weight_decay'] = wd
                # Update the while conditions
                neighbours = search_grid[lr_exp-1:lr_exp+2, wd_exp-1:wd_exp+2]
                unsampled_neighbours = (neighbours == np.inf).sum()
                print('neighbours:\n', neighbours)
                print('unsampled neighbours:', unsampled_neighbours)
                # If only the corners of the 3x3 haven't been sampled,
                # sample them.
                if 0 < unsampled_neighbours <= 4:
                    if search_grid[lr_exp-1, wd_exp-1] == np.inf:
                        iou = run_experiment(lr_exp=lr_exp-1, wd_exp=wd_exp-1)
                        search_grid[lr_exp-1, wd_exp-1] = iou
                        np.save(chkpnt_path, search_grid)
                    elif search_grid[lr_exp-1, wd_exp+1] == np.inf:
                        iou = run_experiment(lr_exp=lr_exp-1, wd_exp=wd_exp+1)
                        search_grid[lr_exp-1, wd_exp+1] = iou
                        np.save(chkpnt_path, search_grid)
                    elif search_grid[lr_exp+1, wd_exp-1] == np.inf:
                        iou = run_experiment(lr_exp=lr_exp+1, wd_exp=wd_exp-1)
                        search_grid[lr_exp+1, wd_exp-1] = iou
                        np.save(chkpnt_path, search_grid)
                    elif search_grid[lr_exp+1, wd_exp+1] == np.inf:
                        iou = run_experiment(lr_exp=lr_exp+1, wd_exp=wd_exp+1)
                        search_grid[lr_exp+1, wd_exp+1] = iou
                        np.save(chkpnt_path, search_grid)
                # If 3x3 region is populated but the max isn't in the centre,
                # I need to write some more code.
                elif (unsampled_neighbours == 0 and
                      neighbours[1, 1] != np.max(neighbours)):
                    raise Exception("Cross-validation code is broken...")
            # While loop broken
            print('Cross validation complete')
            print('{} locations sampled'.format((neighbours != np.inf).sum()))
            print('sp: {}, lr: {}, wd: {}'.format(sp_level, lr_exp, wd_exp))
            final_chkpnt_path = os.path.join(
                'runs',
                cfg['training']['loss']['name'],
                'cross_validation_final_{}_{}.npy'.format(
                    cfg['training']['loss']['name'],
                    sp_level
                )
            )
            np.save(final_chkpnt_path, search_grid)
            export_path = os.path.join(
                'runs',
                cfg['training']['loss']['name'],
                'cross_validation_final_{}_{}.csv'.format(
                    cfg['training']['loss']['name'],
                    sp_level
                )
            )
            np.savetxt(export_path, search_grid, delimiter=",")
            print('peak iou this experiment: {}'.format(neighbours.max()))
