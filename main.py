import os
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from reactranker.data.load_reactions import get_data, Parsing_features
from reactranker.train.utils import build_optimizer, build_lr_scheduler
from reactranker.models.base_model import build_model
from reactranker.train.train_listwise import train
from reactranker.train.test_listwise import test

path = r'input\your\save\path'
data_path = r'input\your\data\path'
val_data_path = None
test_data_path = None
if not os.path.exists(path):
    os.makedirs(path)
log_path = path + '/output.log'
logging.basicConfig(filename=log_path,
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger()
writer = SummaryWriter(path + '/loss_writer')
filtered_size = user_defined

data = get_data(data_path)
data.read_data()
data.filter_bacth(filter_szie=filtered_size)
gpu = your_gpu  # cuda
test_score = []
smiles2graph_dic = Parsing_features()

# training parameter
k_fold = user_defined
batch_size = user_defined
total_epochs = user_defined
task_type = 'listnet'  # to choose loss function
target_name = 'lgk'
smiles_list = ['rsmi_mapped', 'psmi_mapped']  # ['rsmi_mapped', 'psmi_mapped']
split_strategy = 'random_flag'  # 'scaffold' or 'random'
init_lr = user_defined
max_lr = user_defined
final_lr = user_defined
save_metric = 'all'
add_features_dim = 1
add_features_name = 'temp'

if val_data_path is not None and test_data_path is not None:
    logger.info('Note: the test, validate, and test data are already divided.')
logger.info('Training data path is {}, and split strategy is {}'.format(data_path, split_strategy))
logger.info('Use the {} as input smiles'.format(smiles_list))
logger.info('filtered_size is {}'.format(filtered_size))
logger.info('Task type is: {}, and target name is: {}'.format(task_type, target_name))
logger.info('{} fold train with {} epochs every fold. The batch size is: {}'.format(k_fold, total_epochs, batch_size))
logger.info('the initial, maximum, final learning rate are {}, {}, {}'.format(init_lr, max_lr, final_lr))
logger.info('the save metric is: {},'.format(save_metric))
logger.info('additional features is {} dim, and the name is {}'.format(add_features_dim, add_features_name))
logger.info('torch.mean(-torch.log(targets_possibility) + uncertainty_loss + penalty)')
logger.info('change the target possibility to negative, the score without softplus')
if task_type == 'regression_poss':
    mean_targ = np.mean(data.df[target_name].values)
    std_targ = np.std(data.df[target_name].values, ddof=1)
    true_targ = np.exp((data.df[target_name].values - mean_targ) / std_targ)
    data.df[target_name] = true_targ
if save_metric == 'all':
    metric_list = ["T1", "T25_in_T25", "T25"]
    path = [os.path.join(path, i) for i in metric_list]
    if not os.path.exists(path[0]):
        os.makedirs(path[0])
        os.makedirs(path[1])
        os.makedirs(path[2])
for ii in range(k_fold):
    logger.info(
        '\n\n**********************************\n**    This is the fold [{}/{}]   **\n**********************************'.format(
            ii + 1, k_fold))
    print('**********************************')
    print('**   This is the fold [{}/{}]   **'.format(ii + 1, k_fold))
    print('**********************************')
    path_checkpoints = path
    seed = ii
    k_fold_str = str(ii) + '.pt'
    if save_metric != 'all':
        path_checkpoints = os.path.join(path_checkpoints, k_fold_str)
    else:
        path_checkpoints = [os.path.join(i, k_fold_str) for i in path_checkpoints]
    # shuffle data
    if val_data_path is not None and test_data_path is not None:
        train_data = pd.read_csv(data_path)
        val_data = pd.read_csv(val_data_path)
        test_data = pd.read_csv(test_data_path)
        print('this is a test flag', train_data.shape[0], val_data.shape[0], test_data.shape[0])
    else:
        if split_strategy == 'random':
            train_data, val_data, test_data = data.split_data(split_size=(0.8, 0.1, 0.1), split_type='reactants', seed=seed)
        elif split_strategy == 'scaffold':
            train_data, val_data, test_data = data.scaffold_split_data(split_size=(0.8, 0.1, 0.1), balanced=True, seed=seed)
        elif split_strategy == 'random_flag':
            train_data, val_data, test_data = data.split_data(split_size=(0.8, 0.1, 0.1), split_type='flag',
                                                              seed=seed)
        else:
            raise Exception('Split strategy is unknown')

    # train_len = len(train_data.rsmi.unique())
    train_len = train_data.shape[0]
    batch_size = batch_size
    total_epochs = total_epochs
    torch.manual_seed(seed)
    if gpu is not None:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    model = build_model(hidden_size=300,
                        mpnn_depth=3,
                        mpnn_diff_depth=3,
                        ffn_depth=3,
                        use_bias=True,
                        dropout=0.1,
                        task_num=1,
                        ffn_last_layer='with_softplus',
                        add_features_dim=add_features_dim)
                        # task_type=task_type)

    logger.info('Model Structure')
    logger.info(model)
    '''
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['state_dict'])
    '''
    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    optimizer = build_optimizer(model)
    scheduler = build_lr_scheduler(optimizer,
                                   warmup_epochs=2,
                                   total_epochs=total_epochs,
                                   train_data_size=train_len,
                                   batch_size=batch_size,
                                   init_lr=init_lr,
                                   max_lr=max_lr,
                                   final_lr=final_lr)

    train(model,
          scheduler,
          train_data,
          val_data,
          path_checkpoints,
          optimizer,
          total_epochs,
          smiles2graph_dic,
          batch_size=batch_size,
          seed=seed,
          gpu=gpu,
          task_type=task_type,  # to choose the loss function
          writer=writer,
          logger=logger,
          target_name=target_name,
          smiles_list=smiles_list,
          save_metric=save_metric,
          add_features_name=add_features_name)

    print(path_checkpoints)
    if save_metric == 'all':
        test_path = path_checkpoints[0]
    else:
        test_path = path_checkpoints
    score, average_pred_in_targ, score3 = test(model, test_data, test_path, batch_size, smiles2graph_dic,
                                               gpu=gpu, smiles_list=smiles_list, logger=logger, target_name=target_name,
                                               cal_ngcd=True, return_order=False, add_features_name=add_features_name)
    test_score.append([score, average_pred_in_targ, score3])
print("test score for k_fold vailidation is: ", test_score)
logger.info('test score for k_fold vailidation is: {}'.format(test_score))
