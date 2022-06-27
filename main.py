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

path = r'C:\Users\5019\Desktop\ReactionRanker\model_results\picture_data\similarity_seperated_size\UClistwise_500'
data_path = 'C:\\Users\\5019\\Desktop\\data\\different_size_seperated_by_similarity\\b97d3_with_rate_500.csv'
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
filtered_size = 3

data = get_data(data_path)
data.read_data()
data.filter_bacth(filter_szie=filtered_size)
gpu = 0  # cuda
test_score = []
smiles2graph_dic = Parsing_features()

# training parameter
k_fold = 10
batch_size = 2
total_epochs = 100
task_type = 'evidential_ranking'  # to choose loss function
target_name = 'lgk'
smiles_list = None  # ['rsmi_mapped', 'psmi_mapped']
split_strategy = 'random'  # 'scaffold' or 'random'
init_lr = 0.0001
max_lr = 0.001
final_lr = 0.0001
save_metric = None

if val_data_path is not None and test_data_path is not None:
    logger.info('Note: the test, validate, and test data are already divided.')
logger.info('Training data path is {}, and split strategy is {}'.format(data_path, split_strategy))
logger.info('Use the {} as input smiles'.format(smiles_list))
logger.info('filtered_size is {}'.format(filtered_size))
logger.info('Task type is: {}, and target name is: {}'.format(task_type, target_name))
logger.info('{} fold train with {} epochs every fold. The batch size is: {}'.format(k_fold, total_epochs, batch_size))
logger.info('the initial, maximum, final learning rate are {}, {}, {}'.format(init_lr, max_lr, final_lr))
logger.info('the save metric is: {},'.format(save_metric))
logger.info('torch.mean(torch.log(targets_possibility) + uncertainty_loss + penalty)')
if task_type == 'regression_poss':
    mean_targ = np.mean(data.df[target_name].values)
    std_targ = np.std(data.df[target_name].values, ddof=1)
    true_targ = np.exp((data.df[target_name].values - mean_targ) / std_targ)
    data.df[target_name] = true_targ

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
    path_checkpoints = os.path.join(path_checkpoints, k_fold_str)
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
        else:
            raise Exception('Split strategy is unknown')

    train_len = len(train_data.rsmi.unique())
    # train_len = train_data.shape[0]
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
                        task_num=2,
                        ffn_last_layer='with_softplus',)
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
          save_metric=save_metric)

    print(path_checkpoints)
    score, average_pred_in_targ, score3 = test(model, test_data, path_checkpoints, batch_size, smiles2graph_dic,
                                               gpu=gpu, smiles_list=smiles_list, logger=logger, target_name=target_name,
                                               cal_ngcd=True, return_order=False)
    test_score.append([score, average_pred_in_targ, score3])
print("test score for k_fold vailidation is: ", test_score)
logger.info('test score for k_fold vailidation is: {}'.format(test_score))
