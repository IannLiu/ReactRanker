"""
this scrip is a demo for test ranknet including:
loading data and perform ranknet
"""
import os
import logging

import torch
from torch.utils.tensorboard import SummaryWriter

from reactranker.data.load_reactions import get_data, Parsing_features
from reactranker.train.utils import build_optimizer, build_lr_scheduler
from reactranker.models.base_model import build_model
from reactranker.train.train_listwise import train
from reactranker.train.test_listwise import test

path = r'C:\Users\5019\Desktop\ReactionRanker\model_results\MLEDis_Gaussian\run_benchmark_test2'
if not os.path.exists(path):
    os.makedirs(path)
log_path = path + '/listmle_test.log'
logging.basicConfig(filename=log_path,
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger()
writer = SummaryWriter(path + '/loss_writer')

data = get_data('C:\\Users\\5019\\Desktop\\data\\b97d3.csv')
data.read_data()
data.filter_bacth(filter_szie=3)
k_fold = 10
gpu = 1  # cuda
test_score = []
smiles2graph_dic = Parsing_features()
#log_dir = r'C:\Users\5019\Desktop\MPN\benchmark_results\MLEDis_Gaussian\run_benchmark\0.pt'

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
    # shuffle data randomly
    train_data, val_data, test_data = data.split_data(split_size=(0.8, 0.1, 0.1), split_type='reactants', seed=seed)

    train_len = len(train_data.rsmi.unique())
    batch_size = 2
    total_epochs = 100
    torch.manual_seed(seed)
    if gpu is not None:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    model = build_model(hidden_size=300,
                        mpnn_depth=3,
                        mpnn_diff_depth=3,
                        ffn_depth=3,
                        use_bias=True,
                        dropout=0,
                        task_num=2,
                        ffn_last_layer='no_softplus')

    logger.info('Model Sturture')
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
                                   init_lr=0.0001,
                                   max_lr=0.0001,
                                   final_lr=0.00001)

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
          task_type='mledis_gaussian',  # to choose the loss function
          writer=writer,
          logger=logger)

    print(path_checkpoints)
    score, score3, average_pred_in_targ = test(model, test_data, path_checkpoints, batch_size, smiles2graph_dic, gpu=gpu,
                                               logger=logger)
    test_score.append([score, score3])
print("test score for k_fold vailidation is: ", test_score)
logger.info('test score for k_fold vailidation is: {}'.format(test_score))
