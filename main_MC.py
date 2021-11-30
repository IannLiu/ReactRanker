"""
this scrip is a demo for test ranknet including:
loading data and perform ranknet
"""
import os
import logging


import torch
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter

from reactranker.utils import get_data
from reactranker.train.utils import build_optimizer,build_lr_scheduler
from reactranker.models.base_model import build_model
from reactranker.train.train_listwise import train
from reactranker.train.test_listwise import test

path = r'C:\Users\5019\Desktop\ReactionRanker\model_results\MLE_MC\run_benchmark_drop02'
if not os.path.exists(path):
    os.makedirs(path)
log_path = path + '/listmle_test.log'
logging.basicConfig(filename=log_path,
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
writer = SummaryWriter(path + '/loss_writer')

data = get_data('C:\\Users\\5019\\Desktop\\data\\b97d3_list.csv')
data_len = len(data)
sizes = (0.8, 0.1, 0.1)
train_len = int(sizes[0] * data_len)
val_len = int(sizes[1] * data_len)
test_len = int(sizes[1] * data_len)

k_fold = 10
gpu = 1  # cuda
test_score = []
#log_dir = r'C:\Users\5019\Desktop\MPN\benchmark_results\MLEDis_Gaussian\run_benchmark\0.pt'

for ii in range(k_fold):
    logging.info(
        '\n\n**********************************\n**    This is the fold [{}/{}]   **\n**********************************'.format(
            ii + 1, k_fold))
    print('**********************************')
    print('**   This is the fold [{}/{}]   **'.format(ii + 1, k_fold))
    print('**********************************')
    path_checkpoints = path
    seed = ii
    k_fold_str = str(ii) + '.pt'
    # shuffle data randomly
    ini_index = list(range(data_len))
    index = shuffle(ini_index, random_state=ii)
    shuffled_data = []
    for i in index:
        shuffled_data.append(data[i])
    train_data = shuffled_data[0:train_len]
    val_data = shuffled_data[train_len:train_len + val_len]
    test_data = shuffled_data[train_len + val_len:]

    train_len = len(train_data)
    path_checkpoints = os.path.join(path_checkpoints, k_fold_str)
    batch_size = 2
    total_epochs = 60
    torch.manual_seed(seed)
    if gpu is not None:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    model = build_model(hidden_size=300,
                        mpnn_depth=3,
                        mpnn_diff_depth=3,
                        ffn_depth=3,
                        use_bias=True,
                        dropout=0.2,
                        task_num=1,
                        ffn_last_layer='no_softplus')

    logging.info('Model Sturture')
    logging.info(model)
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
          batch_size=batch_size,
          seed=seed,
          gpu=gpu,
          task_type='mle_regression',  # to choose the loss function
          writer=writer,
          log_path=log_path)

    print(path_checkpoints)
    score, score3, average_pred_in_targ = test(model, test_data, path_checkpoints, batch_size, gpu=gpu,
                                               log_path=log_path)
    test_score.append([score, score3])
print("test score for k_fold vailidation is: ", test_score)
logging.info('test score for k_fold vailidation is: {}'.format(test_score))






