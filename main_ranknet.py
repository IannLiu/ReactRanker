"""
this scrip is a demo for test ranknet including:
loading data and perform ranknet
"""
import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter

from reactranker.data.load_reactions import get_data, Parsing_features
from reactranker.models.base_model import build_model
from reactranker.train.run_train_pairwise import run_train
from reactranker.train.test_ranknet import test
from reactranker.train.utils import build_optimizer, build_lr_scheduler

path = r'C:\Users\5019\Desktop\ReactionRanker\model_results\picture_data\differernt_size_trans_score\b97d3_ranknet_300_ini_lr1e-5'
data_path = 'C:\\Users\\5019\\Desktop\\data\\different_data_size\\b97d3_with_rate_300.csv'
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
gpu = 1  # cuda
test_score = []
smiles2graph_dic = Parsing_features()

# training parameter
k_fold = 1
batch_size = 2
total_epochs = 100
task_type = 'baseline'  # to choose loss function
train_strategy = "sum_session"  # 'sum_session'
target_name = 'lgk'
smiles_list = None  # ['rsmi_mapped', 'psmi_mapped']
init_lr = 0.0001
max_lr = 0.001
final_lr = 0.00001

logger.info('Training data path is {}'.format(data_path))
logger.info('Use the {} as input smiles'.format(smiles_list))
logger.info('filtered_size is {}'.format(filtered_size))
logger.info('Task type is: {}, and target name is: {}'.format(task_type, target_name))
logger.info('For pairwise, the train strategy is: {}'.format(train_strategy))
logger.info('{} fold train with {} epochs every fold. The batch size is: {}'.format(k_fold, total_epochs, batch_size))
logger.info('the initial, maximum, final learning rate are {}, {}, {}'.format(init_lr, max_lr, final_lr))

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
    train_data, val_data, test_data = data.split_data(seed=seed)
    path_checkpoints = os.path.join(path_checkpoints, k_fold_str)
    train_len = len(train_data.rsmi.unique())
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
                                   init_lr=init_lr,
                                   max_lr=max_lr,
                                   final_lr=final_lr)

    run_train(model,
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
              train_strategy=train_strategy,
              task_type='baseline',  # to choose the loss function
              writer=writer,
              logger=logger,
              smiles_list=smiles_list,
              target_name=target_name)

    print(path_checkpoints)
    average_score, average_pred_in_targ, average_top1_in_pred\
        = test(model, test_data, path_checkpoints, smiles2graph_dic, batch_size, gpu=gpu, logger=logger,
               smiles_list=smiles_list, target_name=target_name, train_strategy=train_strategy)
    test_score.append([average_score, average_pred_in_targ, average_top1_in_pred])
print("test score for k_fold vailidation is: ", test_score)
logger.info('test score for k_fold vailidation is: {}'.format(test_score))
