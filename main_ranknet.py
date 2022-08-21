"""
this scrip is a demo for test ranknet including:
loading data and perform ranknet
"""
import os
import logging
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from reactranker.data.load_reactions import get_data, Parsing_features
from reactranker.models.base_model import build_model
from reactranker.train.run_train_pairwise import run_train
from reactranker.train.test_ranknet import test

from reactranker.train.utils import build_optimizer, build_lr_scheduler
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
filter_size = user_defined

data = get_data(data_path)
data.read_data()
data.filter_bacth(filter_szie=filter_size)
# data.drop_columns(label_list=['rsmi', 'psmi','rsmi_mapped', 'psmi_mapped', 'ea'], task_type='keep')
k_fold = user_defined
gpu = user_defined  # cuda
test_score = []
train_strategy = 'sum_session'
smiles2graph_dic = Parsing_features()

batch_size = user_defined
total_epochs = user_defined
target_name = 'lgk'
smiles_list = ['rsmi_mapped', 'psmi_mapped']  # ['rsmi_mapped', 'psmi_mapped']
split_strategy = 'random_flag'  # 'scaffold' or 'random'
init_lr = user_defined
max_lr = user_defined
final_lr = user_defined
save_metric = 'all'
add_features_dim = 1
add_features_name = 'temp'

logger.info('Training data path is {}, and split strategy is {}'.format(data_path, split_strategy))
logger.info('Use the {} as input smiles'.format(smiles_list))
logger.info('filtered_size is {}'.format(filter_size))
logger.info('Task type is: {}, and target name is: {}'.format(train_strategy, target_name))
logger.info('{} fold train with {} epochs every fold. The batch size is: {}'.format(k_fold, total_epochs, batch_size))
logger.info('the initial, maximum, final learning rate are {}, {}, {}'.format(init_lr, max_lr, final_lr))
logger.info('the save metric is: {},'.format(save_metric))
logger.info('additional features is {} dim, and the name is {}'.format(add_features_dim, add_features_name))
logger.info('torch.mean(-torch.log(targets_possibility) + uncertainty_loss + penalty)')
logger.info('change the target possibility to negative, the score without softplus')

if save_metric == 'all':
    metric_list = ["T1", "T25_in_T25", "T25"]
    path = [os.path.join(path, i) for i in metric_list]
    if not os.path.exists(path[0]):
        os.makedirs(path[0])
        os.makedirs(path[1])
        os.makedirs(path[2])

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
    if save_metric != 'all':
        path_checkpoints = os.path.join(path_checkpoints, k_fold_str)
    else:
        path_checkpoints = [os.path.join(i, k_fold_str) for i in path_checkpoints]

    if val_data_path is not None and test_data_path is not None:
        train_data = pd.read_csv(data_path)
        val_data = pd.read_csv(val_data_path)
        test_data = pd.read_csv(test_data_path)
        print('this is a test flag', train_data.shape[0], val_data.shape[0], test_data.shape[0])
    else:
        if split_strategy == 'random':
            train_data, val_data, test_data = data.split_data(split_size=(0.8, 0.1, 0.1), split_type='reactants',
                                                              seed=seed)
        elif split_strategy == 'scaffold':
            train_data, val_data, test_data = data.scaffold_split_data(split_size=(0.8, 0.1, 0.1), balanced=True,
                                                                       seed=seed)
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
                        dropout=0.2,
                        task_num=1,
                        ffn_last_layer='no_softplus',
                        add_features_dim=add_features_dim)

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
              target_name=target_name,
              save_metric=save_metric,
              add_features_name=add_features_name)

    print(path_checkpoints)
    if save_metric == 'all':
        test_path = path_checkpoints[0]
    score, score3, average_pred_in_targ = test(model, test_data, test_path, smiles2graph_dic,
                                               batch_size, gpu=gpu, logger=logger,
                                               smiles_list=smiles_list, add_features_name=add_features_name,
                                               target_name=target_name, train_strategy=train_strategy)
    test_score.append([score, score3])
print("test score for k_fold vailidation is: ", test_score)
logger.info('test score for k_fold vailidation is: {}'.format(test_score))
