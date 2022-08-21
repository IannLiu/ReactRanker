from logging import Logger
import torch
import torch.nn as nn
from torch.nn import KLDivLoss
from tqdm import trange
from pandas import DataFrame
from torch.optim.lr_scheduler import _LRScheduler
import copy
from torch.utils.tensorboard import SummaryWriter

from ..utils import save_checkpoint
from .train_pairwise import baseline_pairwise_training_loop, factorized_training_loop, beta_dis_train_loop, \
    beta_evi_train_loop
from ..data.load_reactions import DataProcessor
from .eval import evaluate_top_scores, pairwise_acc, pairwise_baseline_acc


def run_train(model: nn.Module, scheduler: _LRScheduler, train_data_ini: DataFrame,
              val_data_ini: DataFrame, path_checkpoints: str, optimizer, epochs: int,
              smiles2graph_dic, batch_size: int, seed: int, gpu: int, train_strategy: str = 'baseline',
              task_type: str = 'baseline', writer=SummaryWriter, logger: Logger = None,
              smiles_list=None, target_name: str = 'ea', save_metric = None, add_features_name=None):
    """
    :param val_data_ini: DataFrame
    :param train_data_ini: DataFrame
    :param task_type: To choose the pairwise type
    :param train_strategy: 'baseline','sum_session','accelerate_grad'
    :param task_type: 'baseline', 'beta_distribution'
    :return:
    """
    if gpu is not None:
        torch.cuda.set_device(gpu)
    if task_type == 'BetaNet':
        loss_func = KLDivLoss(reduction='sum')

    train_data = copy.deepcopy(train_data_ini)
    val_data = copy.deepcopy(val_data_ini)
    mean = train_data[target_name].mean()
    std = train_data[target_name].std(ddof=0)
    if target_name != 'lgk':
        train_std_targ = train_data[target_name].map(lambda x: -(x - mean) / std)
        val_std_trag = val_data[target_name].map(lambda x: -(x - mean) / std)
    else:
        train_std_targ = train_data[target_name].map(lambda x: (x - mean) / std)
        val_std_trag = val_data[target_name].map(lambda x: (x - mean) / std)
    train_data['std' + target_name] = train_std_targ
    val_data['std' + target_name] = val_std_trag
    print('stds is: ', std)
    print('mean is: ', mean)
    train_data_processor = DataProcessor(train_data)
    val_data_processor = DataProcessor(val_data)
    smiles2graph_dic = smiles2graph_dic
    score_old = float(0)
    if save_metric == 'all':
        score_old = [0, 0, 0]
    loss_func = None
    if train_strategy == 'baseline':
        loss_func = torch.nn.BCELoss()
    for epoch in trange(epochs):
        print('learning rate: ', optimizer.state_dict()['param_groups'][0]['lr'])
        logger.info('learning rate is: {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        # shuffle data
        train_data.sample(frac=1, random_state=seed)
        model.zero_grad()
        model.train()
        if train_strategy == 'baseline' and task_type == 'baseline':
            epoch_loss = baseline_pairwise_training_loop(
                epoch, epochs, model, optimizer, scheduler, smiles2graph_dic, train_data_processor,
                batch_size=batch_size, max_coeff=0.001, gpu=gpu, target_name='std' + target_name)
        elif train_strategy in ['sum_session', 'accelerate_grad'] and task_type == 'baseline':
            epoch_loss = factorized_training_loop(
                epoch, model, None, optimizer, scheduler,
                smiles2graph_dic, train_data_processor, batch_size=batch_size, sigma=1.0,
                training_algo=train_strategy, gpu=gpu, smiles_list=smiles_list,
                target_name='std' + target_name, add_features_name=add_features_name)
        elif task_type == 'BetaNet':
            epoch_loss = beta_dis_train_loop(epoch, model, loss_func, optimizer, scheduler,
                                             smiles2graph_dic, train_data_processor, batch_size=2, alpha0=100,
                                             training_algo='beta_dis', gpu=gpu, smiles_list=smiles_list,
                                             target_name='std' + target_name)
        elif task_type == 'BetaNet_envidential':
            epoch_loss = beta_evi_train_loop(epoch, model, optimizer, scheduler,
                                             smiles2graph_dic, train_data_processor, batch_size=2, max_coeff=0.01,
                                             epochs=epochs,
                                             training_algo='beta_dis', gpu=gpu, smiles_list=smiles_list,
                                             target_name='std' + target_name)

        # evaluate the model
        # Should it be saved and reload????
        model.eval()
        with torch.no_grad():
            if train_strategy != 'baseline':
                average_score, average_pred_in_targ, average_top1_in_pred = \
                    evaluate_top_scores(model, gpu, val_data_processor, smiles2graph_dic,
                                        ratio=0.25, show_info=True, smiles_list=smiles_list,
                                        target_name='std' + target_name, add_features_name=add_features_name)
                # acc = pairwise_acc(model, gpu, val_data_processor, smiles2graph_dic,
                #                    show_info=False, smiles_list=smiles_list, target_name='std' + target_name)
            else:
                acc = pairwise_baseline_acc(model, gpu, val_data_processor, smiles2graph_dic, batch_size=500,
                                            show_info=True, smiles_list=None, target_name='std' + target_name)
        # save checkpoint if loss now < loss before
        if train_strategy != 'baseline':
            if save_metric == None or save_metric == 'average_score':
                if average_score >= score_old:
                    score_old = average_score
                    save_checkpoint(path_checkpoints, model, mean, std)
                    print('Note: the checkpint file is updated')
            elif save_metric == 'all':
                if average_score >= score_old[0]:
                    score_old[0] = average_score
                    save_checkpoint(path_checkpoints[0], model, mean, std)
                    print('Note: the checkpint file is updated')
                if average_pred_in_targ >= score_old[1]:
                    score_old[1] = average_pred_in_targ
                    save_checkpoint(path_checkpoints[1], model, mean, std)
                    print('Note: the checkpint file is updated')
                if average_top1_in_pred >= score_old[2]:
                    score_old[2] = average_top1_in_pred
                    save_checkpoint(path_checkpoints[2], model, mean, std)
                    print('Note: the checkpint file is updated')
        else:
            if acc >= score_old:
                score_old = acc
                save_checkpoint(path_checkpoints, model, mean, std)
            if writer is not None:
                writer.add_scalar('average_score: ', acc, epoch)


        if train_strategy == 'baseline':
            print('Epoch [{}/{}], accuracy: {:.4f}'.format(epoch + 1, epochs, acc))
            logger.info('Epoch [{}/{}],train_loss,{:.4f}, acc,{:.4f}'.format(epoch + 1, epochs, epoch_loss, acc))
        else:
            logger.info('Epoch [{}/{}],train_loss,{:.4f}, average_score_top1,{:.4f}, average_pred_in_targ_top25%,{:.4f}' \
                    .format(epoch + 1, epochs, epoch_loss, average_score, average_top1_in_pred))
#            print('Epoch [{}/{}], accuracy: {:.4f}'.format(epoch + 1, epochs, acc))
            print('Epoch [{}/{}], average score: {:.4f}'.format(epoch + 1, epochs, average_score))
            print('Epoch [{}/{}], average_pred_in_targ_top25%: {:.4f}'.format(epoch + 1, epochs, average_pred_in_targ))
            print('Epoch [{}/{}], average_targtop1_in_predtop25%: {:.4f}'.format(epoch + 1, epochs, average_top1_in_pred))
            print('Epoch [{}/{}], train loss: {:.4f}'.format(epoch + 1, epochs, epoch_loss))
