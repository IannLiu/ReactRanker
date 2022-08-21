from pandas import DataFrame
import numpy as np
import math
from logging import Logger
from typing import Union

import torch
import torch.nn as nn
from tqdm import trange
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from .eval import evaluate_top_scores, calculate_ndcg, calculate_mse, ranking_metrics
from ..utils import save_checkpoint
from ..data.load_reactions import DataProcessor
from .loss import GaussDisLoss, evidential_loss_new, MLEloss, MLEDisLoss, Dirichlet_uq, Listnet_For_evidential, \
    Listnet_For_Gauss, ListnetLoss, Listnet_with_uq, Listnetlognorm, Lognorm, evidential_ranking
import copy


def train(model: nn.Module,
          scheduler: _LRScheduler,
          train_data_ini: DataFrame,
          val_data_ini: DataFrame,
          path_checkpoints: str,
          optimizer,
          epochs: int,
          smiles2graph_dic,
          batch_size: int,
          seed: int,
          gpu: Union[int, str],
          task_type: str = 'mle_gaussian',
          writer=SummaryWriter,
          logger: Logger = None,
          target_name: str = 'ea',
          smiles_list: list = None,
          save_metric=None,
          show_info=False,
          max_coeff=0.0001,
          normalize_target=True,
          add_features_name=None):
    print('Note: Set fixed seed')
    torch.manual_seed(seed)
    if gpu is not None:
        model = model.cuda()
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    train_data = copy.deepcopy(train_data_ini)
    val_data = copy.deepcopy(val_data_ini)
    # calculating annealing_step and global_step for listnet with uq
    nbatch = int(len(train_data.rsmi.unique()) / batch_size)
    if save_metric == 'mse':
        score_old = float('inf')
    else:
        score_old = float(0)
        if save_metric == 'all':
            score_old = [0, 0, 0]
    train_len = train_data.shape[0]
    val_len = val_data.shape[0]
    data_len = train_len + val_len
    logger.info('Note: the length of training and vailidate data is: {}'.format(data_len))
    print("Note: the length of training and vailidate data is", data_len)

    mean = train_data[target_name].mean()
    std = train_data[target_name].std(ddof=0)
    if target_name != 'lgk' and target_name != 'lgk_bi':
        if isinstance(normalize_target, float):
            max_num = train_data[target_name].max()
            min_num = train_data[target_name].min()
            train_std_targ = train_data[target_name].map(lambda x: -(x * normalize_target) / (max_num - min_num))
            val_std_targ = val_data[target_name].map(lambda x: -(x * normalize_target) / (max_num - min_num))

        elif isinstance(normalize_target, str):
            max_num = train_data[target_name].max()
            min_num = train_data[target_name].min()
            scale = normalize_target.split(',')
            low_bond = int(scale[0])
            high_bond = int(scale[1])
            train_std_targ = train_data[target_name].map(
                lambda x: -(x - min_num) * (high_bond - low_bond) / (max_num - min_num) + low_bond)
            val_std_targ = val_data[target_name].map(
                lambda x: -(x - min_num) * (high_bond - low_bond) / (max_num - min_num) + low_bond)
        elif normalize_target:
            train_std_targ = train_data[target_name].map(lambda x: -(x - mean) / std)
            val_std_targ = val_data[target_name].map(lambda x: -(x - mean) / std)
        else:
            train_std_targ = - train_data[target_name]
            val_std_targ = - val_data[target_name]
    elif target_name == 'lgk_bi':
        train_std_targ = train_data[target_name]
        val_std_targ = val_data[target_name]
    else:
        if isinstance(normalize_target, float):
            max_num = train_data[target_name].max()
            min_num = train_data[target_name].min()
            train_std_targ = train_data[target_name].map(lambda x: (x * normalize_target) / (max_num - min_num))
            val_std_targ = val_data[target_name].map(lambda x: (x * normalize_target) / (max_num - min_num))
        elif isinstance(normalize_target, str):
            max_num = train_data[target_name].max()
            min_num = train_data[target_name].min()
            scale = normalize_target.split(',')
            low_bond = int(scale[0])
            high_bond = int(scale[1])
            train_std_targ = train_data[target_name].map(
                lambda x: (x - min_num) * (high_bond - low_bond) / (max_num - min_num) + low_bond)
            val_std_targ = val_data[target_name].map(
                lambda x: (x - min_num) * (high_bond - low_bond) / (max_num - min_num) + low_bond)
        elif normalize_target:
            train_std_targ = train_data[target_name].map(lambda x: (x - mean) / std)
            val_std_targ = val_data[target_name].map(lambda x: (x - mean) / std)
        else:
            train_std_targ = train_data[target_name]
            val_std_targ = val_data[target_name]

    train_data['std' + target_name] = train_std_targ
    NDCG_metric_list = ['NDCG@1', 'NDCG@2', 'NDCG@25%', 'NDCG@all']
    if save_metric in NDCG_metric_list:
        val_data['std' + target_name] = val_data[target_name]
    else:
        val_data['std' + target_name] = val_std_targ

    print('stds is: ', std)
    print('mean is: ', mean)
    # loss instantiation
    if task_type == 'mle_gaussian':
        mle_loss = MLEloss()
        gau_loss = GaussDisLoss()
    elif task_type == 'mledis_gaussian':
        mledis_loss = MLEDisLoss()
        gau_loss = GaussDisLoss()
    elif task_type == 'mle_regression':
        mle_loss = MLEloss()
        reg_loss = nn.MSELoss()
    elif task_type == 'mle' or task_type == 'mle_evidential':
        mle_loss = MLEloss()
    elif task_type == 'mledis_evidential':
        mledis_loss = MLEDisLoss()
    elif task_type == 'listnet':
        listnet_loss = ListnetLoss()
    elif task_type == 'listnet_uq':
        listnet_loss = Listnet_with_uq()
    elif task_type == 'listnet_evidential':
        listnet_loss = Listnet_For_Gauss()
    elif task_type == 'listnet_gauss':
        listnet_loss = ListnetLoss()
        gau_loss = GaussDisLoss()
    elif task_type == 'listnetdis_gauss':
        listnet_loss = Listnet_For_Gauss()
        gau_loss = GaussDisLoss()
    elif task_type == 'listnetdis_lognorm':
        listnet_loss = Listnetlognorm()
        gau_loss = Lognorm()
    elif task_type == 'dirichlet_uq':
        dirichlet_loss = Dirichlet_uq()
    elif task_type == 'gauss_regression':
        gau_loss = GaussDisLoss()
    elif task_type == 'listnet_regression':
        listnet_loss = ListnetLoss()
        criterion = nn.MSELoss()
    elif task_type == 'regression_exploss':
        pass
    elif task_type == 'evidential_ranking':
        eviden_loss = evidential_ranking()
    else:  # default regression
        criterion = nn.MSELoss()

    train_data_processor = DataProcessor(train_data)
    val_data_processor = DataProcessor(val_data)
    smiles2graph_dic = smiles2graph_dic
    for epoch in trange(epochs):
        print('learning rate: ', optimizer.state_dict()['param_groups'][0]['lr'])
        logger.info('learning rate is: {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        model.train()
        loss = torch.FloatTensor([0])
        for reactions_train, targets_train, scope_train, add_features in \
                train_data_processor.generate_batch_reactions(smiles_list=smiles_list, target_name='std' + target_name,
                                                           batch_size=batch_size, seed=epoch,
                                                           add_features_name=add_features_name):
            """
            if i % 200 == 0:
                print('the reaction smiles is:',reactions_train)
                print('the target is:',targets_train)
                print('the scope is: ', scope_train)
            """
            targets_train = torch.FloatTensor(targets_train).squeeze()
            r_inputs_train, p_inputs_train = smiles2graph_dic.parsing_reactions(reactions_train)
            output = model(r_inputs_train, p_inputs_train, gpu=gpu, add_features=add_features)
            if np.any(np.isnan(model.state_dict()['encoder.W_i.weight'].cpu().tolist())):
                print('*' * 40)
                print('the mean of encoder.W_i.weight is: ', model.state_dict()['encoder.W_i.weight'])
                print('the ffn.ffn.7.weight is: ', model.state_dict()['ffn.ffn.7.weight'])
                print('*' * 40)
                print(targets_train)
            if task_type == 'mledis_gaussian':
                # To stabilize the variance, the output variance is log(variance), so transform them
                mu = output[:, [j for j in range(len(output[0])) if j % 2 == 0]]
                variance = torch.exp(output[:, [j for j in range(len(output[0])) if j % 2 == 1]])
                loss = mledis_loss(mu, variance, scope_train, targets_train, gpu) + gau_loss(output[:, 0],
                                                                                             output[:, 1],
                                                                                             targets_train, gpu)
            elif task_type == 'mle_gaussian':
                loss = mle_loss(output[:, 0], scope_train, targets_train, gpu) + gau_loss(output[:, 0],
                                                                                          output[:, 1],
                                                                                          targets_train, gpu)
            elif task_type == 'listnet_gauss':
                loss = listnet_loss(output[:, 0], scope_train, targets_train, gpu) + \
                       gau_loss(output[:, 0], output[:, 1], targets_train, gpu)
            elif task_type == 'listnetdis_gauss':
                mu = output[:, [j for j in range(len(output[0])) if j % 2 == 0]]
                variance = output[:, [j for j in range(len(output[0])) if j % 2 == 1]]
                loss = listnet_loss(mu, variance, scope_train, targets_train, gpu) + \
                       gau_loss(output[:, 0], output[:, 1], targets_train, gpu)
            elif task_type == 'listnetdis_lognorm':
                mu = output[:, [j for j in range(len(output[0])) if j % 2 == 0]]
                variance = output[:, [j for j in range(len(output[0])) if j % 2 == 1]]
                loss = gau_loss(output[:, 0], output[:, 1], targets_train, gpu)
                # listnet_loss(mu, variance, scope_train, targets_train, gpu) + \

            elif task_type == 'listnet':
                loss = listnet_loss(output, scope_train, targets_train, gpu)
            elif task_type == 'listnet_regression':
                if gpu is not None:
                    targets_train = targets_train.cuda(gpu)
                loss = listnet_loss(output, scope_train, targets_train, gpu) + criterion(output, targets_train)
            elif task_type == 'listnet_uq':
                loss = listnet_loss(output, scope_train, targets_train, max_coeff, epoch, epochs, gpu)
            elif task_type == 'evidential':
                mu = output[:, [j for j in range(len(output[0])) if j % 4 == 0]]
                lambdas = output[:, [j for j in range(len(output[0])) if j % 4 == 1]]
                alphas = output[:, [j for j in range(len(output[0])) if j % 4 == 2]]
                betas = output[:, [j for j in range(len(output[0])) if j % 4 == 3]]
                loss = evidential_loss_new(mu, lambdas, alphas, betas, targets_train, gpu, lam=0.1)
            elif task_type == 'mledis_evidential':
                mu = output[:, [j for j in range(len(output[0])) if j % 4 == 0]]
                lambdas = output[:, [j for j in range(len(output[0])) if j % 4 == 1]]
                alphas = output[:, [j for j in range(len(output[0])) if j % 4 == 2]]
                betas = output[:, [j for j in range(len(output[0])) if j % 4 == 3]]
                variance = betas / (lambdas * (alphas - 1))
                loss = mledis_loss(mu, variance, scope_train, targets_train, gpu) + evidential_loss_new(mu, lambdas,
                                                                                                        alphas, betas,
                                                                                                        targets_train,
                                                                                                        gpu, lam=0.1)
            elif task_type == 'listnet_evidential':
                mu = output[:, [j for j in range(len(output[0])) if j % 4 == 0]]
                lambdas = output[:, [j for j in range(len(output[0])) if j % 4 == 1]]
                alphas = output[:, [j for j in range(len(output[0])) if j % 4 == 2]]
                betas = output[:, [j for j in range(len(output[0])) if j % 4 == 3]]
                variance = betas / (lambdas * (alphas - 1))
                loss = listnet_loss(mu, variance, scope_train, targets_train, gpu) + \
                       evidential_loss_new(mu, lambdas, alphas, betas, targets_train, gpu, lam=0.1)
            elif task_type == 'mle_evidential':
                mu = output[:, [j for j in range(len(output[0])) if j % 4 == 0]]
                lambdas = output[:, [j for j in range(len(output[0])) if j % 4 == 1]]
                alphas = output[:, [j for j in range(len(output[0])) if j % 4 == 2]]
                betas = output[:, [j for j in range(len(output[0])) if j % 4 == 3]]
                variance = betas / (lambdas * (alphas - 1))
                loss = mle_loss(output[:, 0], scope_train, targets_train, gpu) + \
                       evidential_loss_new(mu, lambdas, alphas, betas, targets_train, gpu, lam=0.2)
            elif task_type == 'mle_regression':
                if gpu is not None:
                    targets_train = targets_train.cuda(gpu)
                loss = reg_loss(output, targets_train) + mle_loss(output, scope_train, targets_train, gpu)
            elif task_type == 'mle':
                loss = mle_loss(output, scope_train, targets_train, gpu)
            elif task_type == 'mle_dirichlet':
                loss = mle_loss(output, scope_train, targets_train, gpu) + \
                       dirichlet_loss(output, scope_train, targets_train, gpu)
            elif task_type == 'dirichlet_uq':
                loss = dirichlet_loss(output, scope_train, targets_train, max_coeff, epoch, epochs, gpu)
            elif task_type == 'gauss_regression':
                loss = gau_loss(output[:, 0], output[:, 1], targets_train, gpu)
            elif task_type == 'regression_exploss':
                if gpu is not None:
                    targets_train = targets_train.cuda(gpu)
                targets = torch.exp(targets_train)
                preds = torch.exp(output)
                loss = torch.mean((targets - preds) ** 2)
            elif task_type == 'evidential_ranking':
                loss = eviden_loss(output, scope_train, targets_train, max_coeff, epoch, epochs, gpu)
            else:
                if gpu is not None:
                    targets_train = targets_train.cuda(gpu)
                loss = criterion(output, targets_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if writer is not None:
            writer.add_scalar('loss_every_epoch', loss.item())

        # evaluate the model
        model.eval()
        """
        average_score, average_pred_in_targ, average_top1_in_pred = \
            evaluate_top_scores(model, gpu=gpu, data_processor=val_data_processor,
                                smiles2graph_dic=smiles2graph_dic, ratio=0.25, show_info=show_info,
                                # Note : for evidential ranking, we set show_info = task_type
                                smiles_list=smiles_list, target_name='std' + target_name)
        """

        average_score, average_pred_in_targ, average_top1_in_pred, NDCG_ = \
            ranking_metrics(model, gpu=gpu, data_processor=val_data_processor, smiles2graph_dic=smiles2graph_dic,
                            show_info=show_info, smiles_list=smiles_list, target_name='std' + target_name,
                            add_features_name=add_features_name)

        # save checkpoint if loss now < loss before
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
        elif save_metric == 'average_pred_in_targ':
            if average_pred_in_targ >= score_old:
                score_old = average_pred_in_targ
                save_checkpoint(path_checkpoints, model, mean, std)
                print('Note: the checkpint file is updated')
        elif save_metric == 'average_top1_in_pred':
            if average_top1_in_pred >= score_old:
                score_old = average_top1_in_pred
                save_checkpoint(path_checkpoints, model, mean, std)
                print('Note: the checkpint file is updated')
        elif save_metric in NDCG_metric_list:
            score_new = NDCG_[NDCG_metric_list.index(save_metric)]
            if score_new >= score_old:
                score_old = score_new
                save_checkpoint(path_checkpoints, model, mean, std)
                print('Note: the checkpint file is updated')
        elif save_metric == 'mse':
            mse = calculate_mse(model, gpu=gpu, data_processor=val_data_processor, smiles2graph_dic=smiles2graph_dic,
                                batch_size=batch_size, smiles_list=smiles_list, target_name='std' + target_name,
                                show_info=show_info)
            if mse <= score_old:
                score_old = mse
                save_checkpoint(path_checkpoints, model, mean, std)
            print('Note: the checkpint file is updated')
        else:
            raise Exception('Unknown save metric')

        if writer is not None:
            writer.add_scalar("average_score", average_score)

        logger.info(
            'Epoch [{}/{}], train_loss,{:.4f}, top1,{:.4f}, top1_in_pred_top25%,{:.4f}, pred_top25%_in_targ_top25%,{:.4f}'.format(
                epoch + 1, epochs, loss.item(), average_score, average_top1_in_pred, average_pred_in_targ, ))
        print('Epoch [{}/{}], average score: {:.4f}'.format(epoch + 1, epochs, average_score))
        print('Epoch [{}/{}], targ_top1_in_pred_top25%: {:.4f}'.format(epoch + 1, epochs, average_top1_in_pred))
        print('Epoch [{}/{}], pred_top25%_in_targ_top25%: {:.4f}'.format(epoch + 1, epochs, average_pred_in_targ))
        print('Epoch [{}/{}], train loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))
        if save_metric in ['NDCG@1', 'NDCG@2', 'NDCG@25%', 'NDCG@all']:
            logger.info(
                'Epoch [{}/{}], NDCG,{}'.format(epoch + 1, epochs, NDCG_))
            print('Epoch [{}/{}], NDCG: {}'.format(epoch + 1, epochs, NDCG_))
        if save_metric == 'mse':
            logger.info(
                'Epoch [{}/{}], mse loss,{:.4f}'.format(epoch + 1, epochs, score_old))
            print('Epoch [{}/{}], mse loss: {:.4f}'.format(epoch + 1, epochs, score_old))
