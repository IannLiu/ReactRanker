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

from .eval import evaluate_top_scores
from ..utils import save_checkpoint
from ..data.load_reactions import DataProcessor
from .loss import GaussDisLoss, evidential_loss_new, MLEloss, MLEDisLoss, Dirichlet_KL
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
          logger: Logger = None):
    print('Note: Set fixed seed')
    torch.manual_seed(seed)
    if gpu is not None:
        model = model.cuda()
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    train_data = copy.deepcopy(train_data_ini)
    val_data = copy.deepcopy(val_data_ini)

    score_old = float(0)
    train_len = train_data.shape[0]
    val_len = val_data.shape[0]
    data_len = train_len + val_len
    logger.info('Note: the length of training and vailidate data is: {}'.format(data_len))
    print("Note: the length of training and vailidate data is", data_len)

    mean = train_data['ea'].mean()
    std = train_data['ea'].std(ddof=0)
    train_std_targ = train_data['ea'].map(lambda x: -(x - mean) / std)
    val_std_trag = val_data['ea'].map(lambda x: -(x - mean) / std)
    train_data['std_targ'] = train_std_targ
    val_data['std_targ'] = val_std_trag
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
    elif task_type == 'mle_dirichlet':
        mle_loss = MLEloss()
        dirichlet_loss = Dirichlet_KL()
    elif task_type == 'dirichlet':
        dirichlet_loss = Dirichlet_KL()
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
        for reactions_train, targets_train, scope_train in \
                train_data_processor.generate_batch_querys(batch_size=batch_size, seed=epoch):
            """
            if i % 200 == 0:
                print('the reaction smiles is:',reactions_train)
                print('the target is:',targets_train)
                print('the scope is: ', scope_train)
            """
            targets_train = torch.FloatTensor(targets_train).squeeze()
            r_inputs_train, p_inputs_train = smiles2graph_dic.parsing_reactions(reactions_train)
            output = model(r_inputs_train, p_inputs_train, gpu=gpu)
            if np.any(np.isnan(model.state_dict()['encoder.W_i.weight'].cpu().tolist())):
                print('*' * 40)
                print('the mean of encoder.W_i.weight is: ', model.state_dict()['encoder.W_i.weight'])
                print('the ffn.ffn.7.weight is: ', model.state_dict()['ffn.ffn.7.weight'])
                print('*' * 40)
                print(targets_train)
            if task_type == 'mledis_gaussian':
                mu = output[:, [j for j in range(len(output[0])) if j % 2 == 0]]
                variance = output[:, [j for j in range(len(output[0])) if j % 2 == 1]]
                loss = mledis_loss(mu, variance, scope_train, targets_train, gpu) + gau_loss(output[:, 0], output[:, 1],
                                                                                             targets_train, gpu)
            elif task_type == 'mle_gaussian':
                loss = 2 * mle_loss(output[:, 0], scope_train, targets_train, gpu) + gau_loss(output[:, 0],
                                                                                              output[:, 1],
                                                                                              targets_train, gpu)
            elif task_type == 'evidential':
                mu = output[:, [j for j in range(len(output[0])) if j % 4 == 0]]
                lambdas = output[:, [j for j in range(len(output[0])) if j % 4 == 1]]
                alphas = output[:, [j for j in range(len(output[0])) if j % 4 == 2]]
                betas = output[:, [j for j in range(len(output[0])) if j % 4 == 3]]
                loss = evidential_loss_new(mu, lambdas, alphas, betas, targets_train, gpu, lam=0.2)
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
            elif task_type == 'dirichlet':
                loss = dirichlet_loss(output, scope_train, targets_train, gpu)
            else:
                loss = criterion(output, targets_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if writer is not None:
            writer.add_scalar('loss_every_epoch', loss.item())

        # evaluate the model
        model.eval()
        average_score, average_pred_in_targ, average_top1_in_pred = \
            evaluate_top_scores(model, gpu=gpu, data_processor=val_data_processor, \
                                smiles2graph_dic=smiles2graph_dic, ratio=0.25, show_info= False)

        # save checkpoint if loss now < loss before
        if average_score >= score_old:
            score_old = average_score
            save_checkpoint(path_checkpoints, model, mean, std)
            print('Note: the checkpint file is updated')
        if writer is not None:
            writer.add_scalar("average_score", average_score)

        logger.info(
            'Epoch [{}/{}], train_loss,{:.4f}, top1,{:.4f}, top1_in_pred_top25%,{:.4f}, pred_top25%_in_targ_top25%,{:.4f}'.format(
                epoch + 1, epochs, loss.item(), average_score, average_top1_in_pred, average_pred_in_targ,))
        print('Epoch [{}/{}], average score: {:.4f}'.format(epoch + 1, epochs, average_score))
        print('Epoch [{}/{}], targ_top1_in_pred_top25%: {:.4f}'.format(epoch + 1, epochs, average_top1_in_pred))
        print('Epoch [{}/{}], pred_top25%_in_targ_top25%: {:.4f}'.format(epoch + 1, epochs, average_pred_in_targ))
        print('Epoch [{}/{}], train loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))
