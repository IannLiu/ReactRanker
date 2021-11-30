from logging import Logger
import torch
import torch.nn as nn
from tqdm import trange
from pandas import DataFrame
from torch.optim.lr_scheduler import _LRScheduler
import copy
from torch.utils.tensorboard import SummaryWriter

from ..utils import save_checkpoint
from .train_pairwise import baseline_pairwise_training_loop, factorized_training_loop
from ..data.load_reactions import DataProcessor
from .eval import evaluate_top_scores, pairwise_acc


def run_train(model: nn.Module, scheduler:_LRScheduler, train_data_ini: DataFrame,
              val_data_ini: DataFrame, path_checkpoints: str, optimizer, epochs: int,
              smiles2graph_dic, batch_size: int, seed: int, gpu: int, train_strategy: str='baseline',
              task_type: str = 'baseline', writer = SummaryWriter, logger: Logger = None):
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

    train_data = copy.deepcopy(train_data_ini)
    val_data = copy.deepcopy(val_data_ini)
    mean = train_data['ea'].mean()
    std = train_data['ea'].std(ddof=0)
    train_std_targ = train_data['ea'].map(lambda x: -(x - mean) / std)
    val_std_trag = val_data['ea'].map(lambda x: -(x - mean) / std)
    train_data['std_targ'] = train_std_targ
    val_data['std_targ'] = val_std_trag
    print('stds is: ', std)
    print('mean is: ', mean)
    train_data_processor = DataProcessor(train_data)
    val_data_processor = DataProcessor(val_data)
    smiles2graph_dic = smiles2graph_dic
    score_old = float(0)
    loss_func = None
    if train_strategy == 'baseline':
        loss_func = torch.nn.BCELoss()
    for epoch in trange(epochs):
        # shuffle data
        train_data.sample(frac=1, random_state=seed)
        model.zero_grad()
        model.train()
        if train_strategy == 'baseline':
            epoch_loss = baseline_pairwise_training_loop(
                    epoch, model, loss_func, optimizer, scheduler,
                    smiles2graph_dic, train_data_processor, batch_size=batch_size, gpu=gpu)
        elif train_strategy in ['sum_session', 'accelerate_grad']:
            epoch_loss = factorized_training_loop(
                epoch, model, None, optimizer, scheduler,
                smiles2graph_dic, train_data_processor, batch_size=batch_size, sigma=1.0,
                training_algo=train_strategy, gpu=gpu)

        # evaluate the model
        # Should it be saved and reload????
        model.eval()
        with torch.no_grad():
            average_score, average_pred_in_targ, average_top1_in_pred = \
                evaluate_top_scores(model, gpu, val_data_processor, smiles2graph_dic, ratio=0.25, show_info=False)
            acc = pairwise_acc(model, gpu, val_data_processor, smiles2graph_dic,show_info=False)
        # save checkpoint if loss now < loss before
        if average_score >= score_old:
            score_old = average_score
            save_checkpoint(path_checkpoints, model, mean, std)
            print('Note: the checkpint file is updated')
        if writer is not None:
            writer.add_scalar('average_score', average_score, epoch)

        logger.info('Epoch [{}/{}],train_loss,{:.4f}, average_score_top1,{:.4f}, average_pred_in_targ_top25%,{:.4f}'\
                     .format(epoch + 1, epochs, epoch_loss, average_score, average_top1_in_pred))
        print('Epoch [{}/{}], accuracy: {:.4f}'.format(epoch + 1, epochs, acc))
        print('Epoch [{}/{}], average score: {:.4f}'.format(epoch + 1, epochs, average_score))
        print('Epoch [{}/{}], average_pred_in_targ_top25%: {:.4f}'.format(epoch + 1, epochs, average_pred_in_targ))
        print('Epoch [{}/{}], average_targtop1_in_predop25%: {:.4f}'.format(epoch + 1, epochs, average_top1_in_pred))
        print('Epoch [{}/{}], train loss: {:.4f}'.format(epoch + 1, epochs, epoch_loss))

