from collections import defaultdict

import math
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import time

# from .run_criterion import criterion
from ..data.load_reactions import get_time


def eval_cross_entropy_loss(model, device, loader, epoch, smiles2graph_dic,
                            writer=None, phase="Eval", sigma=1.0, smiles_list=None):
    """
    formula in https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf

    C = 0.5 * (1 - S_ij) * sigma * (si - sj) + log(1 + exp(-sigma * (si - sj)))
    when S_ij = 1:  C = log(1 + exp(-sigma(si - sj)))
    when S_ij = -1: C = log(1 + exp(-sigma(sj - si)))
    sigma can change the shape of the curve
    """
    print(get_time(), "{} Phase evaluate pairwise cross entropy loss".format(phase))
    model.eval()
    with torch.no_grad():
        total_cost = 0
        total_pairs = loader.get_num_pairs()
        pairs_in_compute = 0
        for X, target in loader.generate_batch_per_query(df=loader.df, smiles_list=smiles_list, ):
            target = target.reshape(-1, 1)
            rel_diff = target - target.T
            pos_pairs = (rel_diff > 0).astype(np.float32)
            num_pos_pairs = np.sum(pos_pairs, (0, 1))
            # skip negative sessions, no relevant info:
            if num_pos_pairs == 0:
                continue
            neg_pairs = (rel_diff < 0).astype(np.float32)
            num_pairs = 2 * num_pos_pairs  # num pos pairs and neg pairs are always the same
            pos_pairs = torch.tensor(pos_pairs, device=device)
            neg_pairs = torch.tensor(neg_pairs, device=device)
            s_ij = pos_pairs - neg_pairs
            # only calculate the different pairs
            diff_pairs = pos_pairs + neg_pairs
            pairs_in_compute += num_pairs

            rsmi = [s[0] for s in X]
            psmi = [s[1] for s in X]
            r_batch_graph = smiles2graph_dic.parsing_smiles(rsmi)
            p_batch_graph = smiles2graph_dic.parsing_smiles(psmi)
            y_pred = model(r_batch_graph,  p_batch_graph, gpu=device)
            y_pred_diff = y_pred - y_pred.t()

            # logsigmoid(x) = log(1 / (1 + exp(-x))) equivalent to log(1 + exp(-x))
            C = 0.5 * (1 - s_ij) * sigma * y_pred_diff - F.logsigmoid(-sigma * y_pred_diff)
            C = C * diff_pairs
            cost = torch.sum(C, (0, 1))
            if cost.item() == float('inf') or np.isnan(cost.item()):
                import ipdb; ipdb.set_trace()
            total_cost += cost

        assert total_pairs == pairs_in_compute
        avg_cost = total_cost / total_pairs
    print(
        get_time(),
        "Epoch {}: {} Phase pairwise corss entropy loss {:.6f}, total_paris {}".format(
            epoch, phase, avg_cost.item(), total_pairs
        ))
    if writer:
        writer.add_scalars('loss/cross_entropy', {phase: avg_cost.item()}, epoch)

    return avg_cost


def evaluate_top_scores(model, gpu, data_processor, smiles2graph_dic,
                        ratio=0.25, batch_size=2, show_info=False, smiles_list=None,
                        target_name: str = 'ea', add_features_name=None):
    """
    Evaluating the top one, top 3, and top 25% hit ratio
    If s_i > s_j, P_si > P_sj.
    Therefore, rank the score in order, and the highest score is the most possible one.
    """
    # run model
    # model.eval()
    with torch.no_grad():
        score = []
        score_pred_in_targ =[]
        top1_in_pred = []
        ratio = ratio
        iter_counter = 0
        for X, targets, scope, add_features in data_processor.generate_batch_querys(smiles_list=smiles_list,
                                                                                  target_name=target_name,
                                                                                  batch_size=batch_size,
                                                                                  shuffle_query=False,
                                                                                  shuffle_batch=False,
                                                                                  add_features_name=add_features_name):
            rsmi = [s[0] for s in X]
            psmi = [s[1] for s in X]
            r_batch_graph = smiles2graph_dic.parsing_smiles(rsmi)
            p_batch_graph = smiles2graph_dic.parsing_smiles(psmi)

            preds = model(r_batch_graph, p_batch_graph, gpu=gpu, add_features=add_features)
            iter_counter += 1
            idx0 = 0
            for item0 in scope:
                batch_targets0 = targets[idx0:idx0 + item0]
                batch_preds0 = preds[idx0:idx0 + item0]
                idx0 += item0
                if show_info == 'evidential_ranking':
                    if iter_counter % 50 == 0:
                        # total_score = torch.sum(batch_preds0, dim=1)
                        total_score = batch_preds0[:, 0]
                        print('the test targets is: ', np.exp(batch_targets0) / np.sum(np.exp(batch_targets0)))
                        print('the size is: ', item0)
                        print('the predicted score is: ', batch_preds0)
                        # print('the predicted belief is: ', batch_preds0[:, 0]/torch.sum(total_score))
                        print('the uncertainty is: ', batch_preds0[:, 1]/total_score)
                        print('the predicted possibility is: ', total_score / torch.sum(total_score))
                elif show_info:
                    if iter_counter % 50 == 0:
                        print('the test targets is: ', batch_targets)
                        print('the size is: ', item0)
                        print('the output is: ', batch_preds)
                else:
                    pass
            if len(preds.size()) > 1:
                preds = preds[:, 0].cpu().numpy()
            preds = preds.tolist()
            targets = np.squeeze(targets).tolist()
            idx = 0
            for item in scope:
                batch_targets = targets[idx:idx + item]
                batch_preds = preds[idx:idx + item]
                idx += item

                # top 1
                if batch_targets.index(max(batch_targets)) == batch_preds.index(max(batch_preds)):
                    score.append(1)
                else:
                    score.append(0)

                # top 25%
                sort_num1 = sorted(enumerate(batch_targets), key=lambda x: x[1], reverse=True)
                idx1 = [a[0] for a in sort_num1]
                sort_num2 = sorted(enumerate(batch_preds), key=lambda x: x[1], reverse=True)
                idx2 = [a[0] for a in sort_num2]
                num = 0
                length = round(len(batch_preds) * ratio)
                if length == 0:
                    length = 1
                for idx3 in range(length):
                    if idx2[idx3] in idx1[:length]:
                        num += 1
                percent = num / length
                score_pred_in_targ.append(percent)

                # Estimating top3 is not convenient, we can also make other criterion
                # For example, target top1 in top 25%
                if batch_targets.index(max(batch_targets)) in idx2[:length]:
                    top1_in_pred.append(1)
                else:
                    top1_in_pred.append(0)
                """
                # top 3
                if batch_targets.index(max(batch_targets)) in idx2[:3]:
                    score3.append(1)
                else:
                    score3.append(0)
                """

    average_score = sum(score) / len(score)
    # average_score3 = sum(score3) / len(score3)
    average_pred_in_targ = sum(score_pred_in_targ) / len(score_pred_in_targ)
    average_top1_in_pred = sum(top1_in_pred) / len(top1_in_pred)

    return average_score, average_pred_in_targ, average_top1_in_pred


def pairwise_acc(model, gpu, data_processor, smiles2graph_dic, batch_size=2, show_info=True,
                 smiles_list=None, target_name='ea'):
    """
    Evaluating the top one, top 3, and top 25% hit ratio
    If s_i > s_j, P_si > P_sj.
    Therefore, rank the score in order, and the highest score is the most possible one.
    """
    # run model
    model.eval()
    with torch.no_grad():
        iter_counter = 0
        accs = []
        for X, targets in data_processor.generate_batch_per_query(target_name=target_name,
                                                                  shuffle_query=False,
                                                                  shuffle_batch=False,
                                                                  smiles_list=smiles_list):

            rsmi = [s[0] for s in X]
            psmi = [s[1] for s in X]
            r_batch_graph = smiles2graph_dic.parsing_smiles(rsmi)
            p_batch_graph = smiles2graph_dic.parsing_smiles(psmi)
            preds = model(r_batch_graph, p_batch_graph, gpu=gpu)
            iter_counter += 1
            if show_info:
                if iter_counter % 50 == 0:
                    print('the test targets is: ', targets)
                    print('the test outputs is: ', preds)
            if len(preds.size()) > 1:
                preds = preds[:, 0]
            preds = preds.unsqueeze(1).cpu()
            pred_diff = preds - preds.t()
            pred_pos_pairs = (pred_diff > 0)
            targets = targets.reshape(-1, 1)
            rel_diff = targets - targets.T
            rel_pos_pairs = (rel_diff > 0).astype(np.float32)
            if np.sum(rel_pos_pairs, (0, 1)) == 0:
                continue
            pred_rel_diff = pred_pos_pairs - rel_pos_pairs
            acc = torch.sum(torch.abs(pred_rel_diff), (0, 1))/(2 * np.sum(rel_pos_pairs, (0, 1)))
            if acc<0:
                print('!!!!!!!!!!!!!!!!!!!!!!!')
                print(acc)
            accs.append(1-acc.item())

    return np.mean(accs)

def pairwise_baseline_acc(model, gpu, data_processor, smiles2graph_dic, batch_size=100, show_info=True,
                 smiles_list=None, target_name='ea'):
    """
    Evaluating the top one, top 3, and top 25% hit ratio
    If s_i > s_j, P_si > P_sj.
    Therefore, rank the score in order, and the highest score is the most possible one.
    """
    # run model
    model.eval()
    with torch.no_grad():
        iter_counter = 0
        accs = []
        for x_i, y_i, x_j, y_j in data_processor.generate_query_pair_batch(targ=target_name, batchsize=batch_size):
            if x_i is None or x_i.shape[0] == 0:
                continue
            iter_counter += 1
            rsmi1 = [s[0] for s in x_i]
            psmi1 = [s[1] for s in x_i]
            rsmi2 = [s[0] for s in x_j]
            psmi2 = [s[1] for s in x_j]
            # this function should be deleted if the user wanna compare reactions with different reactants
            assert rsmi1 == rsmi2
            if gpu is not None:
                t1 = torch.FloatTensor(y_i).cuda(gpu)
                t2 = torch.FloatTensor(y_j).cuda(gpu)
            else:
                t1 = torch.FloatTensor(y_i)
                t2 = torch.FloatTensor(y_j)
            target_alpha = torch.exp(torch.cat((t1, t2), dim=1))
            target_p = target_alpha / torch.sum(target_alpha, dim=1, keepdim=True)

            # BetaNet Output
            r = smiles2graph_dic.parsing_smiles(rsmi1)
            p_i = smiles2graph_dic.parsing_smiles(psmi1)
            p_j = smiles2graph_dic.parsing_smiles(psmi2)
            model.zero_grad()
            y_pred = model(r, p_i, p_j, gpu=gpu)
            pred_pos_pairs = (y_pred[:, 0] > y_pred[:, 1]).float()
            target_pos_pairs = (target_p[:, 0] > target_p[:, 1]).float()
            diff = pred_pos_pairs-target_pos_pairs
            acc = 1 - torch.sum(torch.abs(diff))/diff.size(0)
            print("accuracy is:", acc)
            accs.append(acc.item())
            if iter_counter % 20 == 0:
                print('the test targets is: ', target_p[:10])
                print('the test outputs is: ', y_pred[:10])

    return np.mean(accs)


def eval_regression_loss(model, gpu, data_processor, smiles2graph_dic, batch_size=50, show_info=True,
                         smiles_list=None, target_name='ea'):
    """
    Evaluating the top one, top 3, and top 25% hit ratio
    If s_i > s_j, P_si > P_sj.
    Therefore, rank the score in order, and the highest score is the most possible one.
    """
    # run model
    model.eval()
    with torch.no_grad():
        iter_counter = 0
        for X, targets in data_processor.generate_batch(target_name=target_name, shuffle_data=False,
                                                        batch_size=batch_size, smiles_list=smiles_list):

            rsmi = [s[0] for s in X]
            psmi = [s[1] for s in X]
            r_batch_graph = smiles2graph_dic.parsing_smiles(rsmi)
            p_batch_graph = smiles2graph_dic.parsing_smiles(psmi)
            preds = model(r_batch_graph, p_batch_graph, gpu=gpu)
            iter_counter += 1
            if show_info:
                if iter_counter % 5 == 0:
                    print('the test targets is: ', targets)
                    print('the test outputs is: ', preds)
            if len(preds.size()) > 1:
                means = preds[:, 0]

            means = means.cpu().numpy()
            loss = np.mean((means-targets)**2)

    return loss, preds


def cal_NDCG(preds, targets, n):
    """
    Calculate NDGC
    :param preds: The predicted score
    :param targets: The targets
    :param n: The nth items should be considered
    """
    if targets.size(0) > n:
        targets = targets[:n]
    IDCG_n = torch.sum(targets / torch.log2(torch.FloatTensor(range(targets.size(0))) + 2))

    if preds.size(0) > n:
        preds = preds[:n]
    DCG_n = torch.sum(preds / torch.log2(torch.FloatTensor(range(preds.size(0)))+2))

    NDCG_n = DCG_n/IDCG_n

    return NDCG_n


def calculate_ndcg(model, gpu, data_processor, smiles2graph_dic, batch_size=2, NDCG_cut=0.5,
                   show_info=False, smiles_list=None, target_name: str = 'ea', logger=None, is_order=True,
                   means=None, stds=None, add_features_name=None):
    """
    Evaluating the top one, top 3, and top 25% hit ratio
    If s_i > s_j, P_si > P_sj.
    Therefore, rank the score in order, and the highest score is the most possible one.
    """
    # run model
    # model.eval()
    with torch.no_grad():
        iter_counter = 0
        NDCG_list = []
        KL_list = []
        total_order = []  # collect the true order and prediction order
        smiles_and_idx = []
        for X, targets, scope, add_features in data_processor.generate_batch_querys(smiles_list=smiles_list,
                                                                      target_name=target_name,
                                                                      batch_size=batch_size,
                                                                      shuffle_query=False,
                                                                      shuffle_batch=False,
                                                                      add_features_name=add_features_name):

            rsmi = [s[0] for s in X]
            psmi = [s[1] for s in X]
            r_batch_graph = smiles2graph_dic.parsing_smiles(rsmi)
            p_batch_graph = smiles2graph_dic.parsing_smiles(psmi)
            preds_ini = model(r_batch_graph, p_batch_graph, gpu=gpu, add_features=add_features)
            iter_counter += 1
            if show_info:
                if iter_counter % 50 == 0:
                    print('the test targets is: ', targets)
                    print('the scope is: ', scope)
                    print('the prediction is: ', preds_ini)
                    """
                    if preds.dim() == 1:
                        print('the test outputs is: ', preds)
                    else:
                        if preds.size(1) == 2:
                            print('the test mean is: ', preds[:, 0])
                            print('the test variance is: ', torch.exp(preds[:, 1]))
                        else:
                            print('the output is: ', preds)
                    """
                    if logger is not None:
                        logger.info('the targets is {}'.format(targets))
                        logger.info('the scope is {}'.format(scope))
                        logger.info('the mean is {}'.format(preds_ini[:, 0]))
                        logger.info('the test variance is {}'.format(torch.exp(preds_ini[:, 1])))

            if means is not None:
                if len(preds_ini.size()) > 1:
                    m, s = torch.split(preds_ini, 1, dim=1)
                    size = preds_ini.size()
                    m1 = (m * stds) + means
                    s1 = s * (stds ** 2)
                    preds_ini = torch.stack((m1, s1), dim=2).view(size)
                else:
                    preds_ini = (preds_ini * stds) + means

            with_uncertainty = False
            if len(preds_ini.size()) > 1:
                with_uncertainty = True
                uncertainty = preds_ini[:, 1].cpu()
                preds = preds_ini[:, 0].cpu()
            else:
                preds = preds_ini.cpu()

            targets = torch.FloatTensor(np.squeeze(targets).tolist())
            count_per_iter = 0
            if is_order:
                for batch_targets, batch_preds in zip(targets.split(scope, dim=0), preds.split(scope, dim=0)):
                    # Calculate KL_divergency
                    P = torch.exp(batch_targets)/torch.sum(torch.exp(batch_targets))
                    Q = torch.exp(batch_preds)/torch.sum(torch.exp(batch_preds))
                    KL_div = torch.sum(P * torch.log(P/Q))
                    KL_list.append(KL_div.item())
                    sorted_targets, idx = torch.sort(batch_targets, descending=True)
                    sorted_preds = torch.gather(batch_preds, dim=0, index=idx)
                    if with_uncertainty:
                        sel_uncertainty = uncertainty[count_per_iter: count_per_iter + sorted_targets.size(0)]
                        sorted_uncertainty = torch.gather(sel_uncertainty, dim=0, index=idx)
                    # collect the pred_order and true order
                    pred_order = torch.argsort(torch.argsort(sorted_preds, descending=True)) + 1
                    true_order = torch.Tensor(range(sorted_targets.size(0))) + 1
                    if with_uncertainty:
                        order = torch.cat((sorted_targets.unsqueeze(dim=1), sorted_preds.unsqueeze(dim=1),
                                           sorted_uncertainty.unsqueeze(dim=1), true_order.unsqueeze(dim=1),
                                           pred_order.unsqueeze(dim=1)), dim=1)
                    else:
                        order = torch.cat((sorted_targets.unsqueeze(dim=1), sorted_preds.unsqueeze(dim=1),
                                           true_order.unsqueeze(dim=1), pred_order.unsqueeze(dim=1)), dim=1)
                    total_order.extend(order.tolist())
                    # calculate NDCG
                    length = sorted_targets.size(0)
                    pred_score = torch.FloatTensor([length+1]) - pred_order
                    true_score = torch.FloatTensor([length + 1]) - true_order
                    NDCG_n = cal_NDCG(pred_score, true_score, n=math.ceil(length * NDCG_cut))
                    NDCG_list.append(NDCG_n.item())
                    # Add smiles
                    rsmi_sel = rsmi[count_per_iter: count_per_iter+sorted_targets.size(0)]
                    psmi_sel = psmi[count_per_iter: count_per_iter+sorted_targets.size(0)]
                    rsmi_new = []
                    psmi_new = []
                    for i in idx.tolist():
                        rsmi_new.append(rsmi_sel[i])
                        psmi_new.append(psmi_sel[i])
                    count_per_iter += sorted_targets.size(0)
                    smi = [[iter_counter] + [a] + [b] for a, b in zip(rsmi_new, psmi_new)]
                    smiles_and_idx.extend(smi)
            else:
                # if not order, all of the NDCG and order can not be calculated
                # Just output the results
                NDCG_list = None
                KL_list = None
                if len(preds_ini.size()) > 1:
                    total_order.extend([[i] + j for i, j in zip(targets.tolist(), preds_ini.cpu().numpy().tolist())])  # Collect preds
                else:
                    total_order.extend([[i, j] for i, j in zip(targets.tolist(), preds_ini.cpu().numpy().tolist())])
                smi = [[a] + [b] for a, b in zip(rsmi, psmi)]
                smiles_and_idx.extend(smi)
                NDCG_mean = None
                KL_list = None

        if is_order:
            NDCG_mean = np.mean(np.array(NDCG_list), axis=0)
            KL_list = np.mean(np.array(KL_list))

    return NDCG_mean, KL_list, total_order, smiles_and_idx


def compute_NDCG(truth: list, pred: list):
    """
    Compute the NDCG

    :param truth: The ground truth. Ranking in descending order. list
    :param pred: the predicted items' score. Ranking in predicted order. list
    :return: NDCG
    """
    length = len(truth)
    DCG = np.sum(np.exp(pred) / np.log2(range(2, length + 2)))
    nDCG = np.sum(np.exp(truth) / np.log2(range(2, length + 2)))

    return DCG / nDCG


def ranking_metrics(model, gpu, data_processor, smiles2graph_dic,
                    show_info=True, smiles_list=None, target_name: str = 'ea', logger=None, add_features_name=None):
    """
    Evaluating the top one, top 25% hit ratio, recall@25%, NDCG@1, NDCG@2, NDCG@25%, NDCG@all
    """
    top1_counter = 0
    top25_couner = 0
    recall = []
    NDCG_list = []
    iter_counter = 0
    # run model
    model.eval()
    with torch.no_grad():
        for X, targets, add_features in data_processor.generate_batch_per_query(smiles_list=smiles_list,
                                                                                target_name=target_name,
                                                                                shuffle_query=False,
                                                                                shuffle_batch=False,
                                                                                add_features_name=add_features_name):
            iter_counter += 1

            rsmi = [s[0] for s in X]
            psmi = [s[1] for s in X]
            r_batch_graph = smiles2graph_dic.parsing_smiles(rsmi)
            p_batch_graph = smiles2graph_dic.parsing_smiles(psmi)
            preds_ini = model(r_batch_graph, p_batch_graph, gpu=gpu, add_features=add_features)
            if show_info:
                if iter_counter % 50 == 0:
                    print('the test targets is: ', targets)
                    print('the prediction is: ', preds_ini)
                    if logger is not None:
                        logger.info('the targets is {}'.format(targets))
                        logger.info('the mean is {}'.format(preds_ini[:, 0]))
                        logger.info('the test variance is {}'.format(torch.exp(preds_ini[:, 1])))

            if len(preds_ini.size()) > 1:
                pred_scores = preds_ini[:, 0].cpu().tolist()
            else:
                pred_scores = preds_ini.cpu().tolist()

            target_scores = targets.tolist()
            targ_length = len(target_scores)
            sorted_preds = sorted(enumerate(pred_scores), key=lambda x: x[1], reverse=True)
            sorted_targs = sorted(enumerate(target_scores), key=lambda x: x[1], reverse=True)
            pred_sort_idx = [sorted_preds[i][0] for i in range(0, targ_length)]
            targ_sort_idx = [sorted_targs[i][0] for i in range(0, targ_length)]
            targ_sort_scores = [sorted_targs[i][1] for i in range(0, targ_length)]

            # top 1 hit ?
            if pred_sort_idx[0] == targ_sort_idx[0]:
                top1_counter += 1

            # top 25% hit ?
            len25 = round(targ_length * 0.25)
            if len25 < 1:
                len25 = 1
            pred_top25_idx = pred_sort_idx[:len25]
            targ_top25_idx = targ_sort_idx[:len25]
            if pred_top25_idx[0] in targ_top25_idx:
                top25_couner += 1

            # top 25% recall
            num = 0
            for idx in pred_top25_idx:
                if idx in targ_top25_idx:
                    num += 1
            recall.append(num/len25)

            # Calculate NDCG
            pred_rank_to_targ_score = [target_scores[i] for i in pred_sort_idx]
            NDCG1 = compute_NDCG([targ_sort_scores[0]], [pred_rank_to_targ_score[0]])
            NDCG2 = compute_NDCG([targ_sort_scores[:2]], [pred_rank_to_targ_score[:2]])
            NDCG25 = compute_NDCG(targ_sort_scores[:len25], pred_rank_to_targ_score[:len25])
            NDCG_all = compute_NDCG(targ_sort_scores, pred_rank_to_targ_score)
            NDCG_list.append([NDCG1, NDCG2, NDCG25, NDCG_all])

    top1 = top1_counter / iter_counter
    top25 = top25_couner / iter_counter
    recall25 = float(np.mean(recall))
    NDCG_ = np.mean(NDCG_list, axis=0)

    return top1, recall25, top25, NDCG_


def calculate_mse(model, gpu, data_processor, smiles2graph_dic, batch_size=2,
                  show_info=True, smiles_list=None, target_name: str = 'ea', logger=None):
    """
    Evaluating the top one, top 3, and top 25% hit ratio
    If s_i > s_j, P_si > P_sj.
    Therefore, rank the score in order, and the highest score is the most possible one.
    """
    # run model
    model.eval()
    with torch.no_grad():
        iter_counter = 0
        for X, targets, scope in data_processor.generate_batch_querys(smiles_list=smiles_list,
                                                                      target_name=target_name,
                                                                      batch_size=batch_size,
                                                                      shuffle_query=False,
                                                                      shuffle_batch=False):

            rsmi = [s[0] for s in X]
            psmi = [s[1] for s in X]
            r_batch_graph = smiles2graph_dic.parsing_smiles(rsmi)
            p_batch_graph = smiles2graph_dic.parsing_smiles(psmi)
            preds_ini = model(r_batch_graph, p_batch_graph, gpu=gpu)
            targets = torch.FloatTensor(np.squeeze(targets).tolist())
            iter_counter += 1
            if show_info:
                if iter_counter % 50 == 0:
                    print('the test targets is: ', targets)
                    print('the scope is: ', scope)
                    print('the prediction is: ', preds_ini)
                    """
                    if preds.dim() == 1:
                        print('the test outputs is: ', preds)
                    else:
                        if preds.size(1) == 2:
                            print('the test mean is: ', preds[:, 0])
                            print('the test variance is: ', torch.exp(preds[:, 1]))
                        else:
                            print('the output is: ', preds)
                    """
                    if logger is not None:
                        logger.info('the targets is {}'.format(targets))
                        logger.info('the scope is {}'.format(scope))
                        logger.info('the mean is {}'.format(preds_ini[:, 0]))
                        logger.info('the test variance is {}'.format(torch.exp(preds_ini[:, 1])))

            if len(preds_ini.size()) > 1:
                preds = preds_ini[:, 0].cpu()
            else:
                preds = preds_ini.cpu()
            MSE = torch.mean(torch.pow((targets-preds), 2))

    return MSE.item()
