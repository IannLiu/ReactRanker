from collections import defaultdict

import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..data.load_reactions import get_time
from ..metrics import NDCG


def eval_cross_entropy_loss(model, device, loader, epoch, smiles2graph_dic, writer=None, phase="Eval", sigma=1.0):
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
        for X, target in loader.generate_batch_per_query(loader.df):
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


def eval_ndcg_at_k(
        inference_model, device, df_valid, valid_loader, batch_size, k_list, epoch, smiles2graph_dic,
        writer=None, phase="Eval"
):
    # print("Eval Phase evaluate NDCG @ {}".format(k_list))
    ndcg_metrics = {k: NDCG(k) for k in k_list}
    qids, rels, scores = [], [], []
    inference_model.eval()
    with torch.no_grad():
        for qid, rel, x in valid_loader.generate_query_batch(df_valid, batch_size):
            if x is None or x.shape[0] == 0:
                continue
            rsmi = [s[0] for s in x]
            psmi = [s[1] for s in x]
            r_batch = smiles2graph_dic.parsing_smiles(rsmi)
            p_batch = smiles2graph_dic.parsing_smiles(psmi)
            y_tensor = inference_model.forward(r_batch, p_batch)
            scores.append(y_tensor.cpu().numpy().squeeze())
            qids.append(qid)
            rels.append(rel)

    qids = np.hstack(qids)
    rels = np.hstack(rels)
    scores = np.hstack(scores)
    result_df = pd.DataFrame({'qid': qids, 'rel': rels, 'score': scores})
    session_ndcgs = defaultdict(list)
    for qid in result_df.qid.unique():
        result_qid = result_df[result_df.qid == qid].sort_values('score', ascending=False)
        rel_rank = result_qid.rel.values
        for k, ndcg in ndcg_metrics.items():
            if ndcg.maxDCG(rel_rank) == 0:
                continue
            ndcg_k = ndcg.evaluate(rel_rank)
            if not np.isnan(ndcg_k):
                session_ndcgs[k].append(ndcg_k)

    ndcg_result = {k: np.mean(session_ndcgs[k]) for k in k_list}
    ndcg_result_print = ", ".join(["NDCG@{}: {:.5f}".format(k, ndcg_result[k]) for k in k_list])
    print(get_time(), "{} Phase evaluate {}".format(phase, ndcg_result_print))
    if writer:
        for k in k_list:
            writer.add_scalars("metrics/NDCG@{}".format(k), {phase: ndcg_result[k]}, epoch)
    return ndcg_result


def evaluate_top_scores(model, gpu, data_processor, smiles2graph_dic,
                        ratio=0.25, batch_size=2, show_info=True):
    """
    Evaluating the top one, top 3, and top 25% hit ratio
    If s_i > s_j, P_si > P_sj.
    Therefore, rank the score in order, and the highest score is the most possible one.
    """
    # run model
    model.eval()
    with torch.no_grad():
        score = []
        score_pred_in_targ =[]
        top1_in_pred = []
        score3 = []
        ratio = ratio
        iter_counter = 0
        for X, targets, scope in data_processor.generate_batch_querys(batch_size=batch_size,
                                                                      shuffle_query=False,
                                                                      shuffle_batch=False):

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
                length = math.ceil(len(batch_preds) * ratio)
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
    #average_score3 = sum(score3) / len(score3)
    average_pred_in_targ = sum(score_pred_in_targ) / len(score_pred_in_targ)
    average_top1_in_pred = sum(top1_in_pred) / len(top1_in_pred)

    return average_score, average_pred_in_targ, average_top1_in_pred


def pairwise_acc(model, gpu, data_processor, smiles2graph_dic, batch_size=2, show_info=True):
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
        for X, targets in data_processor.generate_batch_per_query(shuffle_query=False,
                                                                  shuffle_batch=False):

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
            pred_rel_diff = pred_pos_pairs - rel_pos_pairs
            acc = torch.sum(torch.abs(pred_rel_diff), (0, 1))/(2 * np.sum(rel_pos_pairs, (0, 1)))
            accs.append(1-acc.item())

    return np.mean(accs)
