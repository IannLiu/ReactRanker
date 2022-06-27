import torch
import pandas as pd
import numpy as np
from reactranker.data.load_reactions import DataProcessor


def test(model,
         test_data: pd.DataFrame,
         path_checkpoints,
         batch_size,
         gpu: int,
         logger,
         smiles2graph_dic,
         task_type: str = 'mle_gauss',
         model_type: str = 'multi_stage',
         show_info: bool = True,
         target_name: str = 'ea',
         smiles_list=None):
    """
    param task_type: str. Which is the task type of our object. Including: 'mle_gauss', 'ensembles', 'MC_dropout'
    param test_type: str. 'multi_stage', Comparing results by calculating multi-stage results
                          'single_stage', Comparing results by calculating single-stage results
    
    """
    logger.info('\n==========================================\n'
                '   Now, the test section is beginning!!!   \n'
                '==========================================')
    print('==========================================')
    print('  Now, the test section is beginning!!!   ')
    print('==========================================')
    logger.info('The path of checkpoints is:\n')
    logger.info(path_checkpoints)

    test_len = test_data.shape[0]
    print('the length of test data is:', test_len)
    logger.info('the length of test data is: {}'.format(test_len))

    # build and load model
    state = torch.load(path_checkpoints, map_location=lambda storage, loc: storage)
    means = state['data_scaler']['means']
    stds = state['data_scaler']['stds']
    loaded_state_dict = state['state_dict']
    if means is not None:
        if target_name != 'lgk':
            test_std_targ = test_data[target_name].map(lambda x: -(x - means) / stds)
        else:
            test_std_targ = test_data[target_name].map(lambda x: (x - means) / stds)
        test_data['std' + target_name] = test_std_targ
    model = model
    if gpu is not None:
        model = model.cuda()
    model.load_state_dict(loaded_state_dict)

    # evaluate
    if task_type == 'MC_dropout':
        model.train()
    else:
        model.eval()
    score = []
    score_pred_in_targ = []
    top1_in_pred = []
    ratio = 0.25
    pred_state = []
    target_state = []
    data_processor = DataProcessor(test_data)
    iter_counter = 0
    with torch.no_grad():
        for X, targets, scope in data_processor.generate_batch_querys(smiles_list=smiles_list,
                                                                      target_name='std' + target_name,
                                                                      batch_size=batch_size, shuffle_query=False,
                                                                      shuffle_batch=False):
            # X, targets are np.array object. scope is list object
            rsmi = [s[0] for s in X]
            psmi = [s[1] for s in X]
            r_batch_graph = smiles2graph_dic.parsing_smiles(rsmi)
            p_batch_graph = smiles2graph_dic.parsing_smiles(psmi)
            preds = model(r_batch_graph, p_batch_graph, gpu=gpu)
            iter_counter += 1

            if show_info is True:
                if iter_counter % 50 == 0:
                    print('the test targets is: ', targets)
                    print('the test outputs is: ', preds)
                    print('the scope is: ', scope)
            """
            Calcuating the criterion
            For mle_gaussian:
              For the ranking of n objetcs, ranking them n-1 times(stages). 
              For every stage, we have the expectation and variance
            """
            targets = torch.FloatTensor(targets).squeeze()
            preds = preds.cpu()
            idx_cri = 0
            for obj_num in scope:
                batch_targets = targets[idx_cri:idx_cri + obj_num]
                batch_preds = preds[idx_cri:idx_cri + obj_num]
                # update index
                idx_cri += obj_num
                if task_type == 'listnetdis_gauss':
                    # for listnet@1, pi=si/(s1+s2+...+sk). k is the number of all ranking items.
                    evidential_score = torch.sum(batch_preds[:, 0])
                    for selc_ith in range(obj_num):
                        ith_expect = batch_preds[:, 0][selc_ith] / evidential_score
                        ith_var = batch_preds[:, 1][selc_ith] / (evidential_score ** 2)
                        pred_state.append([selc_ith + 1, ith_expect.item(), ith_var.item()])
                elif task_type == 'listnetdis_evidential':
                    # For the first stage, calculating the varicane and mean for every selection.
                    evidential_score = torch.sum(batch_preds[:, 0])
                    for selc_ith in range(obj_num):
                        ith_expect = batch_preds[:, 0][selc_ith] / evidential_score
                        ith_var = (batch_preds[:, 1][selc_ith])
                        pred_state.append([selc_ith + 1, ith_expect.item(), ith_var.item()])
                else:
                    if model_type == 'listnet':
                        score_sum = torch.sum(torch.exp(batch_preds))
                        score = torch.exp(batch_preds)
                        for selc_ith in range(obj_num):
                            ith_score = score[selc_ith]  / score_sum
                            pred_state.append([selc_ith + 1, ith_score.item()])
                    else:
                        raise Exception('Unkonown model type')
                # calculating target state
                if model_type == 'listnet':
                    target_score = torch.exp(batch_targets)
                    targ_score_sum = torch.sum(target_score)
                    for selc_ith in range(obj_num):
                        ith_targte = target_score[selc_ith]  / targ_score_sum
                        target_state.append([selc_ith + 1, ith_targte.item()])
                else:
                    raise Exception('Unkonown model type')

            # Calculating the accuracy
            if len(preds.size()) > 1:
                preds = preds[:, 0].cpu().numpy().tolist()
            else:
                preds = preds.cpu().tolist()
            targets = np.squeeze(targets.numpy()).tolist()
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

    if show_info is True:
        print(score_pred_in_targ)
    average_score = sum(score) / len(score)
    average_pred_in_targ = sum(score_pred_in_targ) / len(score_pred_in_targ)
    average_top1_in_pred = sum(top1_in_pred) / len(top1_in_pred)
    if show_info is True:
        print('==========================================')
        print('   Note：For test set average score is:   ', average_score)
        print('   Note：Top{} pred in top{} targte is {}'.format(ratio, ratio, average_pred_in_targ))
        print('   Note：Top 1 pred in top{} targte is {}'.format(ratio, average_top1_in_pred))
        print('==========================================')
    logger.info('==========================================')
    logger.info('Top one acc is {}, Top{} pred in top{} targte is {}, Top 1 pred in top{} targte is {}'.format(
                average_score, ratio, ratio, average_pred_in_targ, ratio, average_pred_in_targ))
    return average_score, average_pred_in_targ, average_top1_in_pred, pred_state, target_state
