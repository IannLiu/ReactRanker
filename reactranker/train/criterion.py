import torch
import pandas as pd
import numpy as np
from reactranker.data.load_reactions import DataProcessor
import copy


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
         smiles_list=None,
         saved_data_path=None):
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
            test_std_targ = -test_data[target_name]  # .map(lambda x: -(x - means) / stds)
        else:
            test_std_targ = test_data[target_name]  # .map(lambda x: (x - means) / stds)
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
    data_collector = []
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
            
            # save the predicted and normlized data
            # This section is unecessary, just needed for data analysis
            # =================start=========================
            if means is not None:
                if len(preds.size()) > 1:
                    m, s = torch.split(preds, 1, dim=1)
                    size = preds.size()
                    m1 = (m * stds) + means
                    s1 = s * stds ** 2
                    preds = torch.stack((m1, s1), dim=2).view(size)
                else:
                    preds = (preds * stds) + means

            pred_data = copy.deepcopy(preds.cpu())
            if len(preds.size()) > 1:
                pass
            else:
                pred_data = pred_data.unsqueeze(1)
            pred_data = pred_data.tolist()
            targ_data = copy.deepcopy(targets.tolist())
            rsmi_for_save = [[s[0]] for s in X]
            psmi_for_save = [[s[1]] for s in X]
            for x, y, z, w in zip(rsmi_for_save, psmi_for_save, targ_data, pred_data):
                data_collector.append(x+y+z+w)

            # ===================end=========================

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
                batch_targets_nosort = targets[idx_cri:idx_cri + obj_num]
                batch_preds_nosort = preds[idx_cri:idx_cri + obj_num]
                # update index
                idx_cri += obj_num
                # sorting them according to targets
                batch_targets, sorted_targ_idx = torch.sort(batch_targets_nosort, descending=True)
                batch_preds = torch.index_select(batch_preds_nosort, dim=0, index=sorted_targ_idx)

                if task_type == 'mledis_gauss':
                    # calculating the expectation and variance for every stage.
                    for stage in range(obj_num - 1):
                        score_stage = batch_preds[:, 0][stage:] - batch_preds[:, 0][stage]
                        var_stage = batch_preds[:, 1][stage:] + batch_preds[:, 1][stage]
                        expect = torch.sum(torch.exp(score_stage + var_stage / 2))
                        var = torch.sum((torch.exp(var_stage) - 1) * torch.exp(2 * score_stage + var_stage))
                        pred_state.append([stage + 1, expect.item(), var.item()])
                elif task_type == 'listnetdis_gauss':
                    # for listnet@1, pi=si/(s1+s2+...+sk). k is the number of all ranking items.
                    for selc_ith in range(obj_num):
                        score_stage = batch_preds[:, 0] - batch_preds[:, 0][selc_ith]
                        var_stage = batch_preds[:, 1] + batch_preds[:, 1][selc_ith]
                        ith_expect = torch.sum(torch.exp(score_stage + var_stage / 2))
                        ith_var = torch.sum((torch.exp(var_stage) - 1) * torch.exp(2 * score_stage + var_stage))
                        pred_state.append([selc_ith + 1, ith_expect.item(), ith_var.item()])
                elif task_type == 'mledis_evidential':
                    variance = batch_preds[:, 3] / (batch_preds[:, 1] * (batch_preds[:, 2] - 1))
                    for stage in range(obj_num - 1):
                        score_stage = batch_preds[:, 0][stage:] - batch_preds[:, 0][stage]
                        var_stage = variance[stage:] + variance[stage]
                        expect = torch.sum(torch.exp(score_stage + var_stage / 2))
                        var = torch.sum((torch.exp(var_stage) - 1) * torch.exp(2 * score_stage + var_stage))
                        pred_state.append([stage + 1, expect.item(), var.item()])
                elif task_type == 'listnetdis_evidential':
                    # For the first stage, calculating the varicane and mean for every selection.
                    variance = batch_preds[:, 3] / (batch_preds[:, 1] * (batch_preds[:, 2] - 1))
                    for selc_ith in range(obj_num):
                        score_stage = batch_preds[:, 0] - batch_preds[:, 0][selc_ith]
                        var_stage = variance + variance[selc_ith]
                        ith_expect = torch.sum(torch.exp(score_stage + var_stage / 2))
                        ith_var = torch.sum((torch.exp(var_stage) - 1) * torch.exp(2 * score_stage + var_stage))
                        pred_state.append([selc_ith + 1, ith_expect.item(), ith_var.item()])
                else:
                    if model_type == 'mle':
                        for stage in range(obj_num - 1):
                            score_stage = batch_preds[stage:] - batch_preds[stage]
                            score_sum = torch.sum(torch.exp(score_stage))
                            pred_state.append([stage + 1, score_sum.item()])
                    elif model_type == 'listnet':
                        for selc_ith in range(obj_num):
                            ith_score = batch_preds - batch_preds[selc_ith]
                            ith_score_sum = torch.sum(torch.exp(ith_score))
                            pred_state.append([selc_ith + 1, ith_score_sum.item()])
                    else:
                        raise Exception('Unkonown model type')
                # calculating target state
                if model_type == 'mle':
                    for stage in range(obj_num - 1):
                        targte_score_stage = batch_targets[stage:] - batch_targets[stage]
                        targte_score_sum = torch.sum(torch.exp(targte_score_stage))
                        target_state.append([stage + 1, targte_score_sum.item()])
                elif model_type == 'listnet':
                    for selc_ith in range(obj_num):
                        ith_targte = batch_targets - batch_targets[selc_ith]
                        ith_targte_sum = torch.sum(torch.exp(ith_targte))
                        target_state.append([selc_ith + 1, ith_targte_sum.item()])

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
                
    # save data for further evaluation
    if saved_data_path is not None:
        if len(data_collector[0]) == 4:
            save_df = pd.DataFrame(data=data_collector, columns=['rsmi', 'psmi', 'targ', 'pred'])
        else:
            save_df = pd.DataFrame(data=data_collector, columns=['rsmi', 'psmi', 'targ', 'pred', 'uncertainty'])
        save_df.to_csv(saved_data_path, index=False)
    return average_score, average_pred_in_targ, average_top1_in_pred, pred_state, target_state