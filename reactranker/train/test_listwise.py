from typing import Union
from logging import Logger

import torch
from ..data.load_reactions import DataProcessor
from .eval import evaluate_top_scores


def test(model, test_data,  path_checkpoints, batch_size, smiles2graph_dic,
         gpu: Union[int, str], logger: Logger = None):
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
    
    #build and load model
    state = torch.load(path_checkpoints, map_location=lambda storage, loc: storage)
    means = state['data_scaler']['means']
    stds = state['data_scaler']['stds']
    loaded_state_dict = state['state_dict']
    if means is not None:
        test_std_targ = test_data['ea'].map(lambda x: -(x - means) / stds)
        test_data['std_targ'] = test_std_targ
    model = model
    if gpu is not None:
        model = model.cuda()
    model.load_state_dict(loaded_state_dict)
    
    #evaluate
    model.eval()
    ratio = 0.25
    test_data_processor = DataProcessor(test_data)
    smiles2graph_dic = smiles2graph_dic
    average_score, average_pred_in_targ, average_top1_in_pred = \
        evaluate_top_scores(model, gpu=gpu, data_processor=test_data_processor,
                            smiles2graph_dic=smiles2graph_dic, batch_size=batch_size, ratio=0.25)


    print('==========================================')
    print('   Note：For test set average score is:   ', average_score)
    # print('   Note：For test set average score3 is:   ', average_score3)
    print('   Note：For test set {} average pred in targ is:{}'.format(ratio, average_pred_in_targ))
    print('   Note：For average target top1 in pred {} is:{}'.format(ratio, average_top1_in_pred))
    print('==========================================')
    logger.info('\n==========================================\n  Note：For test set average score is: {:.4f}\n'.format(average_score))
    # logger.info('\n  Note：For test set average score3 is: {:.4f}\n'.format(average_score3))
    logger.info('\n  Note：For test set average top1 in 25% is: {:.4f}\n'.format(average_top1_in_pred))
    logger.info('\n  Note：For test set average 25% is: {:.4f}\n=========================================='.format(average_pred_in_targ))

    return average_score, average_pred_in_targ, average_top1_in_pred
