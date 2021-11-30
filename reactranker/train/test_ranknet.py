from logging import Logger
import torch
import torch.nn as nn
from pandas import DataFrame

from ..data.load_reactions import DataProcessor
from .eval import evaluate_top_scores


def test(model: nn.Module,
         test_data: DataFrame,
         path_checkpoints: str,
         smiles2graph_dic,
         batch_size: int,
         gpu: int,
         logger: Logger = None):

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
    print('means is: ', means)
    print('stds is: ', stds)
    if means is not None:
        test_std_targ = test_data['ea'].map(lambda x: -(x - means) / stds)
        test_data['std_targ'] = test_std_targ
    model = model
    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    model.load_state_dict(loaded_state_dict)
    test_data_processor = DataProcessor(test_data)
    smiles2graph_dic = smiles2graph_dic

    #evaluate the model
    average_score, average_pred_in_targ, average_top1_in_pred \
        = evaluate_top_scores(model, gpu, test_data_processor, smiles2graph_dic=smiles2graph_dic,
                              ratio=0.25, batch_size=batch_size)

    print('==========================================')
    print('   Note：For test set average score is:   ', average_score)
    # print('   Note：For test set average score3 is:   ', average_score3)
    print('   Note：For test set {} average pred in targ is:{}'.format(0.25, average_pred_in_targ))
    print('==========================================')
    logger.info('\n==========================================\nNote：For test set average score is: {:.4f}'.format(average_score))
    logger.info('\nNote：For target in pred top 25% is: {:.4f}\n'.format(average_pred_in_targ))
    logger.info('\nNote：For target top1 in pred top 25% is: {:.4f}\n'.format(average_top1_in_pred))

    return average_score, average_pred_in_targ, average_top1_in_pred
