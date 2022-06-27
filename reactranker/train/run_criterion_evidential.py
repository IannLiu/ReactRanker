import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from .criterion_evidential import test


class run_test:
    
    def __init__(self,
                 gpu: int,
                 storage_location: str,
                 batch_size: int = 2):
        super(run_test, self).__init__()
        self.path = storage_location
        self.batch_size = batch_size
        self.gpu = gpu  # cuda

    def reset(self,
              gpu: int,
              storage_location: str,
              batch_size: int = 2):
        super(run_test, self).__init__()
        self.path = storage_location
        self.batch_size = batch_size
        self.gpu = gpu  # cuda
        
    def run_dis_model(self,
                      logger,
                      smiles2graph_dic,
                      saved_model_name: str,
                      test_data: pd.DataFrame,
                      model: nn.Module,
                      show_info: bool = True,
                      task_type: str = 'mle_gauss',
                      model_type: str = 'multi_stage',
                      target_name: str = 'ea',
                      smiles_list=None):
        # Run distribution models. The output including the score and uncertainties, thus one fold is enough.
        # Checking the model_name and test to ensure that test data has not been trained
    
        logger.info('\n\n**********************************\n**    Dis Model Test Section   **\n**********************************')
        if show_info is True:
            print('**********************************')
            print('**    Dis Model Test Section    **')
            print('**********************************')
            print(model)
        logger.info('Model Sturture')
        logger.info(model)
        path_checkpoints = self.path
        path_checkpoints = os.path.join(path_checkpoints, saved_model_name)
        batch_size = self.batch_size
        
        if self.gpu is not None:
            torch.cuda.set_device(self.gpu)
            model = model.cuda(self.gpu)
        print(path_checkpoints)
        score, score3, average_pred_in_targ, pred_state, target_state = test(model, test_data, path_checkpoints,
                                                                             batch_size, gpu=self.gpu, logger=logger,
                                                                             smiles2graph_dic=smiles2graph_dic,
                                                                             task_type=task_type, model_type=model_type,
                                                                             show_info=show_info,
                                                                             target_name=target_name,
                                                                             smiles_list=smiles_list)

        test_score = [score, score3, average_pred_in_targ]
        print("test score for k_fold vailidation is: ", test_score)
        logger.info('test score for k_fold vailidation is: {}'.format(test_score))
        
        return test_score, torch.FloatTensor(pred_state), torch.FloatTensor(target_state)
        
    def run_mc_model(self,
                     logger,
                     smiles2graph_dic,
                     saved_model_name: str,
                     test_data: pd.DataFrame,
                     model: nn.Module,
                     sample_number=10,
                     show_info: bool = True,
                     task_type: str = 'MC_dropout',
                     model_type: str = 'multi_stage',
                     target_name: str = 'ea',
                     smiles_list=None):
        # Run MC model: same test data with different dropout. The gpu seed should be specified for producitivity.
        # Checking the model_name and test to ensure that test data has not been trained
        
        pred_states = torch.FloatTensor([])
        target_states = torch.FloatTensor([])
        test_score = []
        for i in range(sample_number):
            logger.info('\n\n**********************************\n**    This is the sample [{}/{}]   **\n**********************************'.format(i + 1, sample_number))
            if show_info is True:
                print('**********************************')
                print('**   This is the sample [{}/{}]   **'.format(i + 1, sample_number))
                print('**********************************')
                print(model)
            logger.info('Model Sturture')
            logger.info(model)
            
            path_checkpoints = self.path
            path_checkpoints = os.path.join(path_checkpoints, saved_model_name)
            batch_size = self.batch_size
            torch.manual_seed(i)
            if self.gpu is not None:
                torch.cuda.manual_seed(i)
                torch.cuda.manual_seed_all(i)
                
            if self.gpu is not None:
                torch.cuda.set_device(self.gpu)
                model = model.cuda(self.gpu)
            print('model path:', path_checkpoints)
            score, score3, average_pred_in_targ, \
                pred_state, target_state = test(model, test_data, path_checkpoints, batch_size, gpu=self.gpu,
                                                logger=logger, smiles2graph_dic=smiles2graph_dic,
                                                task_type=task_type, model_type=model_type, show_info=show_info,
                                                target_name=target_name, smiles_list=smiles_list)

            pred_new = torch.FloatTensor(pred_state)
            targets_new = torch.FloatTensor(target_state)
            if i == 0:
                pred_states = torch.cat((pred_states, pred_new), 1)
                target_states = torch.cat((target_states, targets_new), 1)
            else:
                pred_states = torch.cat((pred_states, pred_new[:, 1].unsqueeze(1)), 1)
            test_score.append([score, score3, average_pred_in_targ])
        print("test score for sample vailidation is: ", test_score)
        logger.info('test score for sample vailidation is: {}'.format(test_score))
        
        mean_pred_states = torch.mean(pred_states[:, 1:], dim=1)
        std_pred_state = torch.std(pred_states[:, 1:], dim=1)
        pred = torch.cat((pred_states[:, 0].unsqueeze(1), mean_pred_states.unsqueeze(1),
                          std_pred_state.unsqueeze(1)), dim=1)
        
        return test_score, pred, target_states
        
    def run_ensemble_model(self, logger, smiles2graph_dic, test_data: pd.DataFrame, model: nn.Module, fold_number: int,
                           ensemble_number: int = 5, show_info: bool = True, task_type: str = 'ensemble',
                           model_type: str = 'multi_stage', target_name: str = 'ea', smiles_list=None):
        # This function just run one ensemble fold, and output the mean and variance
        pred_states = torch.FloatTensor([])
        target_states = torch.FloatTensor([])
        test_score = []
        for i in range(ensemble_number):
            logger.info('\n\n**********************************\n**  This is the ensemble [{}/{}]  **\n**********************************'.format(i + 1, ensemble_number))
            if show_info is True:
                print('**********************************')
                print('**  This is the ensemble [{}/{}]  **'.format(i + 1, ensemble_number))
                print('**********************************')
                print(model)
            logger.info('Model Sturture')
            logger.info(model)
            
            saved_model_name = str(fold_number) + '_' + str(i) + '.pt'
            path_checkpoints = self.path
            path_checkpoints = os.path.join(path_checkpoints, saved_model_name)
            batch_size = self.batch_size
            torch.manual_seed(i)
            if self.gpu is not None:
                torch.cuda.manual_seed(i)
                torch.cuda.manual_seed_all(i)
                
            if self.gpu is not None:
                torch.cuda.set_device(self.gpu)
                model = model.cuda(self.gpu)
            print('model path:', path_checkpoints)
            score, score3, average_pred_in_targ,\
                pred_state, target_state = test(model, test_data, path_checkpoints, batch_size, gpu=self.gpu,
                                                logger=logger, smiles2graph_dic=smiles2graph_dic,
                                                task_type=task_type, model_type=model_type, show_info=show_info,
                                                target_name=target_name, smiles_list=smiles_list)
            pred_new = torch.FloatTensor(pred_state)
            targets_new = torch.FloatTensor(target_state)
            if i == 0:
                pred_states = torch.cat((pred_states, pred_new), 1)
                target_states = torch.cat((target_states, targets_new), 1)
            else:
                pred_states = torch.cat((pred_states, pred_new[:, 1].unsqueeze(1)), 1)
            test_score.append([score, score3, average_pred_in_targ])
        print("test score for sample vailidation is: ", test_score)
        logger.info('test score for sample vailidation is: {}'.format(test_score))
        
        mean_pred_states = torch.mean(pred_states[:, 1:], dim=1)
        std_pred_state = torch.std(pred_states[:, 1:], dim=1)
        pred = torch.cat((pred_states[:, 0].unsqueeze(1), mean_pred_states.unsqueeze(1),
                          std_pred_state.unsqueeze(1)), dim=1)
        
        return test_score, pred, target_states


class criterion:

    def __init__(self,
                 target: torch.Tensor,
                 pred: torch.Tensor,
                 ):
        super(criterion, self).__init__()
        self.diff = torch.abs((pred[:, 1] - target[:, 1])) / target[:, 1]
        self.variance = torch.sqrt(pred[:, 2])
        self.pred = pred
        self.data = torch.cat((self.diff.unsqueeze(1), self.variance.unsqueeze(1)), dim=1)

    def spearman_coef(self,
                      show_info: bool = True,
                      show_warnning: bool = True,
                      show_error: bool = True):
        spearman_data = torch.FloatTensor([])
        spearman_coef = []
        spearman_p = []
        for ii in range(self.data.size(0)):
            if self.pred[ii, 0] == 1 and spearman_data.size(0) != 0:
                coef, p = spearmanr(spearman_data[:, 0].numpy(), spearman_data[:, 1].numpy())
                if show_info is True:
                    print('Spearmans correlation coefficient: %.3f' % coef)
                if coef < 0:
                    if show_warnning is True:
                        print('Spearmans correlation coefficient: %.3f' % coef)
                        print(spearman_data)
                if np.isnan(coef):
                    if show_error is True:
                        print('Spearmans correlation coefficient: %.3f' % coef)
                        print(spearman_data)
                spearman_data = torch.FloatTensor([])
                spearman_data = torch.cat((spearman_data, self.data[ii, :].unsqueeze(0)), dim=0)
                if np.isnan(coef) == False:
                    spearman_coef.append(coef)
                    spearman_p.append(p)
            else:
                spearman_data = torch.cat((spearman_data, self.data[ii, :].unsqueeze(0)), dim=0)

        return spearman_coef, spearman_p

    def erro_confidence(self,
                        grid: int = 100,
                        show_info: bool = True,
                        show_warnning: bool = True):
        # 1. sort the varance and diff
        sorted_var, sorted_var_idx = torch.sort(self.variance, descending=True)
        # print(sorted_var)
        sorted_diff = torch.index_select(self.diff, dim=0, index=sorted_var_idx)
        # 2. cal mean and var at different cut-off
        calibrate_diff_data = []
        for i in range(grid):
            data_length = int(i / grid * len(sorted_diff))
            # print(data_length)
            slec_diff_data = sorted_diff[data_length:]
            diff_mae = torch.mean(slec_diff_data)
            diff_rmse = torch.sqrt(torch.mean(torch.pow(slec_diff_data, 2)))
            calibrate_diff_data.append([diff_mae.item(), diff_rmse.item()])

        return calibrate_diff_data
