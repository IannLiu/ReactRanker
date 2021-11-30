import torch
import torch.nn as nn
from torch.distributions import Dirichlet
from itertools import combinations
import numpy as np

import torch.nn.functional as F

class LogCumsumExp(torch.autograd.Function):
	'''
	The PyTorch OP corresponding to the operation: log{ |sum_k^m{ exp{pred_k} } }
	'''
	@staticmethod
	def forward(ctx, input):
		'''
		In the forward pass we receive a context object and a Tensor containing the input;
		we must return a Tensor containing the output, and we can use the context object to cache objects for use in the backward pass.
		Specifically, ctx is a context object that can be used to stash information for backward computation.
		You can cache arbitrary objects for use in the backward pass using the ctx.save_for_backward method.
		:param ctx:
		:param input: i.e., batch_preds of [batch, ranking_size], each row represents the relevance predictions for documents within a ltr_adhoc
		:return: [batch, ranking_size], each row represents the log_cumsum_exp value
		'''

		m, _ = torch.max(input, dim=0, keepdim=True)    #a transformation aiming for higher stability when computing softmax() with exp()
		y = input - m
		y = torch.exp(y)
		y_cumsum_t2h = torch.flip(torch.cumsum(torch.flip(y, dims=[0]), dim=0), dims=[0])    #row-wise cumulative sum, from tail to head
		fd_output = torch.log(y_cumsum_t2h) + m # corresponding to the '-m' operation

		ctx.save_for_backward(input, fd_output)

		return fd_output


	@staticmethod
	def backward(ctx, grad_output):
		'''
		In the backward pass we receive the context object and
		a Tensor containing the gradient of the loss with respect to the output produced during the forward pass (i.e., forward's output).
		We can retrieve cached data from the context object, and
		must compute and return the gradient of the loss with respect to the input to the forward function.
		Namely, grad_output is the gradient of the loss w.r.t. forward's output. 
        Here we first compute the gradient (denoted as grad_out_wrt_in) of forward's output w.r.t. forward's input.
		Based on the chain rule, grad_output * grad_out_wrt_in would be the desired output, 
        i.e., the gradient of the loss w.r.t. forward's input
		:param ctx:
		:param grad_output:
		:return:
		'''

		input, fd_output = ctx.saved_tensors
		#chain rule
		bk_output = grad_output * (torch.exp(input) * torch.cumsum(torch.exp(-fd_output), dim=0))

		return bk_output

class MLEloss(nn.Module):
    
    def __init__(self):
        super(MLEloss, self).__init__()
        
    def forward(self,
                score,
                scope,
                targets_train,
                gpu:int):
        """
        This function is to calculate ListMLE loss(https://dl.acm.org/doi/abs/10.1145/1273496.1273513)
        :paramscore: the output of nural network
        Note: Inputs must be ordered from the most relevant one to the most irrelevant one
        """
        losses = torch.Tensor([0])
        scope = scope.tolist()
        
        if gpu is not None:
            torch.cuda.set_device(gpu)
            losses = losses.cuda(gpu)
            targets_train = targets_train.cuda(gpu)
        for item, batch_targets in zip(score.split(scope, dim = 0), targets_train.split(scope, dim = 0)):
            batch_sorted_idx = torch.argsort(batch_targets, descending=True) 
            # For pred score, we wanna higher score for lower activation energy
            # However, the standard score is obtained from activation energy. So, the higher score corresponds to higer activation energy
            # So, we input the -[score]
            sorted_item = torch.gather(item, dim = 0, index = batch_sorted_idx)
            logcumsumexps = LogCumsumExp.apply(sorted_item)
            loss = torch.mean(logcumsumexps - sorted_item)
            losses += loss
            
        average_losses = losses/ len(scope)
        #print('the MLE loss is: ', average_losses)
            
        return average_losses
        
class MLEDisLoss(nn.Module):
    
    def __init__(self):
        super(MLEDisLoss, self).__init__()
        
    def forward(self,
                mean,
                variance,
                scope,
                targets,
                gpu:int):
        """
        This function is to calculate ListMLE loss(https://dl.acm.org/doi/abs/10.1145/1273496.1273513)
        :paramscore: the output of nural network
        Note: Inputs must be ordered from the most relevant one to the most irrelevant one
        """
        losses = torch.Tensor([0])
        
        if gpu is not None:
            torch.cuda.set_device(gpu)
            losses = losses.cuda(gpu)
            targets = targets.cuda(gpu)
        score = torch.cat((mean, variance), 1)
        for item, batch_targets in zip(score.split(scope, dim=0), targets.split(scope, dim=0)):
            batch_sorted_idx = torch.argsort(batch_targets, descending=True)
            # For pred score, we wanna higher score for lower activation energy
            # However, the standard score is obtained from activation energy.
            # So, the higher score corresponds to higer activation energy
            # So, we input the -[score]
            sorted_item = torch.index_select(item, dim=0, index=batch_sorted_idx)
            x1 = -sorted_item[:, 0].repeat(sorted_item[:, 0].size(dim=0), 1)
            x2 = -x1.t()
            y1 = sorted_item[:, 1].repeat(sorted_item[:, 1].size(dim=0), 1)
            y2 = y1.t()
            loss = torch.mean(-torch.log(1/torch.sum(torch.tril(torch.exp(x1+x2+(y1+y2)/2)), 0)))  # mean?
            losses += loss
            
        average_losses = losses / len(scope)
            
        return average_losses
        
        
class Rankloss(nn.Module):

    def __init__(self):
        super(Rankloss, self).__init__()
        
        self.pred_fun = nn.Sigmoid()
        self.loss_fun = nn.BCELoss()

    def forward(self,
                score: torch.Tensor,
                label: torch.Tensor,
                scope: torch.IntTensor,
                gpu: int,
                label_type: str="energy"):
        """
        This function is to calculate the loss of RankNet
        
        :param score: Nural network output(The score of every reactions)
        :param label: The initail labels of every reactions so that preferred reactions are recognized
        :param scope: Which group the score belongs to(for calculating the loss function)
        :param label_type: The label represents activation energy('energy'), or reaction rate('rate') 
        """
        idx = torch.IntTensor([0])
        losses = torch.Tensor([0])
        if gpu is not None:
            torch.cuda.set_device(gpu)
            label = label.cuda(gpu)
            scope = scope.cuda(gpu)
            idx = idx.cuda(gpu)
            losses = losses.cuda(gpu)
            score = score.cuda(gpu)
        for items in scope:
            s = score[idx:idx+items]
            combs = torch.combinations(s, r=2)
            s1, s2 = combs.split(1, dim=1)
            pred = self.pred_fun(s1-s2)
            
            t = label[idx:idx+items]
            combt = torch.combinations(t, 2)
            t1, t2 = combt.split(1, dim=1)
            if label_type == 'energy':
                target = torch.sign(t2-t1)
            else:
                target = torch.sign(t1-t2)
            p_true= 1/2 * (1 + target)
            # update idx
            idx += items
            
            loss = self.loss_fun(pred, p_true)
            losses +=loss  
        average_loss = losses/len(scope)
    
        return average_loss
        
def Rank_eval(score: torch.Tensor,
              label: torch.Tensor,
              scope: torch.IntTensor,
              label_type: str="energy"):

    """
    This function is to calculate the loss of RankNet
    
    :param score: Nural network output(The score of every reactions)
    :param label: The initail labels of every reactions so that preferred reactions are recognized
    :param scope: Which garoup the score belongs to(for calculating the loss function)
    :param label_type: The label represents activation energy('energy'), or reaction rate('rate') 
    """
    idx = torch.IntTensor([0])
    MAP_score = []
    pred_fun = nn.Sigmoid()
    
    for items in scope:
        s = score[idx:idx+items]
        combs = torch.combinations(s, r=2)
        s1, s2 = combs.split(1, dim=1)
        pred = pred_fun(s1-s2)
        
        t = label[idx:idx+items]
        combt = torch.combinations(t, 2)
        t1, t2 = combt.split(1, dim=1)
        if label_type == 'energy':
            target = torch.sign(t2-t1)
        else:
            target = torch.sign(t1-t2)
        p_true= 1/2 * (1 + target)
        # update idx
        idx += items
        
        pred = pred.tolist()
        p_true = p_true.tolist()
        
        pred_reduce = [j for i in pred for j in i]
        p_true_reduce = [j for i in p_true for j in i]
        
        for  item1, item2 in zip(pred_reduce, p_true_reduce):
            if (float(item1) > 0.5 and float(item2) == 1) or (float(item1) < 0.5 and float(item2) == 0) or (float(item1) == 0.5 and float(item2) == 0.5):
                MAP_score.append(1)
            else:
                MAP_score.append(0)
        
    return MAP_score
    
class GaussDisLoss(nn.Module):

    """
    This function is to define the loss for the regression task.
    Note:Assuming the distribution of score is i.i.d Gaussian distribution
    """
    def __init__(self):
        super(GaussDisLoss, self).__init__()
        self.pi = torch.Tensor([np.pi])
        
    def forward(self, mean_scores, std_scores, targets, gpu:int):
        if gpu is not None:
            torch.cuda.set_device(gpu)
            targets = targets.cuda(gpu)
            self.pi = self.pi.cuda(gpu)
        mse = torch.log(std_scores) + torch.pow((mean_scores - targets), 2) / std_scores
        #mse = torch.pow((mean_scores - targets), 2)
        
        #print(std_scores)
        #print(torch.log(std_scores))
        #print(torch.pow((mean_scores - targets), 2) / std_scores)
        #print('the gaussian loss is: ', torch.mean(mse))

        return torch.mean(mse)
        
class ListNetT1Loss(nn.Module):

    """
    This function is to define the loss for the regression task.
    Note:Assuming the distribution of score is i.i.d Gaussian distribution
    """
    def __init__(self):
        super(ListNetT1Loss, self).__init__()

    def forward(self,
                means,
                scope,
                targets_train,
                gpu:int):
        '''
		#- deprecated way -#
		batch_top1_pros_pred = F.softmax(batch_preds, dim=1)
		batch_top1_pros_std = F.softmax(batch_stds, dim=1)
		batch_loss = torch.sum(-torch.sum(batch_top1_pros_std * torch.log(batch_top1_pros_pred), dim=1))
		'''
        
        losses = torch.Tensor([0])
        scope = scope.tolist()
        
        if gpu is not None:
            torch.cuda.set_device(gpu)
            losses = losses.cuda(gpu)
            targets_train = targets_train.cuda(gpu)
        for item, batch_targets in zip(means.split(scope, dim = 0), targets_train.split(scope, dim = 0)):
            loss = -torch.sum(F.softmax(batch_targets, dim=0) * F.log_softmax(item, dim=0))
            losses += loss
            
            
        average_losses = losses/ len(scope)
        #print('the MLE loss is: ', average_losses)
            
        return average_losses
        
class MLEloss_KL(nn.Module):
    
    def __init__(self):
        super(MLEloss, self).__init__()
        
    def forward(self,
                score,
                scope,
                targets_train,
                gpu:int):
        """
        This function is to calculate ListMLE loss(https://dl.acm.org/doi/abs/10.1145/1273496.1273513)
        :paramscore: the output of nural network
        Note: Inputs must be ordered from the most relevant one to the most irrelevant one
        """
        losses = torch.Tensor([0])
        scope = scope.tolist()
        
        if gpu is not None:
            torch.cuda.set_device(gpu)
            losses = losses.cuda(gpu)
            targets_train = targets_train.cuda(gpu)
        for item, batch_targets in zip(score.split(scope, dim = 0), targets_train.split(scope, dim = 0)):
            sorted_tensor,batch_sorted_idx = torch.sort(batch_targets, descending=False)
            # For pred score, we wanna higher score for lower activation energy
            # However, the standard score is obtained from activation energy. So, the higher score corresponds to higer activation energy
            # So, we input the -[score]
            sorted_item = torch.gather(item, dim = 0, index = batch_sorted_idx)
            logcumsumexps = LogCumsumExp.apply(sorted_item)
            prob_pred =  sorted_item-logcumsumexps
            
            m, _ = torch.max(-sorted_tensor, dim=0, keepdim=True)    #a transformation aiming for higher stability when computing softmax() with exp()
            y = -sorted_tensor - m
            y = torch.exp(y)
            y_cumsum_t2h = torch.flip(torch.cumsum(torch.flip(y, dims=[0]), dim=0), dims=[0])    #row-wise cumulative sum, from tail to head
            fd_output = torch.log(y_cumsum_t2h) + m # corresponding to the '-m' operation
            prob_std = -sorted_tensor - fd_output
            
            loss = torch.mean(torch.exp(prob_std) * (prob_std - prob_pred))

            losses += loss
            
            
        average_losses = losses/ len(scope)
        #print('the MLE loss is: ', average_losses)
            
        return average_losses
        
def evidential_loss_new(mu, v, alpha, beta, targets, gpu, lam=1, epsilon=1e-4):
    """
    Use Deep Evidential Regression negative log likelihood loss + evidential
        regularizer

    :mu: pred mean parameter for NIG
    :v: pred lam parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :targets: Outputs to predict

    :return: Loss
    """
    if gpu is not None:
        torch.cuda.set_device(gpu)
        targets = targets.cuda(gpu)
    # Calculate NLL loss
    twoBlambda = 2*beta*(1+v)
    nll = 0.5*torch.log(np.pi/v) \
        - alpha*torch.log(twoBlambda) \
        + (alpha+0.5) * torch.log(v*(targets-mu)**2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha+0.5)

    L_NLL = nll #torch.mean(nll, dim=-1)

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((targets - mu))
    reg = error * (2 * v + alpha)
    L_REG = reg #torch.mean(reg, dim=-1)

    # Loss = L_NLL + L_REG
    # TODO If we want to optimize the dual- of the objective use the line below:
    loss = L_NLL + lam * (L_REG - epsilon)

    return torch.mean(loss)

class Dirichlet_KL(nn.Module):
    
    def __init__(self):
        super(Dirichlet_KL, self).__init__()
        
        
    def forward(self,
                concentration,
                scope,
                targets,
                gpu:int):
        losses = torch.Tensor([0])
        scope = scope.tolist()
        KL_Div = nn.KLDivLoss()
        
        if gpu is not None:
            torch.cuda.set_device(gpu)
            losses = losses.cuda(gpu)
            targets = targets.cuda(gpu)
        for alpha, batch_targets in zip(concentration.split(scope, dim = 0), targets.split(scope, dim = 0)):
            dirich_target = Dirichlet(torch.exp(batch_targets))
            dirich_target_pdf = dirich_target.log_prob(dirich_target.mean)
            dirich_pred = Dirichlet(torch.exp(alpha))
            dirich_pred_pdf = dirich_pred.log_prob(dirich_target.mean)
            #print('pred_pdf is {}, target_pdf is {}'.format(dirich_pred_pdf, dirich_target_pdf))
            loss = KL_Div(dirich_pred_pdf, dirich_target_pdf)
            
            losses += loss
        average_losses = losses/ len(scope)
        #print(average_losses)
        return average_losses