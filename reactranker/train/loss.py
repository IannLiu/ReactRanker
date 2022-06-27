import torch
import torch.nn as nn
from torch.distributions import Dirichlet
import numpy as np

import torch.nn.functional as F


class LogCumsumExp(torch.autograd.Function):
    """
    The PyTorch OP corresponding to the operation: log{ |sum_k^m{ exp{pred_k} } }
    """

    @staticmethod
    def forward(ctx, input_data):
        """
        In the forward pass we receive a context object and a Tensor containing the input;
        we must return a Tensor containing the output, and we can use the context object to cache objects for use in
        the backward pass.
        Specifically, ctx is a context object that can be used to stash information for backward computation.
        You can cache arbitrary objects for use in the backward pass using the ctx.save_for_backward method.
        :param ctx:
        :param input_data: i.e., batch_preds of [batch, ranking_size], each row represents the relevance predictions for
         documents within a ltr_adhoc
        :return: [batch, ranking_size], each row represents the log_cumsum_exp value
        """
        # a transformation aiming for higher stability when computing softmax() with exp()
        m, _ = torch.max(input_data, dim=0, keepdim=True)
        y = input_data - m
        y = torch.exp(y)
        # row-wise cumulative sum, from tail to head
        y_cumsum_t2h = torch.flip(torch.cumsum(torch.flip(y, dims=[0]), dim=0), dims=[0])
        # corresponding to the '-m' operation
        fd_output = torch.log(y_cumsum_t2h) + m

        ctx.save_for_backward(input_data, fd_output)

        return fd_output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive the context object and
        a Tensor containing the gradient of the loss with respect to the output produced during the forward pass
        (i.e., forward's output).
        We can retrieve cached data from the context object, and
        must compute and return the gradient of the loss with respect to the input to the forward function.
        Namely, grad_output is the gradient of the loss w.r.t. forward's output.
        Here we first compute the gradient (denoted as grad_out_wrt_in) of forward's output w.r.t. forward's input.
        Based on the chain rule, grad_output * grad_out_wrt_in would be the desired output,
        i.e., the gradient of the loss w.r.t. forward's input
        :param ctx:
        :param grad_output:
        :return:
        """

        input_tensor, fd_output = ctx.saved_tensors
        # chain rule
        bk_output = grad_output * (torch.exp(input_tensor) * torch.cumsum(torch.exp(-fd_output), dim=0))

        return bk_output


class MLEloss(nn.Module):

    def __init__(self):
        super(MLEloss, self).__init__()

    def forward(self,
                score,
                scope,
                targets_train,
                gpu: int):
        """
        This function is to calculate ListMLE loss(https://dl.acm.org/doi/abs/10.1145/1273496.1273513)
        :param score: the output of neural network
        Note: Inputs must be ordered from the most relevant one to the most irrelevant one
        """
        losses = torch.Tensor([0])
        # scope = scope.tolist()

        if gpu is not None:
            torch.cuda.set_device(gpu)
            losses = losses.cuda(gpu)
            targets_train = targets_train.cuda(gpu)
        for item, batch_targets in zip(score.split(scope, dim=0), targets_train.split(scope, dim=0)):
            batch_sorted_idx = torch.argsort(batch_targets, descending=True)
            # For pred score, we wanna higher score for lower activation energy
            # However, the standard score is obtained from activation energy.
            # So, the higher score corresponds to higher activation energy
            # So, we input the -[score]
            sorted_item = torch.gather(item, dim=0, index=batch_sorted_idx)
            logcumsumexps = LogCumsumExp.apply(sorted_item)
            loss = torch.mean(logcumsumexps - sorted_item)
            losses += loss

        average_losses = losses / len(scope)

        return average_losses


class MLEDisLoss(nn.Module):

    def __init__(self):
        super(MLEDisLoss, self).__init__()

    def forward(self,
                mean,
                variance,
                scope,
                targets,
                gpu: int):
        """
        This function is to calculate ListMLE loss(https://dl.acm.org/doi/abs/10.1145/1273496.1273513)
        :param score: the output of neural network
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
            # So, the higher score corresponds to higher activation energy
            # So, we input the -[score]
            sorted_item = torch.index_select(item, dim=0, index=batch_sorted_idx)
            x1 = -sorted_item[:, 0].repeat(sorted_item[:, 0].size(dim=0), 1)
            x2 = -x1.t()
            y1 = sorted_item[:, 1].repeat(sorted_item[:, 1].size(dim=0), 1)
            y2 = y1.t()
            loss = torch.mean(-torch.log(1 / torch.sum(torch.tril(torch.exp(x1 + x2 + (y1 + y2) / 2)), 0)))  # mean?
            losses += loss

        average_losses = losses / len(scope)

        return average_losses


class GaussDisLoss(nn.Module):
    """
    This function is to define the loss for the regression task.
    Note:Assuming the distribution of score is i.i.d Gaussian distribution
    """

    def __init__(self):
        super(GaussDisLoss, self).__init__()
        self.pi = torch.Tensor([np.pi])

    def forward(self, mean_scores, std_scores, targets, gpu: int):
        if gpu is not None:
            torch.cuda.set_device(gpu)
            targets = targets.cuda(gpu)
            self.pi = self.pi.cuda(gpu)
        mse = 0.5 * torch.log(2 * self.pi) + 0.5 * torch.log(std_scores) + torch.pow((mean_scores - targets), 2) / (
                    2 * std_scores)

        return torch.mean(mse)


class Lognorm(nn.Module):
    """
    This function is to define the loss for the regression task.
    Note:Assuming the distribution of score is i.i.d Gaussian distribution
    """

    def __init__(self):
        super(Lognorm, self).__init__()
        self.pi = torch.Tensor([np.pi])

    def forward(self, scores, std_scores, targets, gpu: int):
        if gpu is not None:
            torch.cuda.set_device(gpu)
            targets = targets.cuda(gpu)
            self.pi = self.pi.cuda(gpu)
        mse = 0.5 * torch.log(2 * self.pi) + 0.5 * torch.log(std_scores * (scores ** 2)) + \
              torch.pow((torch.log(scores) - targets), 2) / (2 * std_scores)
        print(torch.mean(mse))

        return torch.mean(mse)


class Listnet_For_evidential(nn.Module):
    """
    This function is to define the loss for the multi-object optim
    We combine the ListNet and evidential regression together
    The variance obtained from Evidential regression will be added to
    ListNet. And the score ~ lognormal distribution
    """

    def __init__(self):
        super(Listnet_For_evidential, self).__init__()

    def forward(self,
                mean,
                v,
                alpha,
                # variance,
                scope,
                targets,
                gpu: int):
        """
        #- deprecated way -#
        batch_top1_pros_pred = F.softmax(batch_preds, dim=1)
        batch_top1_pros_std = F.softmax(batch_stds, dim=1)
        batch_loss = torch.sum(-torch.sum(batch_top1_pros_std * torch.log(batch_top1_pros_pred), dim=1))
        """
        losses = torch.Tensor([0])

        if gpu is not None:
            torch.cuda.set_device(gpu)
            losses = losses.cuda(gpu)
            targets = targets.cuda(gpu)
        score = torch.cat((mean, v, alpha), 1)
        for item, batch_targets in zip(score.split(scope, dim=0), targets.split(scope, dim=0)):
            # pred = torch.log(torch.exp(item[:, 0] + item[:, 1]/2)/torch.sum(torch.exp(item[:, 0] + item[:, 1]/2)))
            # targ = F.softmax(batch_targets, dim=0)
            pred = torch.log_softmax(item[:, 0], dim=0)
            targ = torch.softmax(batch_targets, dim=0)
            loss = -torch.mean(targ * pred * (2 * item[:, 1] + item[:, 2]))
            losses += loss

        average_losses = losses / len(scope)
        # print('the MLE loss is: ', average_losses)

        return average_losses


class Listnet_For_Gauss(nn.Module):
    """
    This function is to define the loss for the multi-object optim
    We combine the ListNet and gaussian regression together
    The variance obtained from Evidential regression will be added to
    ListNet. And the score ~ lognormal distribution
    """

    def __init__(self):
        super(Listnet_For_Gauss, self).__init__()

    def forward(self,
                mean,
                variance,
                scope,
                targets,
                gpu: int):

        losses = torch.Tensor([0])
        if gpu is not None:
            torch.cuda.set_device(gpu)
            losses = losses.cuda(gpu)
            targets = targets.cuda(gpu)
        score = torch.cat((mean, variance), 1)
        for item, batch_targets in zip(score.split(scope, dim=0), targets.split(scope, dim=0)):
            x1 = item[:, 0].repeat(item[:, 0].size(dim=0), 1)
            x2 = x1.t()
            y1 = item[:, 1].repeat(item[:, 1].size(dim=0), 1)
            y2 = y1.t()
            pred = 1 / torch.sum((torch.exp(x1 - x2 + (y1 + y2) / 2)), dim=1)
            z1 = batch_targets.repeat(batch_targets.size(dim=0), 1)
            z2 = z1.t()
            targ = 1 / torch.sum(torch.exp(z1 - z2), dim=1)
            loss = -torch.mean((targ * torch.log(pred)))
            losses += loss

        average_losses = losses / len(scope)
        # print('the MLE loss is: ', average_losses)

        return average_losses


class Listnetlognorm(nn.Module):
    """
    This function is to define the loss for the multi-object optim
    We combine the ListNet and gaussian regression together
    The variance obtained from Evidential regression will be added to
    ListNet. And the score ~ lognormal distribution
    """

    def __init__(self):
        super(Listnetlognorm, self).__init__()

    def forward(self,
                mean,
                variance,
                scope,
                targets,
                gpu: int):

        losses = torch.Tensor([0])
        if gpu is not None:
            torch.cuda.set_device(gpu)
            losses = losses.cuda(gpu)
            targets = targets.cuda(gpu)
        score = torch.cat((mean, variance), 1)
        for item, batch_targets in zip(score.split(scope, dim=0), targets.split(scope, dim=0)):
            x1 = item[:, 0].repeat(item[:, 0].size(dim=0), 1)
            x2 = x1.t()
            y1 = item[:, 1].repeat(item[:, 1].size(dim=0), 1)
            y2 = y1.t()
            pred = 1 / torch.sum(x1 / x2 * (torch.exp((y1 + y2) / 2)), dim=1)
            z1 = batch_targets.repeat(batch_targets.size(dim=0), 1)
            z2 = z1.t()
            targ = 1 / torch.sum(torch.exp(z1 - z2), dim=1)
            loss = -torch.mean((targ * torch.log(pred)))
            losses += loss

        average_losses = losses / len(scope)
        # print('the MLE loss is: ', average_losses)

        return average_losses


class ListnetLoss(nn.Module):
    """
    This function is to define the loss for the ListNet
    We define the KL-Divergence as the loss function
    """

    def __init__(self):
        super(ListnetLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self,
                score,
                scope,
                targets,
                gpu: int):
        losses = torch.Tensor([])
        # length = torch.sum(torch.Tensor(scope))
        if gpu is not None:
            torch.cuda.set_device(gpu)
            losses = losses.cuda(gpu)
            targets = targets.cuda(gpu)
        for item, batch_targets in zip(score.split(scope, dim=0), targets.split(scope, dim=0)):
            pred = torch.log(F.softmax(item, dim=0))
            # print(pred)
            targ = F.softmax(batch_targets, dim=0)
            # print(targ)
            loss = -targ * pred
            losses = torch.cat((losses, loss), dim=0)
            # losses += -torch.mean(targ * pred)

        average_losses = torch.mean(losses)   # losses / len(scope)
        # average_losses = losses / length
        # average_loss = losses/ torch.Tensor(len(scope))
        # print('the MLE loss is: ', average_losses)

        return average_losses


class Listnet_with_uq(nn.Module):
    """
    This function is to define the loss for the ListNet
    We define the KL-Divergence as the loss function
    """

    def __init__(self):
        super(Listnet_with_uq, self).__init__()
        self.loss_KL = nn.KLDivLoss(reduction='batchmean')

    def forward(self,
                score,
                scope,
                targets,
                max_coeff,
                epoch,
                epochs,
                gpu: int):

        losses = torch.Tensor([0])

        if gpu is not None:
            torch.cuda.set_device(gpu)
            losses = losses.cuda(gpu)
            targets = targets.cuda(gpu)
        for item, batch_targets in zip(score.split(scope, dim=0), targets.split(scope, dim=0)):
            pred_p = item / torch.sum(item)
            targ_p = F.softmax(batch_targets, dim=0)
            penalty_std = torch.ones(len(item))
            if gpu is not None:
                penalty_std = penalty_std.cuda(gpu)
            real_loss = self.loss_KL(torch.log(pred_p), targ_p)
            # KL = targ_p * torch.log(targ_p/pred_p)
            # residue = KL * (item - penalty_std)
            consist = torch.log(targ_p / pred_p)
            residue = consist * (item - penalty_std)
            penalty_loss = torch.abs(residue)
            annealing_coef = max_coeff * (epoch / (epochs - 1)) ** 3  # (epochs / (epoch + 1))
            loss = real_loss + annealing_coef * penalty_loss
            losses += torch.mean(loss)

        average_losses = losses / len(scope)
        # print('the MLE loss is: ', average_losses)

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
    twoBlambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
          - alpha * torch.log(twoBlambda) \
          + (alpha + 0.5) * torch.log(v * (targets - mu) ** 2 + twoBlambda) \
          + torch.lgamma(alpha) \
          - torch.lgamma(alpha + 0.5)

    L_NLL = nll  # torch.mean(nll, dim=-1)

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((targets - mu))
    reg = error * (2 * v + alpha)
    L_REG = reg  # torch.mean(reg, dim=-1)

    # Loss = L_NLL + L_REG
    # TODO If we want to optimize the dual- of the objective use the line below:
    loss = L_NLL + lam * (L_REG - epsilon)

    return torch.mean(loss)


class Dirichlet_uq(nn.Module):

    def __init__(self):
        super(Dirichlet_uq, self).__init__()

    def forward(self,
                concentration,
                scope,
                targets,
                max_coeff,
                epoch,
                epochs,
                gpu: int):
        losses = torch.Tensor([0])

        if gpu is not None:
            torch.cuda.set_device(gpu)
            losses = losses.cuda(gpu)
            targets = targets.cuda(gpu)
        for alpha, batch_targets in zip(concentration.split(scope, dim=0), targets.split(scope, dim=0)):
            pred_p = alpha / torch.sum(alpha)
            targ_p = F.softmax(batch_targets, dim=0)
            err = (pred_p - targ_p) ** 2
            var = pred_p * (1 - pred_p) / (torch.sum(alpha) + 1)
            consist = torch.log(targ_p / pred_p)
            residue = consist * (alpha - 1)
            penalty_loss = torch.abs(residue)
            annealing_coef = max_coeff * (epoch / (epochs - 1)) ** 3  # (epochs / (epoch + 1))
            loss = torch.mean(err + var + annealing_coef * penalty_loss)

            losses += loss

        average_losses = losses / len(scope)

        return average_losses


class evidential_ranking(nn.Module):
    """
    The input including the ranking total scores and uncertainty factors.
    Uncertainty_score = total_score * uncertainty_factores
    The uncertainty factor should between [0, 1], therefore, the last layer for uncertainty is sigmoid
    The total score should be larger than 1. Therefore, the last layer for score is softmax.
    Or the loss should be writen as softmax
    """

    def __init__(self):
        super(evidential_ranking, self).__init__()
        # self.pi = torch.FloatTensor([np.pi])

    def forward(self,
                possibilities,
                scope,
                targets,
                max_coeff,
                epoch,
                epochs,
                gpu: int):
        losses = torch.Tensor([0])

        if gpu is not None:
            torch.cuda.set_device(gpu)
            losses = losses.cuda(gpu)
            targets = targets.cuda(gpu)
        for item, batch_targets in zip(possibilities.split(scope, dim=0), targets.split(scope, dim=0)):
            """
            evidential_score = item[:, 0]
            uncertainty_score = item[:, 1]
            targets_possibility = F.softmax(batch_targets, dim=0)
            total_score = torch.sum(evidential_score)
            # targets_score = targets_possibility * total_score
            pred_possibility = evidential_score / total_score
            Score_CrossEntropyLoss = -targets_possibility * torch.log(pred_possibility)

            uncertainty_possibility = uncertainty_score / total_score ** 2
            # uncertainty_loss = (pred_possibility - targets_possibility) ** 2 / uncertainty_possibility + torch.log(
            #     2 * 3.141592653589793 * uncertainty_possibility)
            uncertainty_loss = (pred_possibility - targets_possibility) ** 2 / (uncertainty_score * 2) + 0.5 * torch.log(
                 2 * 3.141592653589793 * uncertainty_score)

            # annealing_coef = max_coeff * (epoch / (epochs - 1)) ** 3  # (epochs / (epoch + 1))
            # loss = torch.mean(Score_CrossEntropyLoss + annealing_coef * err1)
            loss = torch.mean(uncertainty_loss + Score_CrossEntropyLoss)

            losses += loss
            """
            norm_mean = item[:, 0]
            norm_variance = item[:, 1]
            # score = torch.exp(norm_mean + norm_variance * 0.5)
            # score = torch.exp(norm_mean)
            # pred_possibility = score / torch.sum(score)
            pred_possibility = F.softmax(norm_mean, dim=0)
            targets_possibility = F.softmax(batch_targets, dim=0)

            # targets_possibility = torch.exp(batch_targets) / torch.sum(torch.exp(batch_targets))
            # total_score = torch.sum(score)
            CrossEntro_Loss = -targets_possibility * torch.log(pred_possibility)
            KLDiv_los = CrossEntro_Loss + targets_possibility * torch.log(targets_possibility)

            # uncertainty_loss = 0.5 * ((torch.log(targets_possibility)-torch.log(pred_possibility)) / norm_variance
            #                           + 0.5) ** 2 + torch.log(targets_possibility) + 0.5 * torch.log(norm_variance)
            uncertainty_loss = 0.5 * (torch.log(targets_possibility) -
                                      torch.log(pred_possibility)) ** 2 / norm_variance + 0.5 * torch.log(
                                      2 * 3.141592653 * norm_variance)

            penalty = torch.abs(norm_mean - batch_targets)

            # loss = torch.mean(0.1 * uncertainty_loss + KLDiv_los)
            # loss = torch.mean(uncertainty_loss + KLDiv_los + penalty)
            loss = torch.mean(torch.log(targets_possibility) + uncertainty_loss + penalty)
            # loss = torch.mean(targets_possibility * uncertainty_loss)

            losses += loss

        average_losses = losses / len(scope)

        return average_losses


"""
def KL_UQ(alpha):
    beta = tf.constant(np.ones((1, K)), dtype=tf.float32)
    S_alpha = tf.reduce_sum(alpha, axis=1, keep_dims=True)
    S_beta = tf.reduce_sum(beta, axis=1, keep_dims=True)
    lnB = tf.lgamma(S_alpha) - tf.reduce_sum(tf.lgamma(alpha), axis=1, keep_dims=True)
    lnB_uni = tf.reduce_sum(tf.lgamma(beta), axis=1, keep_dims=True) - tf.lgamma(S_beta)

    dg0 = tf.digamma(S_alpha)
    dg1 = tf.digamma(alpha)

    kl = tf.reduce_sum((alpha - beta) * (dg1 - dg0), axis=1, keep_dims=True) + lnB + lnB_uni
    return kl


def mse_loss(p, alpha, global_step, annealing_step):
    S = tf.reduce_sum(alpha, axis=1, keep_dims=True)
    E = alpha - 1
    m = alpha / S

    A = tf.reduce_sum((p - m) ** 2, axis=1, keep_dims=True)
    B = tf.reduce_sum(alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keep_dims=True)

    annealing_coef = tf.minimum(1.0, tf.cast(global_step / annealing_step, tf.float32))

    alp = E * (1 - p) + 1
    C = annealing_coef * KL(alp)
    return (A + B) + C"""
