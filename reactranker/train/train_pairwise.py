import torch
import numpy as np
from ..data.load_reactions import get_time


def baseline_pairwise_training_loop(
    epoch, epochs, model, optimizer, scheduler,
    smiles2graph_dic, train_loader, batch_size=1000,
    max_coeff=0.01, gpu=None, target_name='ea',
):
    batch_size = batch_size
    minibatch_loss = []
    minibatch = 0
    count = 0
    for x_i, y_i, x_j, y_j in train_loader.generate_query_pair_batch(targ=target_name,
                                                                     batchsize=batch_size, seed=epoch):
        if x_i is None or x_i.shape[0] == 0:
            continue
        rsmi1 = [s[0] for s in x_i]
        psmi1 = [s[1] for s in x_i]
        rsmi2 = [s[0] for s in x_j]
        psmi2 = [s[1] for s in x_j]
        # this function should be deleted if the user wanna compare reactions with different reactants
        assert rsmi1 == rsmi2
        if len(rsmi1) < batch_size:
            continue
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
        y_pred = model(r, p_i, p_j, gpu=gpu)


        # Calculating loss
        pred_S = torch.sum(y_pred, dim=1, keepdim=True)
        pred_p = y_pred/pred_S
        err = torch.sum((target_p - pred_p)**2, dim=1, keepdim=True)
        var = torch.sum(pred_p*(1-pred_p)/(pred_S + 1), dim=1, keepdim=True)
        # Calculating Penalty Loss
        new_alpha = y_pred * torch.sum(target_p * torch.log(target_p/pred_p), dim=1, keepdim=True) + 1
        beta = torch.ones([batch_size, 2])
        if gpu is not None:
            beta = beta.cuda(gpu)
        S_alpha = torch.sum(new_alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(new_alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(new_alpha)
        kl = torch.sum((new_alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        annealing_coef = max_coeff * (epoch / (epochs - 1)) ** 3
        loss = torch.mean(err)  #   + var)  # + kl*annealing_coef)

        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()


        minibatch_loss.append(loss.item())

        minibatch += 1
        if minibatch % 50 == 0:
            print(get_time(), 'Epoch {}, Minibatch: {}, loss : {}'.format(epoch, minibatch, loss.item()))
            print('Pred in train process: ', pred_p[:20])
            print('Target in train process: ', target_p[:20])


    return np.mean(minibatch_loss)


def factorized_training_loop(epoch, model, loss_func, optimizer, scheduler,
                             smiles2graph_dic, train_data_processor, batch_size=2, sigma=1.0,
                             training_algo='sum_session', gpu=None, smiles_list=None,
                             target_name: str = 'ea'):

    minibatch_loss = []
    count, loss, pairs = 0, 0, 0
    grad_batch, y_pred_batch = [], []
    for X, Y in train_data_processor.generate_batch_per_query(smiles_list=smiles_list,
                                                              target_name=target_name, seed=epoch):
        if X is None or X.shape[0] == 0:
            continue
        rsmi = [s[0] for s in X]
        psmi = [s[1] for s in X]
        r_batch = smiles2graph_dic.parsing_smiles(rsmi)
        p_batch = smiles2graph_dic.parsing_smiles(psmi)
        Y = Y.reshape(-1, 1)
        rel_diff = Y - Y.T
        pos_pairs = (rel_diff > 0).astype(np.float32)
        num_pos_pairs = np.sum(pos_pairs, (0, 1))
        # skip negative sessions, no relevant info:
        if num_pos_pairs == 0:
            continue
        neg_pairs = (rel_diff < 0).astype(np.float32)
        num_pairs = 2 * num_pos_pairs  # num pos pairs and neg pairs are always the same

        pos_pairs = torch.FloatTensor(pos_pairs)
        neg_pairs = torch.FloatTensor(neg_pairs)
        if gpu is not None:
            pos_pairs = pos_pairs.cuda(gpu)
            neg_pairs = neg_pairs.cuda(gpu)

        y_pred = model(r_batch, p_batch, gpu=gpu)
        if len(y_pred.size()) > 1:
            y_pred = y_pred[:, 0]
        y_pred = y_pred.unsqueeze(1)
        if training_algo == 'sum_session':
            C_pos = torch.log(1 + torch.exp(-sigma * (y_pred - y_pred.t())))
            C_neg = torch.log(1 + torch.exp(sigma * (y_pred - y_pred.t())))
            C = pos_pairs * C_pos + neg_pairs * C_neg
            loss += torch.sum(C, (0, 1))
        elif training_algo == 'accelerate_grad':
            y_pred_batch.append(y_pred)
            with torch.no_grad():
                l_pos = 1 + torch.exp(sigma * (y_pred - y_pred.t()))
                l_neg = 1 + torch.exp(- sigma * (y_pred - y_pred.t()))
                batch_loss = -sigma * pos_pairs / l_pos + sigma * neg_pairs / l_neg
                loss += torch.sum(
                    torch.log(l_neg) * pos_pairs + torch.log(l_pos) * neg_pairs,
                    (0, 1)
                )
                back = torch.sum(batch_loss, dim=1, keepdim=True)

                if torch.sum(back, dim=(0, 1)) == float('inf') or back.shape != y_pred.shape:
                    import ipdb; ipdb.set_trace()
                grad_batch.append(back)
        else:
            raise ValueError("training algo {} not implemented".format(training_algo))

        pairs += num_pairs
        count += 1
        # This is a trick named gradient accumulation
        if count % batch_size == 0:
            loss /= pairs
            minibatch_loss.append(loss.item())
            if training_algo == 'sum_session':
                loss.backward()
            elif training_algo == 'accelerate_grad':
                for grad, y_pred in zip(grad_batch, y_pred_batch):
                    y_pred.backward(grad / pairs)

            optimizer.step()
            model.zero_grad()
            scheduler.step()
            loss, pairs = 0, 0  # loss used for sum_session
            grad_batch, y_pred_batch = [], []  # grad_batch, y_pred_batch used for gradient_acc

    if pairs:
        print('+' * 10, "End of batch, remaining pairs {}".format(pairs.item()))
        loss /= pairs
        minibatch_loss.append(loss.item())
        if training_algo == 'sum_session':
            loss.backward()
        else:
            for grad, y_pred in zip(grad_batch, y_pred_batch):
                y_pred.backward(grad / pairs)
        optimizer.step()

    return np.mean(minibatch_loss)


def beta_dis_train_loop(epoch, model, loss_func, optimizer, scheduler,
                        smiles2graph_dic, train_data_processor, batch_size=2, alpha0=100,
                        training_algo='beta_dis', gpu=None, smiles_list=None, target_name='ea'):

    minibatch_loss = []
    count, loss, pairs = 0, 0, 0
    grad_batch, y_pred_batch = [], []
    for reactions, targets in train_data_processor.generate_batch_per_query(smiles_list=smiles_list,
                                                                            target_name=target_name, seed=epoch):
        if reactions is None or reactions.shape[0] == 0:
            continue
        targets = torch.FloatTensor(targets)
        if gpu is not None:
            targets = targets.cuda(gpu)
        targets = torch.sigmoid(targets)
        num_pairs = targets.size()[0]**2 - targets.size()[0]
        if gpu is not None:
            alpha_ini = torch.ones(targets.size()).unsqueeze(1).cuda(gpu) * targets
        else:
            alpha_ini = torch.ones(targets.size()).unsqueeze(1) * targets
        beta_ini = alpha_ini.t()
        std_alpha = alpha_ini/(alpha_ini + beta_ini)
        std_beta = beta_ini/(alpha_ini + beta_ini)
        targ_alpha = std_alpha * alpha0
        targ_beta = std_beta * alpha0
        targ_x1 = std_alpha
        targ_x2 = std_beta

        rsmi = [s[0] for s in reactions]
        psmi = [s[1] for s in reactions]
        r_batch = smiles2graph_dic.parsing_smiles(rsmi)
        p_batch = smiles2graph_dic.parsing_smiles(psmi)
        pred = model(r_batch, p_batch, gpu=gpu)
        pred = torch.sigmoid(pred)
        if gpu is not None:
            pred_alpha_ini = torch.ones(targets.size()).unsqueeze(1).cuda(gpu) * pred
        else:
            pred_alpha_ini = torch.ones(targets.size()).unsqueeze(1) * pred
        pred_beta_ini = pred_alpha_ini.t()
        pred_std_alpha = pred_alpha_ini / (pred_alpha_ini + pred_beta_ini)
        pred_std_beta = pred_beta_ini / (pred_alpha_ini + pred_beta_ini)
        pred_alpha = pred_std_alpha * alpha0
        pred_beta = pred_std_beta * alpha0

        # Calculating Beta function and Loss
        ln_B_target = torch.lgamma(targ_alpha) + torch.lgamma(targ_beta) - torch.lgamma(targ_alpha + targ_beta)
        ln_beta_pdf_target = (targ_alpha - 1) * torch.log(targ_x1) + (targ_beta - 1) * torch.log(targ_x2) - ln_B_target
        ln_B_pred = torch.lgamma(pred_alpha) + torch.lgamma(pred_beta) - torch.lgamma(pred_alpha + pred_beta)
        ln_beta_pdf_pred = (pred_alpha - 1) * torch.log(targ_x1) + (pred_beta - 1) * torch.log(targ_x2) - ln_B_pred
        kl_loss = torch.exp(ln_beta_pdf_target) * (ln_beta_pdf_target - ln_beta_pdf_pred)
        loss += torch.sum(kl_loss, (0, 1))
        """
        print('targ_alpha is: ', targ_alpha)
        print('targ_beta is: ', targ_beta)
        print('pred_alpha: ', pred_alpha)
        print('pred_beta: ', pred_beta)
        print('ln_B_target is: ', ln_B_target)
        print('ln_B_pred is: ', ln_B_pred)
        print('ln_beta_pdf_target is: ', ln_beta_pdf_target)
        print('ln_beta_pdf_pred: ', ln_beta_pdf_pred)
        print(torch.exp(ln_beta_pdf_target) * (ln_beta_pdf_pred - ln_beta_pdf_pred))
        
        print('kl_loss is: ', kl_loss)
        print(torch.sum(kl_loss, (0, 1)))
        print('loss is: ', loss)
        """


        pairs += num_pairs
        count += 1
        # This is a trick named gradient accumulation
        if count % batch_size == 0:
            loss /= pairs
            minibatch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            model.zero_grad()
            scheduler.step()
            loss, pairs = 0, 0  # loss used for sum_session
            grad_batch, y_pred_batch = [], []  # grad_batch, y_pred_batch used for gradient_acc

    if pairs:
        print('+' * 10, "End of batch, remaining pairs {}".format(pairs.item()))
        loss /= pairs
        minibatch_loss.append(loss.item())
        if training_algo == 'sum_session':
            loss.backward()
        optimizer.step()

    return np.mean(minibatch_loss)


def beta_evi_train_loop(epoch, model, optimizer, scheduler,
                        smiles2graph_dic, train_data_processor, batch_size=2, max_coeff=0.001, epochs=100,
                        training_algo='beta_dis', gpu=None, smiles_list=None, target_name='ea'):
    minibatch_loss = []
    count, loss, pairs = 0, 0, 0
    for reactions, targets in train_data_processor.generate_batch_per_query(smiles_list=smiles_list,
                                                                            target_name=target_name, seed=epoch):
        if reactions is None or reactions.shape[0] == 0:
            continue
        targets = torch.FloatTensor(targets)
        if gpu is not None:
            targets = targets.cuda(gpu)
        targets = torch.sigmoid(targets)
        num_pairs = targets.size()[0] ** 2 - targets.size()[0]
        if gpu is not None:
            alpha_ini = torch.ones(targets.size()).unsqueeze(1).cuda(gpu) * targets
        else:
            alpha_ini = torch.ones(targets.size()).unsqueeze(1) * targets
        beta_ini = alpha_ini.t()
        targ_p1 = alpha_ini / (alpha_ini + beta_ini)
        targ_p2 = beta_ini / (alpha_ini + beta_ini)

        # Generating preds
        rsmi = [s[0] for s in reactions]
        psmi = [s[1] for s in reactions]
        r_batch = smiles2graph_dic.parsing_smiles(rsmi)
        p_batch = smiles2graph_dic.parsing_smiles(psmi)
        pred = model(r_batch, p_batch, gpu=gpu)
        if gpu is not None:
            pred_alpha_ini = torch.ones(targets.size()).unsqueeze(1).cuda(gpu) * pred
        else:
            pred_alpha_ini = torch.ones(targets.size()).unsqueeze(1) * pred
        pred_beta_ini = pred_alpha_ini.t()
        pred_p1 = pred_alpha_ini / (pred_alpha_ini + pred_beta_ini)
        pred_p2 = pred_beta_ini / (pred_alpha_ini + pred_beta_ini)

        # Calculating Loss
        err = (targ_p1 - pred_p1)**2 + (targ_p2 - pred_p2)**2
        var = pred_p1 * (1-pred_p1) / (pred_alpha_ini + pred_beta_ini + 1) + \
              pred_p2 * (1-pred_p2) / (pred_alpha_ini + pred_beta_ini + 1)
        consist1 = torch.log(targ_p1 / pred_p1)
        residue1 = consist1 * (pred_alpha_ini - 1)
        penalty_loss1 = torch.abs(residue1)
        consist2 = torch.log(targ_p1 / pred_p1)
        residue2 = consist2 * (pred_alpha_ini - 1)
        penalty_loss2 = torch.abs(residue2)
        annealing_coef = max_coeff * (epoch / (epochs - 1)) ** 3
        loss += torch.sum(err + var + annealing_coef * (penalty_loss1 + penalty_loss2), (0, 1))

        pairs += num_pairs
        count += 1
        # This is a trick named gradient accumulation
        if count % batch_size == 0:
            loss /= pairs
            minibatch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            model.zero_grad()
            scheduler.step()
            loss, pairs = 0, 0  # loss used for sum_session
            grad_batch, y_pred_batch = [], []  # grad_batch, y_pred_batch used for gradient_acc

    if pairs:
        print('+' * 10, "End of batch, remaining pairs {}".format(pairs.item()))
        loss /= pairs
        minibatch_loss.append(loss.item())
        if training_algo == 'sum_session':
            loss.backward()
        optimizer.step()

    return np.mean(minibatch_loss)

