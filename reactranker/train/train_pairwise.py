import torch
import numpy as np
from ..data.load_reactions import get_time


def baseline_pairwise_training_loop(
    epoch, net, loss_func, optimizer, scheduler,
    smiles2graph_dic, train_loader, batch_size=100000,
    precision=torch.float32, gpu=None,
):
    minibatch_loss = []
    minibatch = 0
    count = 0

    for x_i, y_i, x_j, y_j in train_loader.generate_query_pair_batch(batch_size):
        if x_i is None or x_i.shape[0] == 0:
            continue
        rsmi1 = [s[0] for s in x_i]
        psmi1 = [s[1] for s in x_i]
        rsmi2 = [s[0] for s in x_j]
        psmi2 = [s[1] for s in x_j]
        # this function should be deleted if the user wanna compare reactions with different reactants
        assert rsmi1 == rsmi2
        r = smiles2graph_dic.parsing_smiles(rsmi1)
        p_i = smiles2graph_dic.parsing_smiles(psmi1)
        p_j = smiles2graph_dic.parsing_smiles(psmi2)
        # binary label
        y = torch.tensor((y_i > y_j).astype(np.float32), dtype=precision)
        if gpu is not None:
            y.cuda(gpu)

        net.zero_grad()
        y_pred = net(r, p_i, p_j, gpu=gpu)
        loss = loss_func(y_pred, y)

        loss.backward()
        optimizer.step()
        scheduler.step()

        minibatch_loss.append(loss.item())

        minibatch += 1
        if minibatch % 100 == 0:
            print(get_time(), 'Epoch {}, Minibatch: {}, loss : {}'.format(epoch, minibatch, loss.item()))

    return np.mean(minibatch_loss)


def factorized_training_loop(epoch, model, loss_func, optimizer, scheduler,
                             smiles2graph_dic, train_data_processor, batch_size=2, sigma=1.0,
                             training_algo='sum_session', gpu=None):

    minibatch_loss = []
    count, loss, pairs = 0, 0, 0
    grad_batch, y_pred_batch = [], []
    for X, Y in train_data_processor.generate_batch_per_query(seed=epoch):
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
                        smiles2graph_dic, train_data_processor, batch_size=2, sigma=1.0,
                        training_algo='beta_dis', gpu=None):

    minibatch_loss = []
    count, loss, pairs = 0, 0, 0
    grad_batch, y_pred_batch = [], []
    for reactions, targets in train_data_processor.generate_batch_per_query(seed=epoch):
        if reactions is None or reactions.shape[0] == 0:
            continue
        rsmi = [s[0] for s in reactions]
        psmi = [s[1] for s in reactions]
        r_batch = smiles2graph_dic.parsing_smiles(rsmi)
        p_batch = smiles2graph_dic.parsing_smiles(psmi)
        targets = targets.reshape(-1, 1)
        rel_diff = targets - targets.T
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

