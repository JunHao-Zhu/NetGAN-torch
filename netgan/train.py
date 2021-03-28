import os
import time
import numpy as np
import scipy.sparse as sp

import torch
from sklearn.metrics import roc_auc_score, average_precision_score

import utils



def onehot_code(mat, N):
    '''
    discrete form of random walk
    Args:
        mat:
        N: number of nodes

    Returns:

    '''
    stack_mat = []
    for idx in range(mat.shape[0]):
        onehot = np.eye(N)[mat[idx]]
        stack_mat.append(onehot)
    OneHot_mat = np.stack(stack_mat, axis=0)

    return OneHot_mat

def Gen_loss(netG, D_fake, l2_penalty_generator=1e-7):

    gen_cost = -D_fake

    # weight regularization; we omit  W_down from regularization
    gen_l2_loss = 0
    for gen_k, gen_v in netG.state_dict().items():
        if "Linear" in gen_k or "W_up" in gen_k:
            l2_loss = torch.sum(gen_v ** 2) / 2
            torch.add(gen_l2_loss, l2_loss)
    gen_cost += gen_l2_loss * l2_penalty_generator

    return gen_cost


def Disc_loss(netD, real_data, fake_data, batch_size=128, wasserstein_penalty=10, l2_penalty_discriminator=5e-5, device="cpu"):
    D_fake = netD(fake_data).mean()
    D_real = netD(real_data).mean()
    disc_cost = D_fake - D_real

    alpha = torch.rand(size=[batch_size, 1, 1]).to(device)
    difference = fake_data - real_data
    interpolates = alpha * real_data + ((1 - alpha) * difference)
    interpolates.requires_grad = True

    disc_interpolates = netD(interpolates)
    disc_interpolates.sum().backward()
    gradients = interpolates.grad

    slopes = torch.sqrt(torch.square(gradients).sum(dim=0))
    gradients_penalty = torch.square((slopes - 1.)).mean() * wasserstein_penalty
    disc_cost += gradients_penalty

    # weight regularization; we omit W_down from regularization
    disc_l2_loss = 0
    for disc_k, disc_v in netD.state_dict().items():
        if "Linear" in disc_k:
            l2_loss = torch.sum(disc_v ** 2) / 2
            torch.add(disc_l2_loss, l2_loss)
    disc_cost += disc_l2_loss * l2_penalty_discriminator

    return disc_cost


def train(netG, netD, N, rw_len, val_ones, val_zeros, n_sample, Walker, A_orig, device="cpu", max_iters=50000,
          stopping=None, eval_transitions=15e4, transitions_per_iter=1500, max_patience=5, eval_every=500,
          learning_rate=0.0003, disc_iters=3, temp_start=5.0, temperature_decay=1-5e-5, min_temperature=0.5):

    if stopping == None:
        best_performance = 0.0
        patience = max_patience
        print("**** Using VAL criterion for early stopping ****")
    else:
        assert "float" in str(type(stopping)) and  stopping > 0 and stopping <= 1
        print("**** Using EO criterion of {} for early stopping ****".format(stopping))

    # Validation labels
    actual_labels_val = np.append(np.ones(len(val_ones)), np.zeros(len(val_zeros)))

    # Some lists to store data into
    gen_losses = []
    disc_losses = []
    graphs = []
    val_performance = []
    eo = []

    start_time = time.time()

    transitions_per_walk = rw_len - 1
    # Sample lots of random walks, used for evaluation of model.
    sample_many_count = int(np.round(transitions_per_iter / transitions_per_walk))
    n_eval_walks = eval_transitions / transitions_per_walk
    n_eval_iters = int(np.round(n_eval_walks / sample_many_count))

    optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.9))

    print("**** start training ****")
    for _it in range(max_iters):
        if _it > 0 and _it%(2500) == 0:
            t = time.time() - start_time
            print("{:<7}/{:<8} training iterations, took {} seconds so far...".format(_it, max_iters, int(t)))

        # Generator training
        for p in netD.parameters():
            p.requires_grad = False
        optimizerG.zero_grad()

        fake = netG(n_sample)
        G = netD(fake)
        G = G.mean()
        G_loss = Gen_loss(netG, G)
        G_loss.backward()
        optimizerG.step()

        for p in netD.parameters():
            p.requires_grad = True
        # Discriminator training
        _disc_l = []
        for iter_d in range(disc_iters):
            real_inputs = Walker().__next__()
            optimizerD.zero_grad()

            # train with real
            real_inputs = onehot_code(real_inputs, N)
            real_inputs = torch.FloatTensor(real_inputs).to(device)

            # train with fake
            with torch.no_grad():
                fake = netG(n_sample)

            # train with gradient penalty
            D_loss = Disc_loss(netD, real_inputs, fake, batch_size=n_sample, device=device)
            D_loss.backward()

            optimizerD.step()
            print(D_loss)

            _disc_l.append(D_loss.detach().cpu().numpy())

        gen_losses.append(G_loss.detach().cpu().numpy())
        disc_losses.append(np.mean(_disc_l))

        if _it > 0 and _it % eval_every == 0:
            # Sample lots of random walks
            smpls = []
            for _ in range(n_eval_iters):
                sample_many = netG(sample_many_count, discrete=True)
                smpls.append(sample_many.detach().cpu().numpy())

            # Compute score matrix
            gr = utils.score_matrix_from_random_walks(np.array(smpls).reshape([-1, rw_len]), N)
            gr = gr.tocsr()

            # Assemble a graph from the score matrix
            _graph = utils.graph_from_scores(gr, A_orig.sum())
            # Compute edge overlap
            edge_overlap = utils.edge_overlap(A_orig.toarray(), _graph)
            graphs.append(_graph)
            eo.append(edge_overlap)

            edge_scores = np.append(gr[tuple(val_ones.T)].A1, gr[tuple(val_zeros.T)].A1)

            # Compute validation ROC_AUC and average precision scores
            val_performance.append((roc_auc_score(actual_labels_val, edge_scores),
                                   average_precision_score(actual_labels_val, edge_scores)))

            # Update Gumbel temperature
            temperature= np.maximum(temp_start * np.exp(-(1-temperature_decay)*_it),
                                    min_temperature)
            netG.tau = temperature

            print("**** Iter {:<6} Val ROC {:.3f}, AP: {:.3f}, EO {:.3f} ****".format(_it, val_performance[-1][0],
                                                                                      val_performance[-1][1],
                                                                                      edge_overlap/A_orig.sum()))

            if stopping is None:
                if np.sum(val_performance[-1]) > best_performance:
                    best_performance = np.sum(val_performance[-1])
                    patience = max_patience
                else:
                    patience = -1

                if patience == 0:
                    print("**** EARLY STOPPING AFTER {} ITERATIONS ****".format(_it))
                    break
            if edge_overlap/A_orig.sum() >= stopping:
                print("**** EARLY STOPPing AFTER {} ITERATIONS ****".format(_it))
                break

    print("**** Training completed after {} iterations. ****".format(_it))

    log_dict = {"disc_losses": disc_losses, 'gen_losses': gen_losses, 'val_performances': val_performance,
                'edge_overlaps': eo, 'generated_graphs': graphs}

    return log_dict
