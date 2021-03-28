import torch
from netgan.train import train
from netgan.model import Generator, Discriminator
from netgan import utils

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    ### load the data
    _A_obs, _X_obs, _z_obs = utils.load_npz("data/cora_ml.npz")

    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    lcc = utils.largest_connected_components(_A_obs)
    _A_obs = _A_obs[lcc, :][:, lcc]
    _N = _A_obs.shape[0]

    val_share = 0.1
    test_share = 0.05
    seed = 481516234

    train_graph = _A_obs
    assert (train_graph.toarray() == train_graph.toarray().T).all()

    train_ones, val_ones, val_zeros, test_ones, test_zeros = utils.train_val_test_split_adjacency(_A_obs, val_share,
                                                                                                  test_share, seed,
                                                                                                  undirected=True,
                                                                                                  connected=True,
                                                                                                  asserts=True)

    ### Parameter
    rw_len = 16
    batch_size = 128
    temperature = 5.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    walk = utils.RandomWalker(train_graph, rw_len, p=1, q=1, batch_size=batch_size)

    ### build generator and discriminator
    netG = Generator(_N, rw_len, tau=temperature, device=device).to(device)
    netD = Discriminator(_N, rw_len).to(device)

    ### define the stopping criterion
    stopping_criterion = "eo"
    assert stopping_criterion in ["val", "eo"], "Please set the desired stopping criterion."

    if stopping_criterion == "val":
        stopping = None
    if stopping_criterion == "eo":
        stopping = 0.5

    ### train model
    eval_every = 2000

    log_dict = train(netG, netD, _N, rw_len, val_ones, val_zeros, batch_size, walk.walk, _A_obs,
                    device=device, stopping=stopping, eval_every=eval_every, max_patience=20, max_iters=200000)
    print(log_dict.keys())

    plt.plot(np.arange(len(log_dict['val_performances'])) * eval_every,
             np.array(log_dict['val_performances'])[:, 0], label='ROC-AUC')
    plt.plot(np.arange(len(log_dict['val_performances'])) * eval_every,
             np.array(log_dict['val_performances'])[:, 1], label='Avg.Perc')
    plt.title('Validation')
    plt.legend()
    plt.show()