import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, N, rw_len, tau=1., generator_layers=[40], W_down_generator_size=128, noise_dim=16, noise_type="Gaussian", device="cpu"):
        super(Generator, self).__init__()
        self.N = N
        self.rw_len = rw_len
        self.device = device
        self.tau = tau

        self.W_down_generator_size = W_down_generator_size
        self.G_layers = generator_layers
        self.noise_dim = noise_dim
        self.noise_type = noise_type
        self.W_down_generator = nn.Parameter(torch.FloatTensor(self.N, W_down_generator_size))
        self.W_up = nn.Parameter(torch.FloatTensor(self.G_layers[-1], self.N))
        self.b_W_up = nn.Parameter(torch.FloatTensor(self.N))

        self.stacked_lstm = nn.LSTMCell(
            input_size=self.W_down_generator_size,
            hidden_size=self.G_layers[-1]
        )

        self.Linear1 = nn.ModuleList()
        self.Linear2 = nn.ModuleList()
        self.Linear3 = nn.ModuleList()
        for idx, size in enumerate(self.G_layers):
            self.Linear1.append(nn.Linear(self.noise_dim, size))
            self.Linear2.append(nn.Linear(size, size))
            self.Linear3.append(nn.Linear(size, size))

        self.weight_init()

    def weight_init(self):
        stdd = 1. / math.sqrt(self.W_down_generator_size)
        self.W_down_generator.data.uniform_(-stdd, stdd)
        stdu = 1. / math.sqrt(self.N)
        self.W_up.data.uniform_(-stdu, stdu)
        self.b_W_up.data.uniform_(-stdu, stdu)

    def forward(self, n_sample, z=None, gumbel=True, discrete=False):
        '''

        Args:
            n_sample: how many random walks to generate
            z:
            gumbel:
            discrete:

        Returns:
            some blocks of random walks, shape=[n_sample * rw_len * -1]
        '''
        if z is None:
            initial_state_noise = make_noise([n_sample, self.noise_dim], self.noise_type, device=self.device)
        else:
            initial_state_noise = z

        # initial_state = []
        for id, _ in enumerate(self.G_layers):
            intermediate = torch.tanh(self.Linear1[id](initial_state_noise))
            h = torch.tanh(self.Linear2[id](intermediate))
            c = torch.tanh(self.Linear3[id](intermediate))

            initial_state = (h, c)

        state = initial_state
        inputs = torch.zeros([n_sample, self.W_down_generator_size], dtype=torch.float32).to(self.device)
        outputs = []

        for i in range(self.rw_len):
            # Get LSTM outputs
            output, state = self.stacked_lstm(inputs, state)

            # Updata state
            state = (output, state)

            # Blow up to dimansion N using W_up
            output_bef = torch.matmul(output, self.W_up) + self.b_W_up

            # Perform Gumbel softmax to ensure gradients flow
            if gumbel:
                output = gumbel_softmax(output_bef, temperature=self.tau, hard=True, device=self.device)
            else:
                output = F.softmax(output_bef)

            # Back to dimansion d
            inputs = torch.matmul(output, self.W_down_generator)

            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)

        if discrete:
            outputs = torch.argmax(outputs, dim=-1)

        return outputs

class Discriminator(nn.Module):
    def __init__(self, N, rw_len, discriminator_layers=[30], W_down_discriminator_size=128):
        super(Discriminator, self).__init__()
        self.N = N
        self.rw_len = rw_len

        self.W_down_discriminator_size = W_down_discriminator_size
        self.D_layers = discriminator_layers
        self.W_down_discriminator = nn.Parameter(torch.FloatTensor(self.N, W_down_discriminator_size))
        # self.disc_lstm = nn.LSTMCell(W_down_discriminator_size, self.D_layers[-1])
        self.disc_lstm = nn.LSTMCell(
            input_size=W_down_discriminator_size,
            hidden_size=self.D_layers[-1]
        )
        self.disc_Linear = nn.Linear(self.D_layers[-1], 1)

        self.weight_init()

    def weight_init(self):
        stdd = 1. / math.sqrt(self.W_down_discriminator_size)
        self.W_down_discriminator.data.uniform_(-stdd, stdd)

    def forward(self, inputs):
        input_reshape = torch.reshape(inputs, [-1, self.N])
        output = torch.matmul(input_reshape, self.W_down_discriminator)
        output = torch.reshape(output, [self.rw_len, -1, self.W_down_discriminator.shape[-1]])

        output_disc = []
        for dim1 in range(output.shape[0]):
            outputs, _ = self.disc_lstm(output[dim1])
            output_disc.append(outputs)

        last_output = output_disc[-1]

        final_score = self.disc_Linear(last_output)

        return final_score


def make_noise(shape, type="Gaussian", device="cpu"):
    """
    Generate random noise
    Args:
        shape:
        type:

    Returns:

    """
    if type == "Gaussian":
        noise = torch.rand(shape).to(device)
    elif type == "Uniform":
        noise = torch.randn(shape).to(device)
    else:
        print("ERROR: Noise type {} not supported.".format(type))


    return noise

def sample_gumbel(shape, eps=1e-20, device="cpu"):
    """
    Sample from a uniform Gumbel distribution. Code by Eric Jang available at
    http://blog.evjang.com/2016/11/tutorial-categorical-variational.html
    Parameters
    ----------
    shape: Shape of the Gumbel noise
    eps: Epsilon for numerical stability.

    Returns
    -------
    Noise drawn from a uniform Gumbel distribution.

    """
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape, dtype=torch.float32).to(device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature, device="cpu"):
    """Draw a sample from the Gumbel-softmax distribution"""
    y = logits + sample_gumbel(logits.shape, device=device)
    return F.softmax(y / temperature, dim=1)

def gumbel_softmax(logits, temperature, hard=False, device="cpu"):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
      """
    y = gumbel_softmax_sample(logits, temperature, device=device)
    if hard:
        _, ind = y.max(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y, dtype=torch.float32)
        y_hard.scatter_(1, ind, 1.)
        y = (y_hard - y).detach() + y
    return y

