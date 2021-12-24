import torch
from torch import nn
from torch.nn import functional


def regularization(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())


class VAE(nn.Module):
    # Define the initialization function，which defines the basic structure of the neural network
    def __init__(self, args):
        super(VAE, self).__init__()
        self.l = len(args.layer)
        self.L = args.L
        self.device = args.device
        self.inet = nn.ModuleList()
        darray = [args.d] + args.layer
        for i in range(self.l - 1):
            self.inet.append(nn.Linear(darray[i], darray[i + 1]))
        self.mu = nn.Linear(darray[self.l - 1], darray[self.l])
        self.sigma = nn.Linear(darray[self.l - 1], darray[self.l])
        self.gnet = nn.ModuleList()
        for i in range(self.l):
            self.gnet.append(nn.Linear(darray[self.l - i], darray[self.l - i - 1]))

    def encode(self, x):
        h = x
        for i in range(self.l - 1):
            # h = functional.relu(self.inet[i](h))
            h = functional.relu(functional.dropout(self.inet[i](h), p=0.3, training=True))
        return self.mu(h), self.sigma(h)

    def decode(self, z):
        h = z
        for i in range(self.l - 1):
            # h = functional.relu(self.gnet[i](h))
            h = functional.relu(functional.dropout(self.gnet[i](h), p=0.3, training=True))
        return functional.sigmoid(self.gnet[self.l - 1](h))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn([self.L] + list(std.shape)).to(self.device)
            return eps.mul(std).add_(mu)
        else:
            return mu

    # Define the forward propagation function for the neural network.
    # Once defined, the backward propagation function will be autogeneration（autograd）
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



