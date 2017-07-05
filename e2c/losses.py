import torch


def binary_crossentropy(t, o, eps=1e-8):
    return t * torch.log(o + eps) + (1.0 - t) * torch.log(1.0 - o + eps)


def kl_bernoulli(p, q, eps=1e-8):
    # http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/
    kl = p * torch.log((p + eps) / (q + eps)) + \
         (1 - p) * torch.log((1 - p + eps) / (1 - q + eps))
    return kl.mean()
