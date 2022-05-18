# shared utility functions

import torch
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, mse_loss, kl_div
import torchio as tio


class RandomCrop:
    def __init__(self, size=(256, 256, 256)):
        self.size = size

    def __call__(self, sample):
        sampler = tio.data.UniformSampler(patch_size=self.size)
        patch = list(sampler(sample, 1))[0]
        return patch


def weighted_bce_logit(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):
    '''
    Return weighted sum of `torch.nn.BCEWithLogitsLoss` values for the outcome variables.
    '''
    # print(pred.shape, target.shape)
    # target = torch.as_tensor(target, dtype=torch.float32)
    # severe_loss = binary_cross_entropy_with_logits(pred[:, 0], target[:, 0])
    # covid_loss = binary_cross_entropy_with_logits(pred[:, 1], target[:, 1])
    # stacked = torch.stack((severe_loss, covid_loss))
    # return torch.logsumexp(stacked + torch.log(weight), 0)
    return binary_cross_entropy_with_logits(pred, target.float(), pos_weight=weight)


def weighted_cross_entropy(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):
    '''
    Wrapper for `torch.nn.CrossEntropyLoss`.
    '''
    return cross_entropy(pred, target.float(), weight=weight)


def reconstruction_loss(x, x_hat):
    loss = mse_loss(x, x_hat, reduction="none")
    loss = loss.sum(dim=[1, 2, 3, 4]).mean(dim=[0])
    return loss


def vae_loss(x, x_hat, mu, logvar, scale_regular=100):
    '''
    Loss function for the variational autoencoder.
    '''
    mse = reconstruction_loss(x, x_hat)
    kld = scale_regular * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0])
    return mse + kld, mse, kld


def group_wise_lr(model, group_lr_conf: dict, path=""):
    """
    Refer https://pytorch.org/docs/master/optim.html#per-parameter-options

    torch.optim.SGD([
        {'params': model.base.parameters()},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], lr=1e-2, momentum=0.9)

    to

    cfg = {"classifier": {"lr": 1e-3},
           "lr":1e-2, "momentum"=0.9}
    confs, names = group_wise_lr(model, cfg)
    torch.optim.SGD([confs], lr=1e-2, momentum=0.9)

    :param model:
    :param group_lr_conf:
    :return:
    """
    assert type(group_lr_conf) == dict
    confs = []
    nms = []
    for kl, vl in group_lr_conf.items():
        assert type(kl) == str
        assert type(vl) == dict or type(vl) == float or type(vl) == int

        if type(vl) == dict:
            print(kl)
            assert hasattr(model, kl)
            cfs, names = group_wise_lr(getattr(model, kl), vl, path=path + kl + ".")
            confs.extend(cfs)
            names = list(map(lambda n: kl + "." + n, names))
            nms.extend(names)

    primitives = {kk: vk for kk, vk in group_lr_conf.items() if type(vk) == float or type(vk) == int}
    remaining_params = [(k, p) for k, p in model.named_parameters() if k not in nms]
    if len(remaining_params) > 0:
        names, params = zip(*remaining_params)
        conf = dict(params=params, **primitives)
        confs.append(conf)
        nms.extend(names)

    plen = sum([len(list(c["params"])) for c in confs])
    assert len(list(model.parameters())) == plen
    assert set(list(zip(*model.named_parameters()))[0]) == set(nms)
    assert plen == len(nms)
    if path == "":
        for c in confs:
            c["params"] = (n for n in c["params"])
    return confs, nms
