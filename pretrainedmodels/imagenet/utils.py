from functools import partial
from math import pi, cos

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms

from .transforms import ColorJitter, Lighting


def create_optimizer(optimizer_config, model, niters=0):
    """Creates optimizer and schedule from configuration

    Parameters
    ----------
    optimizer_config : dict
        Dictionary containing the configuration options for the optimizer.
    model : Model
        The network model.
    niters: int
        A epoch include niters times iters.
    Returns
    -------
    optimizer : Optimizer
        The optimizer.
    scheduler : LRScheduler
        The learning rate scheduler.
    """
    if optimizer_config["classifier_lr"] != -1:
        # Separate classifier parameters from all others
        net_params = []
        classifier_params = []
        for k, v in model.named_parameters():
            if k.find("fc") != -1 or k.find("last") != -1:
                classifier_params.append(v)
            else:
                net_params.append(v)
        params = [
            {"params": net_params},
            {"params": classifier_params,
                "lr": optimizer_config["classifier_lr"]},
        ]
    else:
        params = model.parameters()

    # params = filter(lambda p: p.requires_grad, params)
    if optimizer_config["freeze"] == 1 and optimizer_config["classifier_lr"] != -1:
        params = classifier_params
        lr = optimizer_config["classifier_lr"]
        # print("Using freeze!")
    else:
        lr = optimizer_config["learning_rate"]
        # print("Not using freeze!")

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(params,
                              lr=lr,
                              momentum=optimizer_config["momentum"],
                              weight_decay=optimizer_config["weight_decay"],
                              nesterov=optimizer_config["nesterov"])
    elif optimizer_config["type"] == "Adam":
        optimizer = optim.Adam(params,
                               lr=optimizer_config["learning_rate"],
                               weight_decay=optimizer_config["weight_decay"])
    else:
        raise KeyError("unrecognized optimizer {}".format(
            optimizer_config["type"]))

    if optimizer_config["schedule"]["mode"] == "step":
        if optimizer_config["schedule"]["type"] == "linear":
            def linear_lr(it):
                return it * optimizer_config["schedule"]["params"]["alpha"] + optimizer_config["schedule"]["params"][
                    "beta"]
            scheduler = lr_scheduler.LambdaLR(optimizer, linear_lr)

        elif optimizer_config["schedule"]["type"] == "warmup_step":
            def linear_lr(it):
                it += 1
                epoch = it // niters
                if epoch == 0:
                    return 0.1 / optimizer_config["learning_rate"]
                elif epoch >= 1 and epoch <= 5:
                    it = it - niters
                    return (0.1 + (optimizer_config["learning_rate"] - 0.1) *
                            (it / (5 * niters))) / optimizer_config["learning_rate"]
                else:
                    return 0.1 ** (epoch // 30)

            scheduler = lr_scheduler.LambdaLR(optimizer, linear_lr)

        elif optimizer_config["schedule"]["type"] == "warmup_liner":
            def warmup_liner_lr(it):
                it += 1
                epoch = it // niters
                if epoch == 0:
                    return 0.1 / optimizer_config["learning_rate"]
                elif epoch >= 1 and epoch <= 5:
                    it = it - niters
                    return (0.1 + (optimizer_config["learning_rate"] - 0.1) *
                            (it / (5 * niters))) / optimizer_config["learning_rate"]
                else:
                    it = it - 6 * niters
                    return 1 - (it / (niters * (optimizer_config["schedule"]["epochs"] - 6)))

            scheduler = lr_scheduler.LambdaLR(optimizer, warmup_liner_lr)

        elif optimizer_config["schedule"]["type"] == "warmup_cos":
            def linear_lr(it):
                it += 1
                epoch = it // niters
                if epoch == 0:
                    return 0.1 / optimizer_config["learning_rate"]
                elif epoch >= 1 and epoch <= 5:
                    it = it - niters
                    return (0.1 + (optimizer_config["learning_rate"] - 0.1) *
                            (it / (5 * niters))) / optimizer_config["learning_rate"]
                else:
                    it = it - 6 * niters
                    return (1 + cos(pi * (it / (niters * (optimizer_config["schedule"]["epochs"] - 6))))) / 2

            scheduler = lr_scheduler.LambdaLR(optimizer, linear_lr)

    else:
        if optimizer_config["schedule"]["type"] == "step":
            scheduler = lr_scheduler.StepLR(
                optimizer, **optimizer_config["schedule"]["params"])
        elif optimizer_config["schedule"]["type"] == "multistep":
            scheduler = lr_scheduler.MultiStepLR(
                optimizer, **optimizer_config["schedule"]["params"])
        elif optimizer_config["schedule"]["type"] == "exponential":
            scheduler = lr_scheduler.ExponentialLR(
                optimizer, **optimizer_config["schedule"]["params"])
        elif optimizer_config["schedule"]["type"] == "constant":
            scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
        elif optimizer_config["schedule"]["type"] == "warmup":
            def warmup_lr(it):
                if it >= 0 and it <= 5:
                    return ((optimizer_config["learning_rate"] - 0.1) * it + 0.5) / (
                        5 * optimizer_config["learning_rate"])
                else:
                    return 0.1 ** (it // 30)

            scheduler = lr_scheduler.LambdaLR(optimizer, warmup_lr)
        elif optimizer_config["schedule"]["type"] == "warmup_liner":
            def warmup_liner_lr(it):
                if it >= 0 and it <= 5:
                    return ((optimizer_config["learning_rate"] - 0.1) * it + 0.5) / (
                        5 * optimizer_config["learning_rate"])
                else:
                    return 1 - (it / (optimizer_config["schedule"]["epochs"] - 5))

            scheduler = lr_scheduler.LambdaLR(optimizer, warmup_liner_lr)

    return optimizer, scheduler


def create_transforms(input_config):
    """Create transforms from configuration

    Parameters
    ----------
    input_config : dict
        Dictionary containing the configuration options for input pre-processing.

    Returns
    -------
    train_transforms : list
        List of transforms to be applied to the input during training.
    val_transforms : list
        List of transforms to be applied to the input during validation.
    """
    normalize = transforms.Normalize(
        mean=input_config["mean"], std=input_config["std"])

    train_transforms = []
    if input_config["scale_train"] != -1:
        train_transforms.append(transforms.Resize(input_config["scale_train"]))
    # https://github.com/pytorch/examples/issues/355
    train_transforms += [
        transforms.RandomResizedCrop(
            input_config["crop_train"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    if input_config["color_jitter_train"]:
        train_transforms.append(ColorJitter())
    if input_config["lighting_train"]:
        train_transforms.append(Lighting())
    train_transforms.append(normalize)

    val_transforms = []
    if input_config["scale_val"] != -1:
        val_transforms.append(transforms.Resize(input_config["scale_val"]))
    val_transforms += [
        transforms.CenterCrop(input_config["crop_val"]),
        transforms.ToTensor(),
        normalize,
    ]

    return train_transforms, val_transforms


def get_model_params(network_config):
    """Convert a configuration to actual model parameters

    Parameters
    ----------
    network_config : dict
        Dictionary containing the configuration options for the network.

    Returns
    -------
    model_params : dict
        Dictionary containing the actual parameters to be passed to the `net_*` functions in `models`.
    """
    model_params = {}

    # model_params["classes"] = network_config["classes"]

    return model_params


class MultiCropEnsemble(nn.Module):
    def __init__(self, module, cropsize, act=nn.functional.softmax, flipping=True):
        super(MultiCropEnsemble, self).__init__()
        self.cropsize = cropsize
        self.flipping = flipping
        self.internal_module = module
        self.act = act

    # Naive code
    def forward(self, x):
        # H, W >= cropsize
        assert(x.size()[2] >= self.cropsize)
        assert(x.size()[3] >= self.cropsize)

        cs = self.cropsize
        x1 = 0
        x2 = x.size()[2] - self.cropsize
        cx = x.size()[2] // 2 - self.cropsize // 2
        y1 = 0
        y2 = x.size()[3] - self.cropsize
        cy = x.size()[3] // 2 - self.cropsize // 2

        def get_output(x): return self.act(self.internal_module.forward(x))

        _y = get_output(x[:, :, x1: x1 + cs, y1: y1 + cs])
        _y = get_output(x[:, :, x1: x1 + cs, y2: y2 + cs]) + _y
        _y = get_output(x[:, :, x2: x2 + cs, y1: y1 + cs]) + _y
        _y = get_output(x[:, :, x2: x2 + cs, y2: y2 + cs]) + _y
        _y = get_output(x[:, :, cx: cx + cs, cy: cy + cs]) + _y

        if self.flipping == True:
            # Naive flipping

            # Bring back to cpu
            arr = (x.data).cpu().numpy()
            arr = arr[:, :, :, :: -1]                              # Flip
            x.data = type(x.data)(np.ascontiguousarray(arr))    # Store

            _y = get_output(x[:, :, x1: x1 + cs, y1: y1 + cs]) + _y
            _y = get_output(x[:, :, x1: x1 + cs, y2: y2 + cs]) + _y
            _y = get_output(x[:, :, x2: x2 + cs, y1: y1 + cs]) + _y
            _y = get_output(x[:, :, x2: x2 + cs, y2: y2 + cs]) + _y
            _y = get_output(x[:, :, cx: cx + cs, cy: cy + cs]) + _y

            _y = _y / 10.0
        else:
            _y = _y / 5.0

        return _y
