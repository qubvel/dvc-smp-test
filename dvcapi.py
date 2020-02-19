import torch
import segmentation_models_pytorch as smp


def load_weights(summon, model: torch.nn.Module, weights_file_path: str):
    summon.pull(weights_file_path)
    with open(weights_file_path, mode='rb') as fd:
        weights = torch.load(fd)
        model.load_state_dict(weights)
    return model


def get_model(summon, model: str, encoder_name: str = "resnet34", encoder_weights: str = "imagenet", **kwargs):
    if model not in smp.__dict__:
        raise ValueError("No such model architecture ({}) in SMP.".format(model))

    model = smp.__dict__[model](encoder_name=encoder_name, encoder_weights=None, **kwargs)

    if encoder_weights is not None:
        path = "weights/{}/{}.pth".format(encoder_weights, encoder_name)
        load_weights(summon, model.encoder, path)

    return model
