import torch
import segmentation_models_pytorch as smp


def get_encoder_weights_path(encoder_name, encoder_weights):
    return "weights/{}/{}.pth".format(encoder_weights, encoder_name)


def load_weights(summon, model, weights_file_path):
    summon.pull(weights_file_path)
    with open(weights_file_path, mode='rb') as fd:
        weights = torch.load(fd)
        model.load_state_dict(weights)
    return model


def get_model(summon, model, encoder_name="resnet34", encoder_weights="imagenet", **kwargs):
    if model not in smp.__dict__:
        raise ValueError("No such model architecture ({}) in SMP.".format(model))

    model = smp.__dict__[model](encoder_name=encoder_name, encoder_weights=None, **kwargs)

    if encoder_weights is not None:
        path = get_encoder_weights_path(encoder_name, encoder_weights)
        load_weights(summon, model.encoder, path)

    return model
