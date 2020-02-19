import segmentation_models_pytorch as smp


WEIGHTS = dict(
    resnet34="weights/resnet34-333f7ec4.pth",
    resnet18="weights/resnet18-5c106cde.pth",
)


def load_weights(summon, model, weights_file):
    summon.pull(weights_file)
    with open(weights_file, mode='rb') as fd:
        print("Loading weights...") 
        weights = torch.load(fd)
        model.load_state_dict(weights)
    return model


def get_model(summon, **kwargs):
    arch = kwargs.pop("arch")
    encoder_name = kwargs.pop("encoder_name")
    encoder_weights = kwargs.pop("encoder_weights", None)
    kwargs["encoder_weights"] = None
    model = smp.__dict__[arch](**kwargs)
    if encoder_weights is not None:
        load_weights(summon, model.encoder, WEIGHTS[encoder_name])
    return model
