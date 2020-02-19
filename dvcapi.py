import segmentation_models_pytorch as smp

def get_model(summon, **kwargs):
    arch = kwargs.pop("arch")
    return smp.__dict__[arch](**kwargs)
