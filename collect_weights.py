import os
import wget
import segmentation_models_pytorch as smp

from multiprocessing import Pool


def load_weights(url, dst_path):
    if not os.path.exists(dst_path):
        wget.download(url, out=dst_path)


if __name__ == "__main__":

    args = []
    for encoder_name, encoder_params in smp.encoders.encoders.items():
        for pretrained_name, pretrained_params in encoder_params["pretrained_settings"].items():
            url = pretrained_params["url"]
            dst_path = f"weights/{pretrained_name}/{encoder_name}.pth"
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            args.append((url, dst_path))

    with Pool(20) as p:
        p.starmap(load_weights, args)
