import sys, os
import torch
import visdom
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict
from ptsemseg.superpixels import to_super_to_pixels


def test(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]

    # Setup image
    print("Read Input Image from : {}".format(args.img_path))
    img = misc.imread(args.img_path)

    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True, img_norm=args.img_norm)
    n_classes = loader.n_classes

    resized_img = misc.imresize(
        img, (loader.img_size[0], loader.img_size[1]), interp="bicubic"
    )

    orig_size = img.shape[:-1]

    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    if args.img_norm:
        img = img.astype(float) / 255.0

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Setup Model
    model_dict = {"arch": model_name}
    model = get_model(model_dict, n_classes, version=args.dataset)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    images = img.to(device)
    outputs = model(images)
    if args.mask_path:
        print("Read Image Mask from : {}".format(args.mask_path))
        mask = torch.load(args.mask_path)
        mask = mask.to(device)
        outputs = to_super_to_pixels(outputs, mask)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)


    decoded = loader.decode_segmap(pred)
    print("Classes found: ", np.unique(pred))
    misc.imsave(args.out_path, decoded)
    print("Segmentation Mask Saved at: {}".format(args.out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="pascal",
        help="Dataset to use ['pascal, camvid, ade20k etc']",
    )

    parser.add_argument(
        "--img_norm",
        dest="img_norm",
        action="store_true",
        help="Enable input image scales normalization [0, 1] \
                              | True by default",
    )
    parser.add_argument(
        "--no-img_norm",
        dest="img_norm",
        action="store_false",
        help="Disable input image scales normalization [0, 1] |\
                              True by default",
    )
    parser.set_defaults(img_norm=True)


    parser.add_argument(
        "--img_path", nargs="?", type=str, default=None, help="Path of the input image"
    )
    parser.add_argument(
        "--mask_path", nargs="?", type=str, default=None, help="Path of the superpixel mask"
    )
    parser.add_argument(
        "--out_path",
        nargs="?",
        type=str,
        default=None,
        help="Path of the output segmap",
    )
    args = parser.parse_args()
    test(args)
