import torch
from os.path import join
from os.path import exists
from os.path import dirname
from os.path import abspath
from os import mkdir
from os import remove
from os import listdir
from tqdm import tqdm
from skimage import io
from skimage.util import img_as_float
from skimage.segmentation import slic

# Define absolute path for accessing dataset files
pkg_dir = dirname(abspath(__file__))
'''For use during runtime'''


def convert_to_superpixels(input, target, mask):
    # Extract size data from input and target
    images, c, h, w = input.size()
    if images > 1:
        raise RuntimeError("Not implemented for batch sizes greater than 1")
    # Initialise vairables to use
    Q = mask.unique().numel()
    output = torch.zeros((Q, c), device=input.device)
    size = torch.zeros(Q, device=input.device)
    counter = torch.ones_like(mask, device=input.device)
    # Calculate the size of each superpixel
    size.put_(mask, counter.float(), True)
    # Calculate the mean value of each superpixel
    input = input.view(c, -1)
    mask = mask.view(1, -1).repeat(c, 1)
    arange = torch.arange(start=1, end=c, device=input.device)
    mask[arange, :] += Q * arange.view(-1, 1)
    output = output.put_(mask, input, True).view(c, Q).t()
    output = (output.t() / size).t()
    return output, target.view(-1), size


def convert_to_pixels(input, output, mask):
    n, c, h, w = output.size()
    for k in range(c):
        output[0, k, :, :] = torch.gather(
            input[:, k], 0, mask.view(-1)).view(h, w)
    return output


def to_super_to_pixels(input, mask):
    target = torch.tensor([])
    input_s, _, _ = convert_to_superpixels(input, target, mask)
    output = convert_to_pixels(input_s, input, mask)
    return output


def setup_superpixels(superpixels):
    root = "../../datasets/VOCdevkit/VOC2011"
    image_save_dir = join(
        pkg_dir,
        root,
        "SegmentationClass/{}_sp".format(superpixels)
    )
    target_s_save_dir = join(
        pkg_dir,
        root,
        "SegmentationClass/pre_encoded_{}_sp".format(superpixels)
    )
    dirs = [image_save_dir, target_s_save_dir]
    dataset_len = len(get_image_list()[0])
    if not any(exists(x) and len(listdir(x)) == dataset_len for x in dirs):
            print("Superpixel dataset of scale {} superpixels either doesn't exist or is incomplete".format(superpixels))
            print("Generating superpixel dataset now...")
            create_masks(superpixels)

    fix_broken_images(superpixels)


'''For pre-processing'''


def create_masks(numSegments=100, limOverseg=None):
    # Generate image list
    image_list, root = get_image_list()
    for image_number in tqdm(image_list):
        # Load image/target pair
        image_name = image_number + ".jpg"
        target_name = image_number + ".png"
        image_path = join(pkg_dir, root, "JPEGImages", image_name)
        target_path = join(pkg_dir, root, "SegmentationClass/pre_encoded", target_name)
        image = img_as_float(io.imread(image_path))
        target = io.imread(target_path)
        target = torch.from_numpy(target)
        # Create mask for image/target pair
        mask, target_s = create_mask(
            image=image,
            target=target,
            numSegments=numSegments,
            limOverseg=limOverseg
        )

        # Save for later
        image_save_dir = join(
            pkg_dir,
            root,
            "SegmentationClass/{}_sp".format(numSegments)
        )
        target_s_save_dir = join(
            pkg_dir,
            root,
            "SegmentationClass/pre_encoded_{}_sp".format(numSegments)
        )
        if not exists(image_save_dir):
            mkdir(image_save_dir)
        if not exists(target_s_save_dir):
            mkdir(target_s_save_dir)
        save_name = image_number + ".pt"
        image_save_path = join(pkg_dir, image_save_dir, save_name)
        target_s_save_path = join(pkg_dir, target_s_save_dir, save_name)
        torch.save(mask, image_save_path)
        torch.save(target_s, target_s_save_path)


def create_mask(image, target, numSegments, limOverseg):
    # Perform SLIC segmentation
    mask = slic(image, n_segments=numSegments, slic_zero=True)
    mask = torch.from_numpy(mask)

    if limOverseg is not None:
        # Oversegmentation step
        superpixels = mask.unique().numel()
        overseg = superpixels
        for superpixel in range(superpixels):
            overseg -= 1
            # Define mask for superpixel
            segment_mask = mask == superpixel
            # Classes in this superpixel
            classes = target[segment_mask].unique(sorted=True)
            # Check if superpixel is on target boundary
            on_boundary = classes.numel() > 1
            # If current superpixel is on a gt boundary
            if on_boundary:
                # Find how many of each class is in superpixel
                class_hist = torch.bincount(target[segment_mask])
                # Remove zero elements
                class_hist = class_hist[class_hist.nonzero()].float()
                # Find minority class in superpixel
                min_class = min(class_hist)
                # Is the minority class large enough for oversegmentation
                above_threshold = min_class > class_hist.sum() * limOverseg
                if above_threshold:
                    # Leaving one class in supperpixel be
                    for c in classes[1:]:
                        # Adding to the oversegmentation offset
                        overseg += 1
                        # Add offset to class c in the mask
                        mask[segment_mask] += (target[segment_mask]
                                               == c).long() * overseg

    # (Re)define how many superpixels there are and create target_s
    superpixels = mask.unique().numel()
    target_s = torch.zeros(superpixels, dtype=torch.long)
    for superpixel in range(superpixels):
        # Define mask for superpixel
        segment_mask = mask == superpixel
        # Apply mask, the mode for majority class
        target_s[superpixel] = target[segment_mask].view(-1).mode()[0]
    return mask, target_s


def get_image_list(split=None):
    root = "../../datasets/VOCdevkit/VOC2011"
    if split is None:
        image_list_path = join(pkg_dir, root, "ImageSets/Segmentation/trainval.txt")
    else:
        image_list_path = join(pkg_dir, root, "ImageSets/Segmentation/", split + ".txt")
    image_list = tuple(open(image_list_path, "r"))
    image_list = [id_.rstrip() for id_ in image_list]
    return image_list, root


'''For superpixel validation'''


def mask_accuracy(target, mask):
    target_s = torch.zeros_like(target)
    superpixels = mask.unique().numel()
    for superpixel in range(superpixels):
        # Define mask for cluster idx
        segment_mask = mask == superpixel
        # Take slices to select image, apply mask, mode for majority class
        target_s[segment_mask] = target[segment_mask].view(-1).mode()[0]
    accuracy = torch.mean((target == target_s).float())
    return accuracy


def dataset_accuracy(superpixels):
    # Generate image list
    image_list, root = get_image_list()
    mask_acc = 0
    mask_dir = "SegmentationClass/SegmentationClass/{}_sp".format(superpixels)
    target_dir = "SegmentationClass/pre_encoded_{}_sp".format(superpixels)
    for image_number in tqdm(image_list):
        mask_path = join(pkg_dir, root, mask_dir, image_number + ".pt")
        target_path = join(pkg_dir, root, target_dir, image_number + ".png")
        mask = torch.load(mask_path)
        target = io.imread(target_path)
        target = torch.from_numpy(target)
        mask_acc += mask_accuracy(target, mask)
    dataset_acc = mask_acc / len(image_list)
    return dataset_acc


def find_smallest_object():
    # Generate image list
    image_list, root = get_image_list()
    smallest_object = 1e6
    for image_number in tqdm(image_list):
        target_name = image_number + ".png"
        target_path = join(pkg_dir, root, "SegmentationClass/pre_encoded", target_name)
        target = io.imread(target_path)
        target = torch.from_numpy(target)
        object_size = torch.ne(target, 0).sum()
        if object_size < smallest_object:
            smallest_object = object_size
            print(smallest_object, image_number)
    return smallest_object


def find_usable_images(split, superpixels):
    # Generate image list
    image_list, root = get_image_list(split)
    usable = []
    target_dir = join(
        pkg_dir,
        root,
        "SegmentationClass/pre_encoded_{}_sp".format(superpixels)
    )
    for image_number in image_list:
        target_name = image_number + ".pt"
        target_path = join(pkg_dir, target_dir, target_name)
        target = torch.load(target_path)
        if target.nonzero().numel() > 0:
            usable.append(image_number)
    return usable, root


def fix_broken_images(superpixels):
    for split in ["train", "val"]:
        usable, root = find_usable_images(split=split, superpixels=superpixels)
        super_path = join(pkg_dir, root, "ImageSets/Segmentation", split + "_super.txt")
        if exists(super_path):
            remove(super_path)
        with open(super_path, "w+") as file:
            for image_number in usable:
                file.write(image_number + "\n")


def find_size_variance(superpixels):
    image_list, root = get_image_list()
    mask_dir = "SegmentationClass/{}_sp".format(superpixels)
    dataset_variance = 0
    for image_number in tqdm(image_list):
        mask_path = join(pkg_dir, root, mask_dir, image_number + ".pt")
        mask = torch.load(mask_path)
        # Initialise number of superpixels tensors
        Q = mask.unique().numel()
        size = torch.zeros(Q)
        counter = torch.ones_like(mask)
        # Calculate the size of each superpixel
        size.put_(mask, counter.float(), True)
        # Calculate the mean and standard deviation of the sizes
        std = size.std()
        mean = size.mean()
        # Add to the variance of the total datasets
        dataset_variance += std / mean
    dataset_variance /= len(image_list)
    return dataset_variance


if __name__ == "__main__":
    import argparse

    # Initiates arguments
    parser = argparse.ArgumentParser(
        description='Module for processing PyTorch tensors as superpixels',
        epilog=None
    )
    parser.add_argument(
        '-c',
        '--create_masks',
        nargs=1,
        type=int,
        metavar='N',
        help='Creates segment masks with N segments'
    )
    parser.add_argument(
        '-f',
        '--fix_broken',
        nargs=1,
        type=int,
        metavar='M',
        help="Fix the M superpixel split so it doen't cause training to fail"
    )
    args = parser.parse_args()

    if args.create_masks:
        print('Creating masks with {} segments each'.format(
            args.create_masks[0]))
        create_masks(args.create_masks[0])

    if args.fix_broken:
        print('Creating train and val splits so that training does not fail')
        fix_broken_images(args.fix_broken[0])
