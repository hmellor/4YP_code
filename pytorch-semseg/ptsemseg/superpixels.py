import torch
from os.path import join
from tqdm import tqdm
from skimage import io
from skimage.util import img_as_float
from skimage.segmentation import slic
import time

# For use during runtime
def convert_to_superpixels(input, target, mask):
    # Extract size data from input and target
    images, c, h, w = input.size()
    if images > 1:
        raise RuntimeError("Not yet implemented for batch sizes greater than 1")
    # Initialise vairables to use
    Q = mask.unique().numel()
    output = torch.zeros((Q,c), device=input.device)
    size = torch.zeros(Q, device=input.device)
    counter = torch.ones_like(mask, device=input.device)
    # Calculate the size of each superpixel
    size.put_(mask,counter.float(),True)
    # Calculate the mean value of each superpixel
    input = input.view(c, -1)
    mask = mask.view(1, -1).repeat(c,1)
    arange = torch.arange(start=1, end=c, device=input.device)
    mask[arange,:] += Q*arange.view(-1,1)
    output = output.put_(mask,input,True).view(c,Q).t()
    output = (output.t()/size).t()
    return output, target.view(-1), size

# For pre-processing
def create_masks(numSegments=100):
    # Generate image list
    image_list, root = get_image_list()
    for image_number in tqdm(image_list):
        # Load image/target pair
        image_name = image_number + ".jpg"
        target_name = image_number + ".png"
        image_path = join(root, "JPEGImages", image_name)
        target_path = join(root, "SegmentationClass/pre_encoded", target_name)
        image = img_as_float(io.imread(image_path))
        target = io.imread(target_path)
        target = torch.from_numpy(target)
        # Create mask for image/target pair
        mask, target_s = create_mask(image, target, numSegments)

        # Save for later
        save_name = image_number + ".pt"
        image_save_path = join(root, "SegmentationClass/SuperPixels", save_name)
        target_s_save_path = join(root, "SegmentationClass/pre_encoded_superpixels", save_name)
        torch.save(mask, image_save_path)
        torch.save(target_s, target_s_save_path)
    print(dataset_accuracy())
    print(find_size_variance())

def create_mask(image, target, numSegments):
    # Perform SLIC segmentation
    mask = slic(image, n_segments = numSegments, slic_zero=True)
    mask = torch.from_numpy(mask)

    superpixels = mask.unique().numel()
    overseg = superpixels

    # Oversegmentation step
    for superpixel in range(superpixels):
        overseg -= 1
        # Define mask for superpixel
        segment_mask = mask==superpixel
        # Classes in this superpixel
        classes = target[segment_mask].unique(sorted=True)
        # Check if superpixel is on target boundary
        on_boundary = classes.numel() > 1
#        print("Is sp  on boundary?:", on_boundary)
        if on_boundary:
            # Find how many of each class is in superpixel
            class_count = torch.bincount(target[segment_mask])
            # Remove zero elements
            class_count = class_count[class_count.nonzero()].float()
#            print(class_count)
            minority_class = min(class_count)
     #       print(minority_class, class_count.sum() *0.05)
            above_threshold = minority_class > class_count.sum() * 0.05
 #           print("Is minority class big enough:", above_threshold)
            if above_threshold:
                # Leaving one class in supperpixel be
                for c in classes[1:]: 
                    # Adding to the oversegmentation offset
                    overseg += 1
                    # Add offset to class c in the mask
                    mask[segment_mask] += (target[segment_mask]==c).long()*overseg

    # Redefine how many superpixels there are and create target_s
    superpixels = mask.unique().numel()
    target_s = torch.zeros(superpixels, dtype=torch.long)
    for superpixel in range(superpixels):
        # Define mask for superpixel
        segment_mask = mask==superpixel
        # Apply mask, then 2D mode for majority class
        target_s[superpixel] = target[segment_mask].mode()[0].mode()[0]
    return mask, target_s

def get_image_list(split=None):
    root = "../../datasets/VOCdevkit/VOC2011"
    if split == None:
        image_list_path = join(root, "ImageSets/Segmentation/trainval.txt")
    else:
        image_list_path = join(root, "ImageSets/Segmentation/", split + ".txt")
    image_list = tuple(open(image_list_path, "r"))
    image_list = [id_.rstrip() for id_ in image_list]
    return image_list, root

# For superpixel validation
def mask_accuracy(target, mask):
    target_s = torch.zeros_like(target)
    superpixels = mask.unique().numel()
    for superpixel in range(superpixels):
        # Define mask for cluster idx
        segment_mask = mask==superpixel
        # First take slices to select image, then apply mask, then 2D mode for majority class
        target_s[segment_mask] = target[segment_mask].mode()[0].mode()[0]
    accuracy = torch.mean((target==target_s).float())
    return accuracy

def dataset_accuracy():
    # Generate image list
    image_list, root = get_image_list()
    mask_acc = 0
    mask_dir = "SegmentationClass/SuperPixels"
    target_dir = "SegmentationClass/pre_encoded"
    for image_number in tqdm(image_list):
        mask_path = join(root, mask_dir, image_number + ".pt")
        target_path = join(root, target_dir, image_number + ".png")
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
        target_path = join(root, "SegmentationClass/pre_encoded", target_name)
        target = io.imread(target_path)
        target = torch.from_numpy(target)
        object_size = torch.ne(target, 0).sum()
        if object_size < smallest_object:
            smallest_object = object_size
            print(smallest_object, image_number)
    return smallest_object

def find_broken_images(split=None):
    # Generate image list
    image_list, root = get_image_list(split)
    broken_images = 0
    for image_number in tqdm(image_list):
        target_name = image_number + ".pt"
        target_path = join(root, "SegmentationClass/pre_encoded_superpixels", target_name)
        target = torch.load(target_path)
        if target.nonzero().numel() < 1:
            broken_images += 1
            print(target.nonzero(), target_name)
    return broken_images

def find_size_variance():
    image_list, root = get_image_list()
    mask_dir = "SegmentationClass/SuperPixels"
    dataset_variance = 0
    for image_number in tqdm(image_list):
        mask_path = join(root, mask_dir, image_number + ".pt")
        mask = torch.load(mask_path)
        # Initialise number of superpixels tensors
        Q = mask.unique().numel()
        size = torch.zeros(Q)
        counter = torch.ones_like(mask)
        # Calculate the size of each superpixel
        size.put_(mask,counter.float(),True)
        # Calculate the mean and standard deviation of the sizes
        std = size.std()
        mean = size.mean()
        # Add to the variance of the total datasets
        dataset_variance += std / mean
    dataset_variance /= len(image_list)
    return dataset_variance
