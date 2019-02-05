import torch
from os.path import join
from tqdm import tqdm
from skimage import io
from skimage.util import img_as_float
from skimage.segmentation import slic
import time

def convert_to_superpixels(input, target, mask):
    # Extract size data from input and target
    images, c, h, w = input.size()
    if images > 1:
        raise RuntimeError("Not yet implemented for batch sizes greater than 1")
    Q = mask.unique().numel()
    t = torch.zeros((Q,c), device=input.device)
    size = torch.zeros(Q, device=input.device)
    counter = torch.ones_like(mask, device=input.device)

    size.put_(mask,counter.float(),True)

    input = input.view(c, -1)
    mask = mask.view(1, -1).repeat(c,1)
    arange = torch.arange(start=1, end=c, device=input.device)
    mask[arange,:] += Q*arange.view(-1,1)
    t = t.put_(mask,input,True).view(c,Q).t()
    #t = (t.t()/size).t() # makes it mean score

    # # Load the pre-processed segmentation
    # segments_u = 0
    # for image in range(images):
    #     segments_u += mask[image,:,:].unique().numel()
    # # Initialise superpixel tensors
    # input_s  = torch.zeros((segments_u,c), device=input.device)
    # size_s  = torch.zeros(segments_u, device=input.device)
    # classes = torch.arange(c, device=input.device)
    # # Iterate through all the images
    # for img in range(images):
    #     # Define variable for number of unique superpixels for current image
    #     img_superpixels = mask[img,:,:].unique().numel()
    #     img_offset = img*img_superpixels
    #     # Iterate through all the clusters
    #     for idx in range(img_superpixels):
    #         # Define mask for cluster idx
    #         segment_mask = mask[img,:,:]==idx
    #         # First take slices to select image, then apply mask, then sum scores
    #         input_s[img_offset+idx,classes] = input[img,classes,:,:][:,segment_mask].sum()
    #         size_s[img_offset+idx] = segment_mask.sum()

    return t, target.view(-1), size

def find_smallest_object():
    root = "../../datasets/VOCdevkit/VOC2011"
    image_list_path = join(root, "ImageSets/Segmentation/trainval.txt")
    image_list = tuple(open(image_list_path, "r"))
    image_list = [id_.rstrip() for id_ in image_list]
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

def create_mask(image, target, numSegments):
    # Perform SLIC segmentation
    mask = slic(image, n_segments = numSegments, sigma = 5)
    mask = torch.from_numpy(mask)

    superpixels = mask.unique().numel()
    target_s = torch.zeros(superpixels, dtype=torch.long)

    for superpixel in range(superpixels):
        # Define mask for cluster idx
        segment_mask = mask==superpixel
        # First take slices to select image, then apply mask, then 2D mode for majority class
        target_s[superpixel] = target[segment_mask].mode()[0].mode()[0]
    return mask, target_s

def create_masks(numSegments=100):
    # Establish images to create masks for
    root = "../../datasets/VOCdevkit/VOC2011"
    image_list_path = join(root, "ImageSets/Segmentation/trainval.txt")
    image_list = tuple(open(image_list_path, "r"))
    image_list = [id_.rstrip() for id_ in image_list]
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

def create_optimal_masks(lower_bound=20, upper_bound=300, threshhold=0.98):
    root = "../../datasets/VOCdevkit/VOC2011"
    image_list_path = join(root, "ImageSets/Segmentation/trainval.txt")
    image_list = tuple(open(image_list_path, "r"))
    image_list = [id_.rstrip() for id_ in image_list]
    for image_number in tqdm(image_list):
        image_name = image_number + ".jpg"
        target_name = image_number + ".png"
        image_path = join(root, "JPEGImages", image_name)
        target_path = join(root, "SegmentationClass/pre_encoded", target_name)
        image = img_as_float(io.imread(image_path))
        target = io.imread(target_path)
        target = torch.from_numpy(target)

        mask, target_s = create_mask(image, target, lower_bound)
        acc = image_accuracy(target, mask)
        superpixels = lower_bound
        while acc < threshhold and superpixels < upper_bound:
            superpixels += 20
            mask, target_s = create_mask(image, target, superpixels)
            acc = image_accuracy(target, mask)
#            print(acc, mask.unique().numel())

        # Save for later
        save_name = image_number + ".pt"
        image_save_path = join(root, "SegmentationClass/SuperPixels_optimal", save_name)
        target_s_save_path = join(root, "SegmentationClass/pre_encoded_superpixels_optimal", save_name)
        torch.save(mask, image_save_path)
        torch.save(target_s, target_s_save_path)

def image_accuracy(target, mask):
    target_s = torch.zeros_like(target)
    superpixels = mask.unique().numel()
    for superpixel in range(superpixels):
        # Define mask for cluster idx
        segment_mask = mask==superpixel
        # First take slices to select image, then apply mask, then 2D mode for majority class
        target_s[segment_mask] = target[segment_mask].mode()[0].mode()[0]
    accuracy = torch.mean((target==target_s).float())
    return accuracy

def dataset_accuracy(optimal=None):
    root = "../../datasets/VOCdevkit/VOC2011"
    image_list_path = join(root, "ImageSets/Segmentation/trainval.txt")
    image_list = tuple(open(image_list_path, "r"))
    image_list = [id_.rstrip() for id_ in image_list]
    image_acc = 0
    if optimal:
        mask_dir = "SegmentationClass/SuperPixels_optimal"
        target_dir = "SegmentationClass/pre_encoded"
    else:
        mask_dir = "SegmentationClass/SuperPixels"
        target_dir = "SegmentationClass/pre_encoded"
    for image_number in tqdm(image_list):
        mask_path = join(root, mask_dir, image_number + ".pt")
        target_path = join(root, target_dir, image_number + ".png")
        mask = torch.load(mask_path)
        target = io.imread(target_path)
        target = torch.from_numpy(target)
        image_acc += image_accuracy(target, mask)
    dataset_acc = image_acc / len(image_list)
    return dataset_acc

#create_masks()
#create_optimal_masks()
#print("  100 Superpixels: ", dataset_accuracy())
#print("Smart Superpixels: ", dataset_accuracy(optimal=True))
