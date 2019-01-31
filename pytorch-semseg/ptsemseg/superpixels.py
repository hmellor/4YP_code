import torch
from os.path import join
from tqdm import tqdm
from skimage import io
from skimage.util import img_as_float
from skimage.segmentation import slic
import time

def convert_to_superpixels(input, target, mask):
#    print("NEW IMAGE")
#    t=time.time()
    # Extract size data from input and target
    images, c, h, w = input.size()
    # Load the pre-processed segmentation
    segments_u = 0
    for image in range(images):
        segments_u += mask[image,:,:].unique().numel()
    # Initialise superpixel tensors
    input_s  = torch.zeros((segments_u,c), device=input.device)
    classes = torch.arange(c, device=input.device)
    # Iterate through all the images
#    print("Time to initialise variables (conversion):", time.time()-t)
#    t=time.time()
    for img in range(images):
        # Define variable for number of unique superpixels for current image
        img_superpixels = mask[img,:,:].unique().numel()
        img_offset = img*img_superpixels
        # Iterate through all the clusters
#        t1=time.time()
        for idx in range(img_superpixels):
            # Define mask for cluster idx
            segment_mask = mask[img,:,:]==idx
            # First take slices to select image, then apply mask, then taking mean
            input_s[img_offset+idx,classes] = input[img,classes,:,:][:,segment_mask].mean()
#        print("Inner loop time (conversion)", time.time()-t1)
#    print("Outer loop time (conversion)", time.time()-t)
    return input_s, target.squeeze()

def create_masks():
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
        # Perform SLIC segmentation
        numSegments = 300
        mask = slic(image, n_segments = numSegments, sigma = 5)
        mask = torch.from_numpy(mask)

        superpixels = mask.unique().numel()
        target_s = torch.zeros(superpixels)

        for superpixel in range(superpixels):
                # Define mask for cluster idx
                segment_mask = mask==superpixel
                # First take slices to select image, then apply mask, then 2D mode for majority class
                target_s[superpixel] = target[segment_mask].mode()[0].mode()[0]

        # Save for later
        save_name = image_number + ".pt"
        image_save_path = join(root, "SegmentationClass/SuperPixels", save_name)
        target_s_save_path = join(root, "SegmentationClass/pre_encoded_superpixels", save_name)
        torch.save(mask, image_save_path)
        torch.save(target_s, target_s_save_path)
