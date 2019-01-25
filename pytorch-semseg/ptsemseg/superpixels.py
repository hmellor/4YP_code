import torch
from os.path import join
from tqdm import tqdm
from skimage import io
from skimage.util import img_as_float
from skimage.segmentation import slic

def convert_to_superpixels(input, target, mask):
    # Extract size data from input and target
    images, c, h, w = input.size()
    nt, ht, wt = target.size()
    print("input", input.size())
    print("target", target.size())
    print("mask", mask.size())
    # Load the pre-processed segmentation
    superpixels = torch.zeros(images,h,w)
    segments_u = 0
    for image in range(images):
        superpixels[image,:,:] = torch.load('{}.pt'.format(names[image]))
        segments_u += superpixels[image,:,:].unique().numel()
        print("Total superpixels:           {}".format(segments_u))
    # Initialise superpixel tensors
    input_s  = torch.zeros(segments_u,c)
    target_s = torch.zeros(segments_u)
    # Some prints for sanity checks
    print("Input shape:              {}\nTarget shape:             {}".format(input.shape, target.shape))
    print("Input super-pixel shape:  {}\nTarget super-pixel shape: {}".format(input_s.shape, target_s.shape))
    print("superpixels shape:           {}".format(superpixels.shape))
    # Iterate through all the images
    for img in range(images):
        # Define variable for number of unique superpixels for current image
        img_seg_u = superpixels[img,:,:].unique().numel()
        # Iterate through all the clusters
        for idx in range(img_seg_u):
            # Define mask for cluster idx
            mask = superpixels[img,:,:]==idx
            # First take slices to select image, then apply mask, then 2D mode for majority class
            target_s[(img*img_seg_u)+idx] = target[img,:,:][mask].mode()[0].mode()[0]
            # Iterate through all the classes
            for k in range(c):
                # Same process as before but also iterating through classes and taking mean because these are scores
                input_s[(img*img_seg_u)+idx,k] = input[img,k,:,:][mask].mean()

def create_masks():
    root = "../../datasets/VOCdevkit/VOC2011"
    image_list_path = join(root, "ImageSets/Segmentation/trainval.txt")
    image_list = tuple(open(image_list_path, "r"))
    image_list = [id_.rstrip() for id_ in image_list]
    for image_number in tqdm(image_list):
        image_name = image_number + ".jpg"
        image_path = join(root, "JPEGImages", image_name)
        image = img_as_float(io.imread(image_path))
        # Perform SLIC segmentation
        numSegments = 100
        segments = slic(image, n_segments = numSegments, sigma = 5)

        # Convery to torch and save for later
        segments = torch.from_numpy(segments)
        save_name = image_number + ".pt"
        save_path = join(root, "SegmentationClass/SuperPixels", save_name)
        torch.save(segments, save_path)
