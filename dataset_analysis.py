from ptsemseg.superpixels import fix_broken_images
from ptsemseg.superpixels import dataset_accuracy
from ptsemseg.superpixels import find_size_variance

sp_levels = [100, 1000, 10000]
fix_broken_images(min(sp_levels))

for sp_level in sp_levels:
    print("{} Superpixel Dataset".format(sp_level))
    acc = dataset_accuracy(sp_level)
    print("Accuracy: {}".format(acc))
    var = find_size_variance(sp_level)
    print("Variance: {}\n".format(var))
