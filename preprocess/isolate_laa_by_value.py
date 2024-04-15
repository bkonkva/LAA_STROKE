import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from glob import glob
from scipy.ndimage import label, center_of_mass

# DIRS AND PATHS
DIR_ROOT = os.path.join("D:\\", "LAA_STROKE", "data", "UW")
# DIR_LABELS = os.path.join(DIR_ROOT, r"labels\inference_results\ensemble")
DIR_LABELS = os.path.join(DIR_ROOT, r"labels\inference_results\3d")
DIR_WRITE = os.path.join(DIR_ROOT, r"labels\segmentation_masks\LAA_ISOLATED\ARRAY_from_nnunet")

list_masks = glob(os.path.join(DIR_LABELS, "**/*.gz"), recursive = True)

def most_common_nonzero(arr):
    flat_arr = arr.flatten()
    non_zero = flat_arr[flat_arr != 0]
    counts = np.bincount(non_zero)
    return np.argmax(counts)

for num, path_mask in enumerate(tqdm(list_masks)):
    if num < 115:
        continue
    case = os.path.split(path_mask)[-1]
    nifti_full = nib.load(path_mask)
    mask_full = nifti_full.get_fdata()
    mask_full[mask_full<4] = 0
    # remove all but largest CC
    labeled_arr, num_features = label(mask_full)
    try:
        biggest_roi = most_common_nonzero(labeled_arr)
    except:
        print(f'empty: {case}')
        continue
    mask_full[labeled_arr != biggest_roi] = 0


    nifti_laa = nib.Nifti1Image(mask_full, nifti_full.affine, nifti_full.header)
    path_write = os.path.join(DIR_WRITE, case)
    nib.save(nifti_laa, path_write)