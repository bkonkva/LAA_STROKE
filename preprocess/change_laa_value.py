import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

DIR_ROOT = os.path.join("D:\\", "LAA_STROKE", "data", "UW")
DIR_LABELS = os.path.join(DIR_ROOT, "labels", "segmentation_masks")
DIR_LABEL_FULL = os.path.join(DIR_LABELS, "FULL")
DIR_LABEL_LAA = os.path.join(DIR_LABELS, "LAA_ISOLATED", "ARRAY")
DIR_WRITE = os.path.join(DIR_LABELS, "LAA_SEPARATE")

for case in tqdm(os.listdir(DIR_LABEL_LAA), desc = "GIVING LAA UNIQUE ARRAY VALUE"):
    # load masks
    nifti_full = nib.load(os.path.join(DIR_LABEL_FULL, case))
    # mask_full = nifti_full.get_fdata()
    # nifti_laa = nib.load(os.path.join(DIR_LABEL_LAA, case))
    # mask_laa = nifti_laa.get_fdata()
    # # set LAA mask as new value
    # mask_new = np.copy(mask_full).astype(int)
    # mask_new[mask_laa > 0] =  4
    # nifti_full_laa = nib.Nifti1Image(mask_new, nifti_full.affine, nifti_full.header)
    path_write = os.path.join(DIR_WRITE, case)
    # nib.save(nifti_full_laa, path_write)
    nifti_new = nib.load(path_write)
    print('test')


# nib_new = nib.load(path_write)
# mask = nib_new.get_fdata()
#     mask[mask==5] = 4
#     nib_new_new = nib.Nifti1Image(mask, nib_new.affine, nib_new.header)
#     nib.save(nib_new_new, path_write)
    







