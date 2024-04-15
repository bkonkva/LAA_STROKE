from glob import glob
from tqdm import tqdm
import nibabel as nib
import os
import nrrd

DIR_ROOT = os.path.join("D:\\", "LAA_STROKE", "data", "LAA_TEST")
DIR_NERD = os.path.join(DIR_ROOT, 'NRRD')
DIR_WRITE = os.path.join(DIR_ROOT, 'LABELS_FINAL')
DIR_IMAGES = os.path.join(DIR_ROOT, "ROUND1_LABELS")

LIST_NERD_FILES = glob(DIR_NERD+'/*.nrrd')

def main():
    for nrrd_file in tqdm(LIST_NERD_FILES, desc = "CONVERTING NRRD FILES TO NIFTI"):
        # # Read the .nrrd file
        data, header = nrrd.read(nrrd_file)
        
        # # Create a NIfTI1Image object
        case = os.path.split(nrrd_file)[-1].split(".")[0] + ".nii.gz"
        nifti_img = nib.load(os.path.join(DIR_IMAGES, case))
        new_nifti = nib.Nifti1Image(data, affine = nifti_img.affine, header = nifti_img.header)
        # # Update the NIf
        path_write = os.path.join(DIR_WRITE, case)
        nib.save(new_nifti, path_write)

if __name__ == "__main__":
    main()