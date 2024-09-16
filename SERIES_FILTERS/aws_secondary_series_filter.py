import os
from tqdm import tqdm
from glob import glob
import shutil
import pydicom as dcm
import nibabel as nib
import numpy as np
from scipy.ndimage import label, find_objects
import dicom2nifti

# DIRS
DIR_VISN20 = r"D:\LAA_STROKE\data\VA\VISN20"
DIR_DICOM = os.path.join(DIR_VISN20, r"raw\unzipped")
DIR_IMAGES_ORIGINAL = os.path.join(DIR_VISN20, r"preprocessed\nifti_flat")
DIR_IMAGES_FILTERED = os.path.join(DIR_VISN20, r"preprocessed\nifti_filtered")
DIR_LABELS_ORIGINAL = os.path.join(DIR_VISN20, r"results\2d_fold0")
DIR_LABELS_FILTERED = os.path.join(DIR_VISN20, r"results\2d_fold0_filtered")
DIR_LABELS_POSTPROCESSED = os.path.join(DIR_VISN20, r"results\2d_fold0_postprocessed")


# retitle underscore files...
LIST_SKIP = [
    "chest_toes",
    "thoracic iw",
    "aaa",
    "carotid",
    "art",
    "neph",
    "abd",
    "chest_abd",
    "any i(w)",
    "non con",
    "wo",
]
LIST_KEEP = ["cpta", "chest", "pa"]


def keep_largest_roi(binary_mask):
    labeled_mask, num_features = label(binary_mask)
    if num_features == 0:
        # mask is empty
        return False
    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0
    largest_component_label = component_sizes.argmax()
    largest_component_mask = labeled_mask == largest_component_label
    return largest_component_mask


def postprocess_mask(path_nifti):
    nifti_img = nib.load(path_nifti)
    mask = nifti_img.get_fdata()
    mask = (mask > 0).astype(np.int32)
    mask = keep_largest_roi(mask)
    if not isinstance(mask, np.ndarray):
        return False
    new_nifti = nib.Nifti1Image(
        mask.astype(np.int32), nifti_img.affine, nifti_img.header
    )
    nib.save(
        new_nifti, os.path.join(DIR_LABELS_POSTPROCESSED, os.path.basename(path_nifti))
    )
    return


def find_src_path(study_instance_uid, series_number, study_description):
    type_a = glob(
        os.path.join(DIR_IMAGES_ORIGINAL, f"**/{study_instance_uid}_{series_number}*"),
        recursive=True,
    )
    if len(type_a) > 0:
        return type_a[0]
    type_b = glob(
        os.path.join(
            DIR_IMAGES_ORIGINAL,
            f'**/{"_".join(study_instance_uid.split("."))}_{series_number}*',
        ),
        recursive=True,
    )
    if len(type_b) > 0:
        return type_b[0]
    return False


for patient in tqdm(os.listdir(DIR_DICOM)):
    dir_patient = os.path.join(DIR_DICOM, patient)
    list_studies = [
        study for study in os.listdir(dir_patient) if not study.endswith("zip")
    ]
    for study in list_studies:
        try:
            dir_study = os.path.join(dir_patient, study)
            path_dicom_file = os.path.join(dir_study, os.listdir(dir_study)[0])
            dicom_file = dcm.dcmread(path_dicom_file, stop_before_pixels=True)
            # check DICOM file for ICN, SSN, patient name
            accession = dicom_file.AccessionNumber
            # print(f"CHECK {patient} : {study} : {accession}")
            study_description = dicom_file.StudyDescription
            study_instance_uid = dicom_file.StudyInstanceUID
            series_number = dicom_file.SeriesNumber
            # filter
            if any(substring in study_description.lower() for substring in LIST_SKIP):
                print(f"skipping: {study_description}")
                continue
            nifti_src = find_src_path(
                study_instance_uid, series_number, study_description
            )
            if not nifti_src:
                print(f"missing: {patient} : {study}")
                continue
            nifti_dst = os.path.join(DIR_IMAGES_FILTERED, f"{accession}_0000.nii.gz")
            label_src = os.path.join(
                DIR_LABELS_ORIGINAL,
                f'{os.path.basename(nifti_src).split("_0000")[0]}.nii.gz',
            )
            label_dst = os.path.join(DIR_LABELS_FILTERED, f"{accession}.nii.gz")
            if any(substring in study_description.lower() for substring in LIST_KEEP):
                print(f"keeping: {study_description}")
                shutil.copyfile(nifti_src, nifti_dst)
                shutil.copyfile(label_src, label_dst)
                postprocess_mask(label_dst)
                break
            shutil.copyfile(nifti_src, nifti_dst)
            shutil.copyfile(label_src, label_dst)
            postprocess_mask(label_dst)
        except:
            print("shutil error")
