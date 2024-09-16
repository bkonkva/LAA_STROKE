# FILTER THROUGH SERIES TO IDENTIFY APPROPRIATE IMAGING
import re
import os
import shutil
import pydicom as dcm

DIR_FILES = r"F:\\test_set"  # "G://"
LIST_STUDIES = [file for file in os.listdir(DIR_FILES) if not "$" in file]
LIST_KEEP = ["%", "thin", "clear", "gate"]
LIST_SKIP = ["super"]
DIR_FILTERED = r"C:\\Users\\PUGlaa\\Desktop\\FILTERED_TESTSET"  # "F://FILTERED_TESTSET"


def sanitize_filename(filename):
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename


def check_str(series_desc: str, ref_list: list) -> bool:
    if any(substring in series_desc.lower() for substring in ref_list):
        return True
    return False


def check_thickness(dicom_file) -> bool:
    # check if slice thickness exceeds 1.25
    if "SliceThickness" in dicom_file and not dicom_file.SliceThickness == None:
        if float(dicom_file.SliceThickness) > 1.25:
            return True
    return False


def check_axial(dicom_file) -> bool:
    if "ImageOrientationPatient" in dicom_file:
        iop = dicom_file.ImageOrientationPatient
        is_axial_orientation = (round(iop[2], 6) == 0 and round(iop[5], 6) == 1) or (
            round(iop[2], 6) == 0 and round(iop[5], 6) == -1
        )
        if is_axial_orientation:
            return True
    return False


def check_for_number(series_desc: str) -> bool:
    pattern = r"\d{2,}"
    match = re.search(pattern, series_desc)
    return bool(match)


def filter_dicom(dicom_file) -> bool:
    if "SeriesDescription" in dicom_file:
        series_desc = dicom_file.SeriesDescription
        if check_str(series_desc, LIST_SKIP):
            return False
        # if not check_axial(dicom_file):
        #     return False
        if check_thickness(dicom_file):
            return False
        if check_str(series_desc, LIST_KEEP):
            return True
        if check_for_number(series_desc):
            return True
    return False


for num, dicom_case in enumerate(LIST_STUDIES):
    # find all dicom files in a study dir
    file_write = False
    dicom_dir = os.path.join(DIR_FILES, dicom_case)
    if not os.path.isdir(dicom_dir):
        continue
    dicom_paths = [
        os.path.join(dicom_dir, path)
        for path in os.listdir(dicom_dir)
        if path.endswith(".dcm")
    ]
    print(f"{num}/{len(LIST_STUDIES)}: {len(dicom_paths)} dicom files")
    # read each file
    for dicom_path in dicom_paths:
        dicom_file = dcm.dcmread(dicom_path, stop_before_pixels=True)
        # pass filters
        if filter_dicom(dicom_file):
            # copy/move file to new path with series unique folder
            study_dir = os.path.join(DIR_FILTERED, dicom_case)
            if not os.path.exists(study_dir):
                os.mkdir(study_dir)
            series_dir = os.path.join(
                study_dir,
                f"{str(dicom_file.SeriesNumber)} {sanitize_filename(dicom_file.SeriesDescription)}",
            )
            if not os.path.exists(series_dir):
                os.mkdir(series_dir)
            dst = os.path.join(series_dir, os.path.split(dicom_path)[-1])
            shutil.copyfile(dicom_path, dst)
            file_write = True
    # cleanup
    if file_write:
        series_dirs = [
            series_dir
            for series_dir in os.listdir(study_dir)
            if not series_dir.endswith(".zip")
        ]
        for series_dir in series_dirs:
            series_path = os.path.join(study_dir, series_dir)
            if len(os.listdir(series_path)) < 20:
                shutil.rmtree(series_path)
            else:
                shutil.make_archive(series_path, "zip", series_path)
                shutil.rmtree(series_path)
    else:
        print(f"no appropriate series found for {dicom_case}")
