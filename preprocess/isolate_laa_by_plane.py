import os
import json
import nrrd
import numpy as np
import nibabel as nib
from scipy.ndimage import label, center_of_mass
import vtk 
from vtkmodules.util import numpy_support
from tqdm import tqdm

# DIRS AND PATHS
DIR_ROOT = os.path.join("D:\\", "LAA_STROKE", "data", "UW")
# path_json = os.path.join(DIR_ROOT, "code", "test", "2549981393_0000_OSTIUM.mrk.json")
# # path_nifti = os.path.join(DIR_ROOT, "data", "UW", "preprocessed", "1343552822_0000.nii.gz")
# path_nifti = os.path.join(DIR_ROOT, "data", "UW", "labels", "ground_truth", "la_labels2", "2549981393_0000.nii.gz")
# path_new_nifti = os.path.join(DIR_ROOT, "code", "test", "test.nii.gz")
# path_mesh = os.path.join(DIR_ROOT, "code", "test", "mesh_test.stl")
# path_vtk = os.path.join(DIR_ROOT, "code", "test", "mesh_test.vtk")

DIR_LABELS = os.path.join(DIR_ROOT, r"labels\segmentation_masks\FULL")
# DIR_NERD = os.path.join(DIR_ROOT, "version_1")
DIR_JSON = os.path.join(DIR_ROOT, r"labels\landmarks")
DIR_WRITE = os.path.join(DIR_ROOT, r"labels\segmentation_masks\LAA_ISOLATED\ARRAY_for_MESH")
# DIR_WRITE = os.path.join(DIR_ROOT, "LAA_ISOLATED", "MESH")
# DIR_IMAGES = os.path.join(DIR_ROOT, "ROUND1_LABELS")

def load_data(case: str): #case e.g., 2549981393_0000
    # READ JSON
    with open(os.path.join(DIR_JSON, f'{case.split("_0000")[0]}_OSTIUM.mrk.json'), 'r') as file:
        file_json = json.load(file)
    # READ NIFTI
    file_nifti = nib.load(os.path.join(DIR_LABELS, f'{case}.nii.gz'))
    return file_json, file_nifti

def landmark_voxels(file_nifti, file_json):
    affine = file_nifti.affine
    affine_inverse = np.linalg.inv(affine)
    landmark_orientation = file_json["markups"][0]["controlPoints"][0]["orientation"]
    orientation_matrix = np.array(landmark_orientation).reshape(3,3)
    landmark_points = [file_json["markups"][0]["controlPoints"][x]["position"] for x in [0,1,2]]
    landmark_points = np.dot(landmark_points, orientation_matrix)
    landmark_voxels = [return_voxel_coords(point, affine_inverse) for point in landmark_points]
    return landmark_voxels, affine


# READ META DATA
# meta_data = file_nifti.header
# pixel_spacing = meta_data['pixdim'][1:4]


# FUNCTIONS
def return_voxel_coords(landmark_point, affine_inverse):
    x,y,z = landmark_point
    physical_coordinates = np.array([x, y, z, 1])
    voxel_coordinates = np.dot(affine_inverse, physical_coordinates)
    return np.round(voxel_coordinates[:3]).astype(int)

# GRAB VOXEL COORDINATES


# RCONVERT NUMPY TO VTK ARRAY
# file_nifti = nib.load(path_nifti)





# arr_vtk = numpy_support.numpy_to_vtk(num_array=arr_nifti.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
# vtk_image = vtk.vtkImageData()
# vtk_image.SetDimensions(arr_nifti.shape)
# vtk_image.GetPointData().SetScalars(arr_vtk)

# # Applymarching CUBES
# marching_cubes = vtk.vtkMarchingCubes()
# marching_cubes.SetInputData(vtk_image)
# marching_cubes.SetValue(0,0.5)

# marching_cubes.Update()

# stl_writer = vtk.vtkSTLWriter()
# stl_writer.SetFileName(path_mesh)
# stl_writer.SetInputfion(marching_cubes.GetOutputPort())
# stl_writer.Write()






# for point in landmark_voxels:
#     y, x, z = point
#     arr[y-3:y+3,x-3:x+3,z-3:z+3] = 2000

# new_nifti = nib.Nifti1Image(arr, file_nifti.affine, file_nifti.header)
# nib.save(new_nifti, path_new_nifti)


#############


def calculate_centroid(point_1, point_2, point_3):
    x = (point_1[0] + point_2[0] + point_3[0])/3
    y = (point_1[1] + point_2[1] + point_3[1])/3
    z = (point_1[2] + point_2[2] + point_3[2])/3
    return x,y,z

def calculate_normal_vector(p1,p2,p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    return np.cross(v1, v2)

def calculate_angle_with_axes(normal):
    unit_vectors = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    angles = []
    for unit in unit_vectors:
        cos_angle = np.dot(normal, unit)/(np.linalg.norm(normal)*np.linalg.norm(unit))
        angle = np.arccos(cos_angle)
        angles.append(90-np.degrees(angle))
    return angles


def plane_cut(nifti_file, landmark_voxels: list, mesh: bool = False):
    arr = nifti_file.get_fdata()
    arr[arr!=1] = 0
    normal_vector = calculate_normal_vector(landmark_voxels[0],landmark_voxels[1],landmark_voxels[2])
    # angles = calculate_angle_with_axes(normal_vector)
    center = calculate_centroid(landmark_voxels[0],landmark_voxels[1],landmark_voxels[2])
    A, B, C = normal_vector 
    D = -np.dot(normal_vector, center)
    x,y,z = np.indices(np.shape(arr))
    # orientation = nib.orientations.io_orientation(nifti_file.affine)
    # axis_codes = nib.orientations.ornt2axcodes(orientation)
    
    mask = A*x + B*y + C*z + D > 0

    if normal_vector[0]>0:
        new_mask = ~np.copy(mask)
        # arr[~mask] = 0
    else:
        new_mask = np.copy(mask)
        # arr[mask] = 0

    arr[new_mask] = 0
    

    # RETURN ARRAY
    if not mesh:

        return arr, center, new_mask
    
    # RETURN MESH

    vtk_data = get_vtk_data(arr)
    mesh = mesh_nifti_vtk(vtk_data)

    plane = vtk.vtkPlane()
    plane.SetOrigin(A,B,C)
    plane.SetNormal(x,y,z)

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(mesh)
    cutter.Update()
    return True

# # select remaining ROI closest to centroid 

def select_laa(arr: np.array, center) -> np.array:
    labeled_arr, num_features = label(arr)
    center_arr = labeled_arr[int(center[0])-1:int(center[0])+2, int(center[1])-1:int(center[1])+2,int(center[2])-1:int(center[2])+2]
    center_vals = np.unique(center_arr)
    center_vals = center_vals[center_vals>0]
    if len(center_vals) == 0:
        return arr
    laa_val = center_vals[0]
    if len(center_vals) > 1:
        print('ERROR')
    arr[labeled_arr!=laa_val] = 0
    return arr

def save_nifti(arr: np.array, affine, header, case):
    write_path = os.path.join(DIR_WRITE, f'{case}.nii.gz')
    nifti_new = nib.Nifti1Image(arr, affine, header = header)
    nifti_new.to_filename(write_path)


# Example usage
# original_affine = img.affine  # Affine from the original NIfTI image
# target_affine = new_affinnibe  # New affine after reorientation
# voxel_transformation = ...  # The voxel-wise transformation matrix applied to reorient the image
# transformed_coordinates = transform_coordinates(coordinates, original_affine, target_affine, voxel_transformation)


import numpy as np
import nibabel as nib


def reorient_nifti_and_voxel(nifti_img, voxel_coords):
    # Load the NIfTI file
    # nifti_img = nib.load(nifti_file_path)
    
    # Get the original image's affine and data
    original_affine = nifti_img.affine
    data = nifti_img.get_fdata()

    # Determine the orientation of the current image and the desired orientation (RAS)
    current_orientation = nib.orientations.io_orientation(nifti_img.affine)
    target_orientation = nib.orientations.axcodes2ornt(('R', 'A', 'S'))

    # Get the transformation between the current orientation and the target orientation
    transformation = nib.orientations.ornt_transform(current_orientation, target_orientation)

    # Apply the transformation to reorient the image data to RAS
    data_RAS = nib.orientations.apply_orientation(data, transformation)

    # Calculate the new affine matrix for the reoriented image
    new_affine = nib.orientations.inv_ornt_aff(transformation, data.shape)

    # Convert voxel coordinates to physical space using the original affine
    new_voxel_coords = [rotate_voxel_coords(coord, original_affine, new_affine, np.shape(data_RAS)) for coord in voxel_coords]
    
    return data_RAS, new_voxel_coords

# def rotate_voxel_coords(voxel_coords, original_affine, new_affine, data_shape):
#     center = np.array(data_shape) / 2

#     translated_coords = np.array(voxel_coords) - center
#     voxel_coords_homogeneous = np.append(voxel_coords, 1)  # Convert to homogeneous coordinates for affine transformation
#     physical_coords = original_affine.dot(voxel_coords_homogeneous)[:3]
    
#     # Convert the physical space coordinates back to voxel coordinates using the new affine
#     new_voxel_coords_homogeneous = np.linalg.inv(new_affine).dot(np.append(physical_coords, 1))
#     return np.round(new_voxel_coords_homogeneous[:3]).astype(int)  # Round to nearest voxel and convert to integer 


def rotate_voxel_coords(voxel_coords, original_affine, new_affine, data_shape): 
    # Step 1: Determine the center of the image
    center = np.array(data_shape) / 2

    # Step 2: Translate coordinates to center
    translated_coords = np.array(voxel_coords) - center

    # Convert to homogeneous coordinates for affine transformation
    translated_coords_homogeneous = np.append(translated_coords, 1)

    # Apply the original affine to get physical coordinates
    physical_coords = original_affine.dot(translated_coords_homogeneous)[:3]

    # Step 3: Apply rotation (new affine) to get new voxel coordinates in physical space
    new_voxel_coords_homogeneous = np.linalg.inv(new_affine).dot(np.append(physical_coords, 1))

    # Convert back to voxel space and round to nearest voxel
    rotated_coords = np.round(new_voxel_coords_homogeneous[:3]).astype(int)

    # Step 4: Translate back by adding the center
    adjusted_coords = rotated_coords + center

    # Ensure coordinates are within the image bounds
    adjusted_coords = np.clip(adjusted_coords, 0, np.array(data_shape) - 1)

    return adjusted_coords

# # Usage example
# nifti_file_path = 'path_to_your_nifti_file.nii'
# original_voxel_coords = [10, 20, 40]  # Original voxel coordinates

# # Reorient image and get new voxel coordinates
# data_RAS, new_voxel_coords = reorient_nifti_and_voxel(nifti_file_path, original_voxel_coords)

# print("New Voxel Coordinates:", new_voxel_coords)
# voxel_value = data_RAS[tuple(new_voxel_coords)]



# Example usage
# nifti_file = 'path/to/your/nifti_file.nii'
# reoriented_data, new_affine = reorient_image_to(nifti_file, target_orientation='RAS')
from glob import glob 
import SimpleITK as sitk

def get_vtk_data(nifti_array):
    """
    Convert ostium array to a VTK object
    """
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(nifti_array.shape)
    vtk_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    for z in range(nifti_array.shape[2]):
        for y in range(nifti_array.shape[1]):
            for x in range(nifti_array.shape[0]):
                vtk_data.SetScalarComponentFromFloat(x,y,z,0,nifti_array[x,y,z])
 
    return vtk_data
 
def mesh_nifti_vtk(vtk_data):
    """
    Mesh the VTK object. Returns mesh as
    a VTK image
    """
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(vtk_data)
    marching_cubes.ComputeNormalsOn()
    marching_cubes.SetValue(0,1.0)
    #marching_cubes.SetValue(1,2.0)
    #marching_cubes.SetValue(2,3.0)
    #marching_cubes.SetValue(3,4.0)
    marching_cubes.Update()
 
    return marching_cubes.GetOutput()

def save_mesh(mesh, case):
    """
    Save the mesh as a VTK image
    """
    write_path = os.path.join(DIR_WRITE, f'{case}.vtk')
    mesh_save = vtk.vtkPolyDataWriter()
    mesh_save.SetFileName(write_path)
    mesh_save.SetInputData(mesh)
    mesh_save.Write()




def main():

    for case in tqdm([label.split(".nii.gz")[0] for label in os.listdir(DIR_LABELS)], desc="ISOLATING LAA"):
        json_path = os.path.join(DIR_JSON, f'{case.split("_0000")[0]}_OSTIUM.mrk.json')
        # try:
        if not os.path.exists(json_path): 
            continue
        file_json, file_nifti = load_data(case)
        # get landmarks from json
        landmarks, affine = landmark_voxels(file_nifti, file_json)
        # reorient 
        # data_RAS, new_voxel_coords = reorient_nifti_and_voxel(file_nifti, landmarks)
        # 
        # arr_cut, center = plane_cut(file_nifti, landmarks, mesh = True)
        arr_cut, center, plane_mask = plane_cut(file_nifti, landmarks, mesh = False)

        arr_laa = select_laa(arr_cut, center)

        arr_laa[plane_mask] = np.unique(arr_laa)[-1]

        # vtk_data = get_vtk_data(arr_laa)
        # mesh_laa = mesh_nifti_vtk(vtk_data)
        # save_mesh(mesh_laa, case)
        save_nifti(arr_laa, affine, file_nifti.header, case)



if __name__ == "__main__":
    main()


    

