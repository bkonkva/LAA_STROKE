import nibabel as nib
from pathlib import Path
import dicom2nifti
import dicom2nifti.settings as settings
import matplotlib.pyplot as plt
#from nilearn.image import binarize_img
import os.path
#from nilearn.surface import surface
import numpy as np
import vtk
import sys
from utils import same_image_test
import json
from plane import import_json, plot_plane

## Import file

nifti_image = nib.load(r'C:\Users\OITNYNGUTIEA\OneDrive - Department of Veterans Affairs\Documents\ai-radiology-preprocessing\planes\1343552822_0000.nii.gz')
plane_file = r'C:\Users\OITNYNGUTIEA\OneDrive - Department of Veterans Affairs\Documents\ai-radiology-preprocessing\planes\OSTIUM_1343552822.json'

#nifti_path = sys.argv[1]

#if nifti_path.endswith('.nii.gz'):
#    pass
#else:

#    settings.disable_validate_slice_increment()
#    settings.disable_validate_slicecount()
#    settings.enable_resampling()
#    settings.set_resample_spline_interpolation_order(1)
#    settings.set_resample_padding(-1000)

#    dicom_directory = Path("./data/SCD_IMAGES_01/SCD0000101/Localizers_1")
#    dicom2nifti.convert_directory(dicom_directory, ".")

#nifti_img = nib.load(nifti_path)


#file = pydicom.dcmread('/Users/amygutierrez/Documents/ai-radiology-preproc/data/SCD_IMAGES_01/SCD0000101/Localizers_1/IM-0020-0001.dcm')
#plt.imshow(file.pixel_array)
#plt.show()

# plotting the images

def plotting3D_nifti(nifti_image, plane):
    nifti_array = nifti_image.get_fdata()
    non_zero = np.array(np.nonzero(nifti_array)).T
    data = nifti_array[non_zero[:,0], non_zero[:,1], non_zero[:,2]]

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    #x_dim, y_dim, z_dim = data.shape
    ax.plot_surface(plane[0],plane[1],plane[2], cmap='pink')

    #x, y, z = np.meshgrid(range(x_dim), range(y_dim), range(z_dim))
    ax.scatter(non_zero[:,0],non_zero[:,1],non_zero[:,2], c=data, cmap='gray',marker='.')
    plt.savefig(r'C:\Users\OITNYNGUTIEA\OneDrive - Department of Veterans Affairs\Documents\ai-radiology-preprocessing\planes\segmentation_mask.jpg')

    #  Plot plane
    #fig = plt.figure(figsize=(10,10))
    #ax.plot_surface(plane[0],plane[1],plane[2],alpha=0.2)
    plt.show()
#raise Exception(print('done'))
def plotting_nifti(nifti_image, nifti_image2=None,title=str):
    '''
    Will plot nifti image. Will overlay a mask (nifti_image2), if 
    provided. Otherwise will only plot nifti_image
    '''

    image = nifti_image.get_fdata()
    if len(image.shape) == 4:
        image = image[:,:,:,0]

    fig_rows = 2
    fig_cols = 2
    n_subplots = fig_rows * fig_cols
    n_slice = image.shape[2]
    step_size = n_slice // n_subplots
    plot_range = n_subplots * step_size
    start_stop = int((n_slice - plot_range) / 2)

    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])

    for idx, img in enumerate(range(start_stop, plot_range, step_size)):
        axs.flat[idx].imshow(image[:, :, img], cmap='gray')
        axs.flat[idx].axis('off')
    
    if nifti_image2 is not None:
        image2 = nifti_image2.get_fdata()
        if len(image2.shape) == 4:
            image2 = image2[:,:,:,0]
        for idx, img in enumerate(range(start_stop, plot_range, step_size)):
            axs.flat[idx].imshow(image2[:, :, img], cmap='jet', alpha=0.5)
            axs.flat[idx].axis('off')
            
    plt.tight_layout()
    fig.suptitle(title)
    plt.show()

def read_metadata(nifti_image, path_json):
    with open(path_json, 'r') as file:
        plane_json = json.load(file)

    # READ META DATA
    meta_data = nifti_image.header
    pixel_spacing = meta_data['pixdim'][1:4]
    affine = nifti_image.affine
    affine_inverse = np.linalg.inv(affine)


    landmark_orientation = plane_json["markups"][0]["controlPoints"][0]["orientation"]
    orientation_matrix = np.array(landmark_orientation).reshape(3,3)

    return pixel_spacing, affine_inverse, orientation_matrix, plane_json

# FUNCTIONS
def return_voxel_coords(landmark_point, affine_inverse):
    x,y,z = landmark_point
    physical_coordinates = np.array([x, y, z, 1])
    voxel_coordinates = np.dot(affine_inverse, physical_coordinates)
    return np.round(voxel_coordinates[:3]).astype(int)

def voxel_to_coord(nifti_image, plane_json, affine_inverse, orientation_matrix):
    # GRAB VOXEL COORDINATES
    landmark_points = [plane_json["markups"][0]["controlPoints"][x]["position"] for x in [0,1,2]]
    landmark_points = np.dot(landmark_points, orientation_matrix)
    landmark_voxels = [return_voxel_coords(point, affine_inverse) for point in landmark_points]

    # # REWRITE VOXEL VALUES FOR TEST
    # file_nifti = nib.load('./planes/1343552822_0000.nii.gz')
    # arr = file_nifti.get_fdata().copy()
    # for point in landmark_voxels:
    #     y, x, z = point
    #     arr[y-3:y+3,x-3:x+3,z-3:z+3] = 2000
    # path_new_nifti = './planes/new_nifti.nii.gz'

    # new_nifti = nib.Nifti1Image(arr, file_nifti.affine, file_nifti.header)
    # nib.save(new_nifti, path_new_nifti)

    return landmark_voxels


def mesh_nifti(nifti_img, spacing):

    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    nifti_array = nifti_img.get_fdata()
    if len(nifti_array.shape) > 3:
        nifti_array = nifti_array[:,:,:,0]
        verts, faces, normals, values = measure.marching_cubes(nifti_array, 0)
    elif len(nifti_array.shape) == 3:
        verts, faces, normals, values = measure.marching_cubes(nifti_array,0)
    #obj_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    #for i, f in enumerate(faces):
    #    obj_3d.vectors[i] = verts[f]
    verts = verts*spacing

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(np.min(verts[:,0]), np.max(verts[:,0]))
    ax.set_ylim(np.min(verts[:,1]), np.max(verts[:,1])) 
    ax.set_zlim(np.min(verts[:,2]), np.max(verts[:,2]))

    mesh_coordinates = verts[faces]

    mesh = Poly3DCollection(mesh_coordinates)
    #mesh.set_edgecolor('k')
    #ax.add_collection3d(mesh)
    #plt.tight_layout()
    #if center is not None:
    #    ax.scatter(center[0], center[1], center[2], c='red', marker='*', s=1000)

    return mesh, verts

def mesh_nifti_vtk(nifti_image):
    nifti_array = nifti_image.get_fdata()

    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(nifti_array.shape)

    vtk_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    for z in range(nifti_array.shape[2]):
        for y in range(nifti_array.shape[1]):
            for x in range(nifti_array.shape[0]):
                vtk_data.SetScalarComponentFromFloat(x,y,z,0,nifti_array[x,y,z])

    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(vtk_data)
    marching_cubes.ComputeNormalsOn()
    marching_cubes.SetValue(0,1.0)
    marching_cubes.SetValue(1,2.0)
    marching_cubes.SetValue(2,3.0)
    marching_cubes.SetValue(3,4.0)

    marching_cubes.Update()
    


    #output_port = contour_filter.GetOutputPort()

    poly_data = marching_cubes.GetOutput()

    #poly_data = vtk.vtkPolyData()
    #poly_data.SetPoints(output_port.GetPoints())
    #poly_data.SetPolys(output_port.GetPolys())

    verts = np.array([poly_data.GetPoint(i) for i in range(poly_data.GetNumberOfPoints())])
    #faces = np.array([list(poly_data.GetCell(i).GetPointId(j) for j in range(poly_data.GetCell(i).GetNumberOfPoints())) for i in range(poly_data.GetNumberOfCells())])
    #faces = np.concatenate(faces)
    faces = []
    for i in range(poly_data.GetNumberOfCells()):
        cell = poly_data.GetCell(i)
        for j in range(cell.GetNumberOfPoints()):
            faces.append(cell.GetPointId(j))

    faces = np.array(faces).reshape(-1,3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #ax.plot_trisurf(verts[:,0], verts[:,1], faces, verts[:,2], cmap='viridis')
    #plt.show()
    return verts, faces

# Plot DICOM image in nifti format
#plotting_nifti(nifti_img,'Sunnybrook Localizers_1')

# Binarize and plot
#bin_image = binarize_img(nifti_img, threshold="85%", mask_img=None)
#raise Exception(np.unique(nifti_img.get_fdata()))
#mesh_nifti(nifti_img)
#plotting_nifti(nifti_img)
#image_path = './data/preprocessed/1219145927_0000.nii.gz'


x = nifti_image.header['qoffset_x']
y = nifti_image.header['qoffset_y']
z = nifti_image.header['qoffset_z']

n_i, n_j, n_k = nifti_image.shape
center_i = (n_i - 1) // 2  # // for integer division
center_j = (n_j - 1) // 2
center_k = (n_k - 1) // 2
center_point = np.array((center_i, center_j, center_k))
#print(nifti_image.shape)

#mesh_nifti(nifti_image, center_point)

def nifti_metadata(image_path):
    import SimpleITK as sitk
    import json

    itk_image = sitk.ReadImage(image_path)
    header = {k: itk_image.GetMetaData(k) for k in itk_image.GetMetaDataKeys()}
    with open("header_itk_preprocessed.json", "w") as outfile:
        json.dump(header, outfile, indent=4)

def isolate_segments(nifti_image):
    nifti_array = nifti_image.get_fdata()
    unique_vals, idx = np.unique(nifti_array, return_index=True)
    return(unique_vals, idx)

#vals, idx = isolate_segments(nifti_image)
#print(len(idx))    

#sx, sy, sz = nifti_image.header.get_zooms()
#volume = sx*sy*sz
#print(sx, sy, sz)

def plotting_surfaces(verts, faces, plane):

    #  Plot plane
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(plane[0],plane[1],plane[2],alpha=0.2, cmap='viridis')

    # Plot mesh
    #ax.set_xlim(-200,400)
    #ax.set_ylim(-200,400) 
    #ax.set_zlim(-200,400)

    ax.plot_trisurf(verts[:,0], verts[:,1], faces, verts[:,2], cmap='gray')

    #mesh.set_edgecolor('k')
    #ax.add_collection3d(mesh)
    #plt.tight_layout()

    plt.show()
    #plt.savefig(r'C:\Users\OITNYNGUTIEA\OneDrive - Department of Veterans Affairs\Documents\ai-radiology-preprocessing\planes\plan_and_mesh_vtk.jpg')


pixel_spacing, affine_inverse, orientation_matrix, plane_json = read_metadata(nifti_image, plane_file)
landmark_voxel = voxel_to_coord(nifti_image, plane_json, affine_inverse, orientation_matrix)


#plotting_nifti(nifti_new, nifti_image2=None,title='Nifti_new')



#pixel_spacing = nifti_image.header.get_zooms()

#coordinates = import_json(plane_file)
surface = plot_plane(landmark_voxel, pixel_spacing)


#mesh, verts = mesh_nifti(nifti_image, pixel_spacing)

#plotting_surfaces(mesh, verts, surface)

#plotting3D_nifti(nifti_image, surface)
verts, faces = mesh_nifti_vtk(nifti_image)
plotting_surfaces(verts, faces, surface)

