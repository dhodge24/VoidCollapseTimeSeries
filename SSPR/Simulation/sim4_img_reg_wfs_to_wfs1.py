"""

The purpose of this code is to register/align the simulated white fields (with speckle, blur, and noise) to the
the first white field. Since the beam varies in wavelength slightly, this changes the propagation distance and causes
the intensity images to dilate/contract. To alleviate this problem, we align the white fields to the first white field.
This captures and remove high frequency variations in our data and produce better flat-field corrected images.

"""

import os
import SimpleITK as sitk
from tifffile import imread
import numpy as np
from tqdm import tqdm
import shutil
from tifffile import imwrite


def clear_directory(dir_main, dir_sub):
    """
    Clear all files in the specified directory.
    """
    dir_path = os.path.join(dir_main, dir_sub)

    # Create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    else:
        # Delete all files in the directory if it already exists
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the directory
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def command_iteration(method):
    """Tells us the current resolution level and optimizer scales"""
    if method.GetOptimizerIteration() == 0:
        print(f"\tLevel: {method.GetCurrentLevel()}")
        print(f"\tScales: {method.GetOptimizerScales()}")
    print(f"#{method.GetOptimizerIteration()}")
    #print(f"\tMetric Value: {method.GetMetricValue():10.5f}")
    #print(f"\tLearningRate: {method.GetOptimizerLearningRate():10.5f}")

def command_multiresolution_iteration(method):
    """Tells us when we change resolutions"""
    print(f"\tStop Condition: {method.GetOptimizerStopConditionDescription()}")
    print("============= Resolution Change =============")


save_dfs = True
save_warped_wf = True
save_ffc_img = True
save_jac = False
save_gridlines = False

run_wfs = "561"
run_holo = "586"

# Directory where files are saved
dir_main = "/Users/danielhodge/Desktop/"
dir_sim_wfs_with_speckle = "run" + run_wfs + "_for_run" + run_holo + "_sim_wfs_with_speckle/"

# Files to import
tiff_wfs_with_speckle_filename = "run" + run_wfs + "_for_run" + run_holo + "_sim_wfs_with_speckle.tiff"

# Directories to create
dir_wfs_to_wf1_img_reg_sim_warped = "run_" + run_wfs + "_to_" + run_wfs + "_for_run" + run_holo + "_img_reg_sim_warped/"
dir_wfs_dfs_x = "run" + run_wfs + "_dfs/x/"
dir_wfs_dfs_y = "run" + run_wfs + "_dfs/y/"
dir_wfs_warped = "run" + run_wfs + "_warped_wfs/"
dir_wfs_ffc = "run" + run_wfs + "_ffc/"
dir_wfs_jac = "run" + run_wfs + "_jac/"
dir_wfs_gridlines = "run" + run_wfs + "_gridlines/"

# Files to save
tiff_wfs_to_wf1_warped_filename = ("run" + run_wfs + "_to_" + run_wfs + "_for_run" + run_holo +
                                   "_wfs_with_speckle_warped.tiff")

os.chdir(dir_main)  # Change directory to where you want to save the data
cwd = os.getcwd()

# Get the current working directory
print('Current working directory is: ', cwd)

# Import files white fields and object holograms
wfs = np.array(imread(dir_main + dir_sim_wfs_with_speckle + tiff_wfs_with_speckle_filename))


# Create directory -- if it already exists, skip. Also, clear the directory of all images.
dfx_loc = dir_main + dir_wfs_to_wf1_img_reg_sim_warped + dir_wfs_dfs_x  # Directory for displacement fields
if not os.path.exists(dfx_loc):
  os.makedirs(dfx_loc)
if os.path.exists(dir_main + dir_wfs_to_wf1_img_reg_sim_warped + dir_wfs_dfs_x):
    clear_directory(dir_main=dir_main + dir_wfs_to_wf1_img_reg_sim_warped, dir_sub=dir_wfs_dfs_x)

# Create directory -- if it already exists, skip. Also, clear the directory of all images.
dfy_loc = dir_main + dir_wfs_to_wf1_img_reg_sim_warped + dir_wfs_dfs_y  # Directory for displacement fields
if not os.path.exists(dfy_loc):
  os.mkdir(dfy_loc)
if os.path.exists(dir_main + dir_wfs_to_wf1_img_reg_sim_warped + dir_wfs_dfs_y):
    clear_directory(dir_main=dir_main + dir_wfs_to_wf1_img_reg_sim_warped, dir_sub=dir_wfs_dfs_y)

# Create directory -- if it already exists, skip. Also, clear the directory of all images.
warped_wf_loc = dir_main + dir_wfs_to_wf1_img_reg_sim_warped + dir_wfs_warped  # Directory for warped white fields
if not os.path.exists(warped_wf_loc):
  os.mkdir(warped_wf_loc)
if os.path.exists(dir_main + dir_wfs_to_wf1_img_reg_sim_warped + dir_wfs_warped):
    clear_directory(dir_main=dir_main + dir_wfs_to_wf1_img_reg_sim_warped, dir_sub=dir_wfs_warped)

# Create directory -- if it already exists, skip. Also, clear the directory of all images.
ffc_loc = dir_main + dir_wfs_to_wf1_img_reg_sim_warped + dir_wfs_ffc  # Directory for flat-field corrected images
if not os.path.exists(ffc_loc):
  os.mkdir(ffc_loc)
if os.path.exists(dir_main + dir_wfs_to_wf1_img_reg_sim_warped + dir_wfs_ffc):
    clear_directory(dir_main=dir_main + dir_wfs_to_wf1_img_reg_sim_warped, dir_sub=dir_wfs_ffc)

# Create directory -- if it already exists, skip. Also, clear the directory of all images.
jac_loc = dir_main + dir_wfs_to_wf1_img_reg_sim_warped + dir_wfs_jac  # Directory for Jacobian images
if not os.path.exists(jac_loc):
  os.mkdir(jac_loc)
if os.path.exists(dir_main + dir_wfs_to_wf1_img_reg_sim_warped + dir_wfs_jac):
    clear_directory(dir_main=dir_main + dir_wfs_to_wf1_img_reg_sim_warped, dir_sub=dir_wfs_jac)

# Create directory -- if it already exists, skip. Also, clear the directory of all images.
gridlines_loc = dir_main + dir_wfs_to_wf1_img_reg_sim_warped + dir_wfs_gridlines
if not os.path.exists(gridlines_loc):
  os.mkdir(gridlines_loc)
if os.path.exists(dir_main + dir_wfs_to_wf1_img_reg_sim_warped + dir_wfs_gridlines):
    clear_directory(dir_main=dir_main + dir_wfs_to_wf1_img_reg_sim_warped, dir_sub=dir_wfs_gridlines)

# Assign fixed and moving images
fixed = wfs[0, ...]
moving = wfs

# Convert numpy arrays to ITK image type so algorithm can be applied
fixed = sitk.GetImageFromArray(fixed)


### BEGIN REGISTRATION ###

# Set registration method and track progress
R = sitk.ImageRegistrationMethod()
R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
R.AddCommand(sitk.sitkMultiResolutionIterationEvent, lambda: command_multiresolution_iteration(R))

# Create initial displacement field
initial_df = sitk.Image(fixed.GetWidth(), fixed.GetHeight(), sitk.sitkVectorFloat64)
initial_df.CopyInformation(fixed)
initial_df_tx = sitk.DisplacementFieldTransform(initial_df)
del initial_df  # Clear some memory

# # Plot what the initial displacement field is before applying the image registration method
# init_df = sitk.TransformToDisplacementField(transform=initial_df_tx,
#                                        outputPixelType=sitk.sitkVectorFloat64,
#                                        size=fixed.GetSize(),
#                                        outputOrigin=fixed.GetOrigin(),
#                                        outputSpacing=fixed.GetSpacing(),
#                                        outputDirection=fixed.GetDirection())
# init_dfx = sitk.GetImageFromArray(np.float32(sitk.GetArrayFromImage(init_df)[:, :, 0]))
# init_dfy = sitk.GetImageFromArray(np.float32(sitk.GetArrayFromImage(init_df)[:, :, 1]))
# sitk.Show(init_dfx, "Initial df x")
# sitk.Show(init_dfy, "Initial df y")

# Regularization
initial_df_tx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=3.3, varianceForTotalField=3.3)

# Set initial transform (Need a good initial guess for better convergence)
R.SetInitialTransform(initial_df_tx, inPlace=False)  # inPlace=False -- initial_df_tx will NOT change every iteration

# Set similarity metric settings
R.SetMetricAsANTSNeighborhoodCorrelation(radius=2)

# Utilize the gradient of the fixed image when evaluating the registration metric
R.MetricUseFixedImageGradientFilterOn()

# Utilize the gradient of the moving image when evaluating the registration metric
R.MetricUseMovingImageGradientFilterOn()

# Set interpolator settings
R.SetInterpolator(sitk.sitkLinear)

# Set optimizer settings
R.SetOptimizerAsGradientDescent(learningRate=0.1,
                                numberOfIterations=300,
                                convergenceMinimumValue=1e-12,
                                convergenceWindowSize=20,
                                estimateLearningRate=R.EachIteration)
R.SetOptimizerScalesFromPhysicalShift()

# Set up the multi-resolution framework
R.SetShrinkFactorsPerLevel([256, 128, 64, 32, 16, 8, 4, 2])
R.SetSmoothingSigmasPerLevel([128, 64, 32, 16, 8, 4, 2, 2])
R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

wfs_aligned = []
for i in tqdm(range(len(wfs))):
    # Run the registration method and output the transformation
    moving_img = wfs[i]
    moving_img = sitk.GetImageFromArray(moving_img)

    outTx = R.Execute(moving=moving_img, fixed=fixed)

    # Displacement fields
    df = sitk.TransformToDisplacementField(transform=outTx,
                                           outputPixelType=sitk.sitkVectorFloat64,
                                           size=fixed.GetSize(),
                                           outputOrigin=fixed.GetOrigin(),
                                           outputSpacing=fixed.GetSpacing(),
                                           outputDirection=fixed.GetDirection())

    # x and y components of the displacement field
    dfx = sitk.GetImageFromArray(np.float32(sitk.GetArrayFromImage(df)[:, :, 0]))
    dfy = sitk.GetImageFromArray(np.float32(sitk.GetArrayFromImage(df)[:, :, 1]))
    if save_dfs:
        sitk.WriteImage(sitk.GetImageFromArray(np.float32(sitk.GetArrayFromImage(dfx))),
                        dfx_loc + f"dfx_evt_{i+1}.tiff")
        sitk.WriteImage(sitk.GetImageFromArray(np.float32(sitk.GetArrayFromImage(dfy))),
                        dfy_loc + f"dfy_evt_{i+1}.tiff")

    # Resample the image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(1)
    resampler.SetTransform(outTx)
    moving_warped = resampler.Execute(moving_img)
    if save_warped_wf:
        sitk.WriteImage(sitk.GetImageFromArray(np.float32(sitk.GetArrayFromImage(moving_warped))),
                        warped_wf_loc + f"warped_wf_evt_{i+1}.tiff")
        wfs_aligned.append(np.float32(sitk.GetArrayFromImage(moving_warped)))

    # Create and show the flat-field corrected image
    result = fixed / moving_warped
    if save_ffc_img:
        sitk.WriteImage(sitk.GetImageFromArray(np.float32(sitk.GetArrayFromImage(result))),
                        ffc_loc + f"ffc_evt_{i+1}.tiff")


    # Calculate Jacobian given a displacement field. jac > 1 = dilated, jac < 1 = contracted, jac = 1 = no deformation
    # Negative Jacobian values mean that the registration is non-diffeomorphic and not valid
    if save_jac:
        jac = sitk.DisplacementFieldJacobianDeterminant(image1=df)
        sitk.WriteImage(sitk.GetImageFromArray(np.float32(sitk.GetArrayFromImage(jac))),
                        jac_loc + f"jac_evt_{i+1}.tiff")

    # Generates displacement grid with altered lines (x and y displacements)
    if save_gridlines:
        # Create grid image that occupies the same spatial location as the fixed image
        grid_image = sitk.GridSource(outputPixelType=sitk.sitkFloat32,
                                     size=fixed.GetSize(),
                                     sigma=(1, 1),
                                     gridSpacing=(5.0, 5.0),
                                     spacing=fixed.GetSpacing(),
                                     origin=fixed.GetOrigin())
        grid_image.CopyInformation(moving_img)

        # Resample the grid
        grid_resampler = sitk.ResampleImageFilter()
        grid_resampler.SetReferenceImage(fixed)
        grid_resampler.SetInterpolator(sitk.sitkLinear)
        grid_resampler.SetDefaultPixelValue(0)
        grid_resampler.SetTransform(outTx)
        grid = grid_resampler.Execute(grid_image)
        sitk.WriteImage(sitk.GetImageFromArray(np.float32(sitk.GetArrayFromImage(grid))),
                        gridlines_loc + f"gridlines_evt_{i+1}.tiff")
        # grid = sitk.InvertIntensity(grid)  # Changes it to black background instead of white

if save_warped_wf:
    wfs_aligned = np.stack(wfs_aligned, axis=0)
    imwrite(dir_main + dir_wfs_to_wf1_img_reg_sim_warped + dir_wfs_warped +
            tiff_wfs_to_wf1_warped_filename, wfs_aligned)
