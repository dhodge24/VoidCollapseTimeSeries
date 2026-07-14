"""

The purpose of this code is to register/align the experimental white fields (with speckle, blur, and noise) to the
object holograms (with speckle, blur, and noise) to generate flat-field corrected images and displacement fields.

"""

# Import libraries
import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import shutil
from tifffile import imread

# Import custom modules


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
    if method.GetOptimizerIteration() == 0:
        print(f"\tLevel: {method.GetCurrentLevel()}")
        print(f"\tScales: {method.GetOptimizerScales()}")
    print(f"#{method.GetOptimizerIteration()}")
    #print(f"\tMetric Value: {method.GetMetricValue():10.5f}")
    #print(f"\tLearningRate: {method.GetOptimizerLearningRate():10.5f}")

def command_multiresolution_iteration(method):
    print(f"\tStop Condition: {method.GetOptimizerStopConditionDescription()}")
    print("============= Resolution Change =============")


apply_mask = False

save_dfs = True
save_warped_wf = True
save_ffc_img = True
save_jac = True
save_gridlines = True

# Choose whether or not to use the warped white fields to align to the dynamic image
use_warped_wfs = True

output_size = [3000, 3000]
crop_size = [2500, 2500]

run_wfs = "561"
run_holo = "572"

# Main directories
dir_main = "/Users/danielhodge/Desktop/"
dir_holo_preprocessed = "run" + run_holo + "_exp_preprocessed/"
if use_warped_wfs:
    # Directory for warped white fields to register to the dynamic image
    dir_wfs_to_wf1_img_reg_warped = "run_" + run_wfs + "_to_" + run_wfs + "_for_run" + run_holo + "_img_reg_exp_warped/"
else:
    dir_wfs_preprocessed = "run" + run_wfs + "_exp_preprocessed/"

# Files to import
if use_warped_wfs:
    tiffs_wfs_preprocessed_warped = "*.tiff"
else:
    tiffs_wfs_preprocessed = "run" + run_wfs + "_exp_preprocessed.tiff"
# tiffs_holo_exp_preprocessed = "run" + run_holo + "_exp_preprocessed.tiff"
tiffs_holo_exp_preprocessed = "run" + run_holo + "_exp_preprocessed_no_Talbot.tiff"


# Directories to create and save tiffs to
if use_warped_wfs:
    dir_wfs_to_holo_img_reg = "run_" + run_wfs + "_to_" + run_holo + "_img_reg_exp_warped/"
else:
    dir_wfs_to_holo_img_reg = "run_" + run_wfs + "_to_" + run_holo + "_img_reg_exp/"
dir_holos_dfs_x = "run" + run_holo + "_dfs/x/"
dir_holos_dfs_y = "run" + run_holo + "_dfs/y/"
dir_wfs_warped = "run" + run_wfs + "_warped_wf/"
dir_holos_ffc = "run" + run_holo + "_ffc/"
dir_holos_jac = "run" + run_holo + "_jac/"
dir_holos_gridlines = "run" + run_holo + "_gridlines/"


os.chdir(dir_main)  # Change directory to where you want to save the data
cwd = os.getcwd()

# Get the current working directory
print('Current working directory is: ', cwd)


if use_warped_wfs:
    wfs = np.array(imread(dir_main +
                          dir_wfs_to_wf1_img_reg_warped +
                          dir_wfs_warped +
                          tiffs_wfs_preprocessed_warped), dtype=np.float32)
else:
    wfs = np.array(imread(dir_main +
                          dir_wfs_preprocessed +
                          tiffs_wfs_preprocessed), dtype=np.float32)

holos = np.array(imread(dir_main + dir_holo_preprocessed + tiffs_holo_exp_preprocessed), dtype=np.float32)

# Create directory -- if it already exists, skip. Also, clear the directory of all images.
dfx_loc = dir_main + dir_wfs_to_holo_img_reg + dir_holos_dfs_x  # Directory for displacement fields
if not os.path.exists(dfx_loc):
  os.makedirs(dfx_loc)
if os.path.exists(dir_main + dir_wfs_to_holo_img_reg + dir_holos_dfs_x):
    clear_directory(dir_main=dir_main + dir_wfs_to_holo_img_reg, dir_sub=dir_holos_dfs_x)

# Create directory -- if it already exists, skip. Also, clear the directory of all images.
dfy_loc = dir_main + dir_wfs_to_holo_img_reg + dir_holos_dfs_y  # Directory for displacement fields
if not os.path.exists(dfy_loc):
  os.mkdir(dfy_loc)
if os.path.exists(dir_main + dir_wfs_to_holo_img_reg + dir_holos_dfs_y):
    clear_directory(dir_main=dir_main + dir_wfs_to_holo_img_reg, dir_sub=dir_holos_dfs_y)

# Create directory -- if it already exists, skip. Also, clear the directory of all images.
warped_wf_loc = dir_main + dir_wfs_to_holo_img_reg + dir_wfs_warped  # Directory for warped white fields
if not os.path.exists(warped_wf_loc):
  os.mkdir(warped_wf_loc)
if os.path.exists(dir_main + dir_wfs_to_holo_img_reg + dir_wfs_warped):
    clear_directory(dir_main=dir_main + dir_wfs_to_holo_img_reg, dir_sub=dir_wfs_warped)

# Create directory -- if it already exists, skip. Also, clear the directory of all images.
ffc_loc = dir_main + dir_wfs_to_holo_img_reg + dir_holos_ffc  # Directory for flat-field corrected images
if not os.path.exists(ffc_loc):
  os.mkdir(ffc_loc)
if os.path.exists(dir_main + dir_wfs_to_holo_img_reg + dir_holos_ffc):
    clear_directory(dir_main=dir_main + dir_wfs_to_holo_img_reg, dir_sub=dir_holos_ffc)

# Create directory -- if it already exists, skip. Also, clear the directory of all images.
jac_loc = dir_main + dir_wfs_to_holo_img_reg + dir_holos_jac  # Directory for Jacobian images
if not os.path.exists(jac_loc):
  os.mkdir(jac_loc)
if os.path.exists(dir_main + dir_wfs_to_holo_img_reg + dir_holos_jac):
    clear_directory(dir_main=dir_main + dir_wfs_to_holo_img_reg, dir_sub=dir_holos_jac)

# Create directory -- if it already exists, skip. Also, clear the directory of all images.
gridlines_loc = dir_main + dir_wfs_to_holo_img_reg + dir_holos_gridlines  # Directory for gridlines
if not os.path.exists(gridlines_loc):
  os.mkdir(gridlines_loc)
if os.path.exists(dir_main + dir_wfs_to_holo_img_reg + dir_holos_gridlines):
    clear_directory(dir_main=dir_main + dir_wfs_to_holo_img_reg, dir_sub=dir_holos_gridlines)

# Assign fixed and moving images
fixed = holos[0, ...].astype('float32')
movings = wfs.astype('float32')

# Only work with values inside the mask
ellipse_size_y = 0.90
ellipse_size_x = 0.90
transition_length_y = 1
transition_length_x = 1
fade_to_val = 0
num_segments = None
# # Might need this code for masking a region if the sample does not take the entire computational domain
# _, mask = fadeoutImage(img=fixed,
#                           fadeMethod='ellipse',
#                           ellipseSize=[ellipse_size_y, ellipse_size_x],
#                           transitionLength=[transition_length_y, transition_length_x],
#                           fadeToVal=fade_to_val,
#                           numSegments=num_segments,
#                           bottomApply=False)
# # mask = create_circular_mask(size=movings[0].shape[0], percentage=0.575, smooth_pixels=1)
# mask[mask > 0] = 1
# mask = mask.astype('bool')


# Convert numpy arrays to ITK image type so algorithm can be applied
fixed_img = sitk.GetImageFromArray(fixed)

### BEGIN REGISTRATION ###

# Set registration method and track progress
R = sitk.ImageRegistrationMethod()
R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
R.AddCommand(sitk.sitkMultiResolutionIterationEvent, lambda: command_multiresolution_iteration(R))

# Create initial displacement field
initial_df = sitk.Image(fixed_img.GetWidth(), fixed_img.GetHeight(), sitk.sitkVectorFloat64)
initial_df.CopyInformation(fixed_img)
initial_df_tx = sitk.DisplacementFieldTransform(initial_df)
del initial_df  # Clear some memory

# Regularization
initial_df_tx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=3.3, varianceForTotalField=3.3)

# Set initial transform (Need a good initial guess for better convergence)
R.SetInitialTransform(initial_df_tx, inPlace=False)  # inPlace=False -- initial_df_tx will NOT change every iteration

# Set similarity metric settings
R.SetMetricAsANTSNeighborhoodCorrelation(radius=2)
R.MetricUseFixedImageGradientFilterOn()  # Utilize the gradient of the fixed image when evaluating the reg. metric
R.MetricUseMovingImageGradientFilterOn()  # Utilize the gradient of the moving image when evaluating the reg. metric

# Set interpolator settings
R.SetInterpolator(sitk.sitkLinear)

# Set optimizer settings
R.SetOptimizerAsGradientDescent(learningRate=0.5,
                                numberOfIterations=500,
                                convergenceMinimumValue=1e-12,
                                convergenceWindowSize=20,
                                estimateLearningRate=R.EachIteration)
R.SetOptimizerScalesFromPhysicalShift()

# Set up the multi-resolution framework
R.SetShrinkFactorsPerLevel([256, 128, 64, 32, 16, 8, 4, 2])
R.SetSmoothingSigmasPerLevel([128, 64, 32, 16, 8, 4, 2, 2])


for i in tqdm(range(len(movings))):
    # Run the registration method and output the transformation

    # THIS BLOCK IS TO SCALE THE WFS TO THE DYNAMIC IMAGE (DIFFERENT PHOTON COUNTS)
    # Scaling for better registration performance
    mean_moving_img = np.mean(movings[i])
    mean_fixed_img = np.mean(fixed)
    moving_img = movings[i] * (mean_fixed_img / mean_moving_img)  # Scale moving_img to have the same mean as fixed_img

    # If the sample/object doesn't take up the whole computational domain, then a mask is needed so use the code below
    # mean_moving_img = np.mean(movings[i][mask])
    # mean_fixed_img = np.mean(fixed[mask])
    # # Scale moving_img to have the same mean as fixed_img
    # moving_img = movings[i] * (mean_fixed_img / mean_moving_img) * mask

    # Convert numpy arrays to ITK image type so algorithm can be applied
    moving_img = sitk.GetImageFromArray(moving_img)


    outTx = R.Execute(moving=moving_img, fixed=fixed_img)

    # Displacement fields
    df = sitk.TransformToDisplacementField(transform=outTx,
                                           outputPixelType=sitk.sitkVectorFloat64,
                                           size=fixed_img.GetSize(),
                                           outputOrigin=fixed_img.GetOrigin(),
                                           outputSpacing=fixed_img.GetSpacing(),
                                           outputDirection=fixed_img.GetDirection())

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
    resampler.SetReferenceImage(fixed_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(1)
    resampler.SetTransform(outTx)
    moving_warped = resampler.Execute(moving_img)
    if save_warped_wf:
        sitk.WriteImage(sitk.GetImageFromArray(np.float32(sitk.GetArrayFromImage(moving_warped))),
                        warped_wf_loc + f"warped_wf_evt_{i+1}.tiff")

    # Create and show the flat-field corrected image
    result = fixed_img / moving_warped
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
                                     size=fixed_img.GetSize(),
                                     sigma=(1, 1),
                                     gridSpacing=(5.0, 5.0),
                                     spacing=fixed_img.GetSpacing(),
                                     origin=fixed_img.GetOrigin())
        grid_image.CopyInformation(moving_img)

        # Resample the grid
        grid_resampler = sitk.ResampleImageFilter()
        grid_resampler.SetReferenceImage(fixed_img)
        grid_resampler.SetInterpolator(sitk.sitkLinear)
        grid_resampler.SetDefaultPixelValue(1)
        grid_resampler.SetTransform(outTx)
        grid = grid_resampler.Execute(grid_image)
        sitk.WriteImage(sitk.GetImageFromArray(np.float32(sitk.GetArrayFromImage(grid))),
                        gridlines_loc + f"gridlines_evt_{i+1}.tiff")
        # grid = sitk.InvertIntensity(grid)  # Changes it to black background instead of white -- optional preference
