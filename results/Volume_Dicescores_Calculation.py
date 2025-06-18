import json
import skimage
import matplotlib.pyplot as plt
import os
import numpy as np

# Set the datasets directory
datasets_dir = "C:\\Datasets"

# Get the list of folders (datasets) within the datasets directory
folders = [name for name in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, name))]

# Initialize lists for storing dice scores and volumes
dice_scores_full = []
dice_scores_false_lumen = []
dice_scores_true_lumen = []
volume_Full_Aorta = []
volume_Full_Reference_Image = []
volume_false_lumen = []
volume_true_lumen = []
volume_true_lumen_mask = []
volume_false_lumen_mask = []

# Process each dataset
for folder in folders:
    folder_path = os.path.join(datasets_dir, folder)

    if os.path.isdir(folder_path):
        # Get the file paths for the required files
        mask_tcl_v_file = os.path.join(folder_path, "mask_tcl.v")
        mask_tcl_raw_file = os.path.join(folder_path, "mask_tcl.raw")
        dissected_aorta_file = os.path.join(folder_path, "Dissected_Aorta.npy")
        full_aorta_file = os.path.join(folder_path, "Full_Aorta.npy")
        # Load JSON file
        with open(mask_tcl_v_file, "r") as json_file:
            data = json.load(json_file)
        
        # Extract relevant data from JSON
        image_size = data["size"]
        image_spacing = data["spacing"]
        data_type = data['dataType']

        desired_spacing = np.array([image_spacing[0], image_spacing[1], 3])  # 0.25x.25x3
        factor = np.round(desired_spacing / image_spacing).astype(int)
        factor = tuple(factor)

        # Load reference image data
        reference_data = np.fromfile(mask_tcl_raw_file, dtype=data_type)
        reference_image = reference_data.reshape((image_size[2], image_size[0], image_size[1]))
        reference_image = skimage.measure.block_reduce(reference_image, block_size=(factor[2], factor[0], factor[1]), func=np.median)

        #Removing extra regions in Reference_Mask
        # Find the largest connected region
        reference_image_new = np.where(reference_image != 0, 1, reference_image)
        connected_regions, num_connected_regions = skimage.measure.label(reference_image_new, return_num=True)

        # Filter out regions other than 1
        region_sizes = np.bincount(connected_regions.ravel())
        region_sizes = region_sizes[1:]  # Removing the first element, as it corresponds to region label 0 (background)
        largest_region_label = np.argmax(region_sizes) + 1  # Adding 1 to get the correct label in the original image

        # Create a mask for the largest connected region with label 1
        largest_connected_region = (connected_regions == largest_region_label).astype(int)

        # Remove regions not in common with the largest connected region
        reference_image[np.logical_and(reference_image != 0, largest_connected_region == 0)] = 0
        
        # Load other required data
        Dissected_Aorta = np.load(dissected_aorta_file)
        Full_Aorta = np.load(full_aorta_file)
        
        #Remove Full Aorta when = to Dissected Aorta
        for i in range(Full_Aorta.shape[0]):

            # Check if the current elements of Full_Aorta and Dissected_Aorta slices are close
            if np.allclose(Full_Aorta[i], Dissected_Aorta[i]):
                # Create a new array of zeros with the same shape and data type as Dissected_Aorta[i]
                Dissected_Aorta[i] = np.zeros_like(Dissected_Aorta[i])
        True_Lumen = Full_Aorta - Dissected_Aorta
        Volume_True_Lumen = np.count_nonzero(True_Lumen)
        Volume_Full_Aorta = np.count_nonzero(Full_Aorta)
        Volume_Dissected_Aorta = np.count_nonzero(Dissected_Aorta)
        Volume_Reference_Image = np.count_nonzero(reference_image)
        Volume_Reference_Image_dissected = np.count_nonzero((reference_image != 1) & (reference_image != 0))
        Volume_Reference_Image_True_Lumen = np.count_nonzero(reference_image == 1)

        voxel_volume = desired_spacing[0] * desired_spacing[1] * desired_spacing[2]
        volume_Reference_Full = Volume_Reference_Image * voxel_volume / 1000  
        volume_Reference_False_Lumen =  Volume_Reference_Image_dissected * voxel_volume / 1000  
        volume_Full = Volume_Full_Aorta * voxel_volume / 1000  
        volume_False_Lumen = Volume_Dissected_Aorta * voxel_volume / 1000
        volume_True_Lumen = volume_Full - volume_False_Lumen
        volume_Reference_True_Lumen = Volume_Reference_Image_True_Lumen * voxel_volume / 1000

        # Create output arrays based on conditions
        output_array_full = np.where(reference_image != 0, 1, reference_image)
        output_array_reference_false_lumen = np.where((reference_image != 0) & (reference_image != 1), reference_image, 0)
        output_array_true_lumen = np.where((reference_image != 0) & (reference_image != 2), reference_image, 0)

        # Perform intersection calculations for dice scores
        intersection = np.logical_and(output_array_full, Full_Aorta)
        dice_score_full = 2.0 * np.sum(intersection) / (Volume_Reference_Image + Volume_Full_Aorta)

        intersection = np.logical_and(output_array_reference_false_lumen,  Dissected_Aorta)
        dice_score_false_lumen = 2.0 * np.sum(intersection) / (Volume_Reference_Image_dissected + Volume_Dissected_Aorta)
        
        intersection = np.logical_and(output_array_true_lumen,  True_Lumen)
        dice_score_true_lumen = 2.0 * np.sum(intersection) / (Volume_Reference_Image_True_Lumen + Volume_True_Lumen)

        # Append dice scores and volumes to respective lists
        dice_scores_full.append(dice_score_full)
        dice_scores_false_lumen.append(dice_score_false_lumen)
        dice_scores_true_lumen.append(dice_score_true_lumen)
        volume_Full_Aorta.append(volume_Full)
        volume_Full_Reference_Image.append(volume_Reference_Full)
        volume_false_lumen.append(volume_False_Lumen)
        volume_true_lumen.append(volume_True_Lumen)
        volume_true_lumen_mask.append(volume_Reference_True_Lumen)
        volume_false_lumen_mask.append(volume_Reference_False_Lumen)
        print(f'Done with {folder}')

# Calculate mean and standard deviation
mean_dice_scores_full = np.mean(dice_scores_full)
mean_dice_scores_false_lumen = np.mean(dice_scores_false_lumen)
mean_dice_scores_true_lumen = np.mean(dice_scores_true_lumen)
mean_volume_Full_Aorta = np.mean(volume_Full_Aorta)
mean_volume_Full_Reference_Image = np.mean(volume_Full_Reference_Image)
mean_volume_false_lumen = np.mean(volume_false_lumen)
mean_volume_true_lumen = np.mean(volume_true_lumen)
mean_volume_true_lumen_mask = np.mean(volume_true_lumen_mask)
mean_volume_false_lumen_mask = np.mean(volume_false_lumen_mask)

std_dice_scores_full = np.std(dice_scores_full)
std_dice_scores_false_lumen = np.std(dice_scores_false_lumen)
std_dice_scores_true_lumen = np.std(dice_scores_true_lumen)
std_volume_Full_Aorta = np.std(volume_Full_Aorta)
std_volume_Full_Reference_Image = np.std(volume_Full_Reference_Image)
std_volume_false_lumen = np.std(volume_false_lumen)
std_volume_true_lumen = np.std(volume_true_lumen)
std_volume_true_lumen_mask = np.std(volume_true_lumen_mask)
std_volume_false_lumen_mask = np.std(volume_false_lumen_mask)

# Print mean and standard deviation
print(f"Volume in mL (Reference Full Aorta Volume): Mean={mean_volume_Full_Reference_Image}, Std={std_volume_Full_Reference_Image}")
print(f"Volume in mL (Full Aorta Volume ): Mean={mean_volume_Full_Aorta}, Std={std_volume_Full_Aorta}")
print(f"Volume in mL (False Lumen Volume): Mean={mean_volume_false_lumen}, Std={std_volume_false_lumen}")
print(f"Volume in mL (True Lumen Volume): Mean={mean_volume_true_lumen}, Std={std_volume_true_lumen}")
print(f"Volume in mL (True Lumen Mask Volume): Mean={mean_volume_true_lumen_mask}, Std={std_volume_true_lumen_mask}")
print(f"Volume in mL (False Lumen Mask Volume): Mean={mean_volume_false_lumen_mask}, Std={std_volume_false_lumen_mask}")
print(f"Volume in mL (Full Aorta Dice): Mean={mean_dice_scores_full}, Std={std_dice_scores_full}")
print(f"Volume in mL (False Lumen Dice): Mean={mean_dice_scores_false_lumen}, Std={std_dice_scores_false_lumen}")
print(f"Volume in mL (True Lumen Dice): Mean={mean_dice_scores_true_lumen}, Std={std_dice_scores_true_lumen}")
