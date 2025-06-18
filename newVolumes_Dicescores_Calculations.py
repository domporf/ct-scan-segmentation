import json
import skimage
import matplotlib.pyplot as plt
import os
import numpy as np

# Set the datasets directory
datasets_dir = "C:\\Datasets"

# Get the list of folders (datasets) within the datasets directory
folders = [name for name in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, name))]

# Define a function to calculate the interquartile range (IQR) for a given dataset
def calculate_iqr(data):
    sorted_data = np.sort(data)
    q1 = np.percentile(sorted_data, 25)
    q3 = np.percentile(sorted_data, 75)
    iqr = q3 - q1
    return q1, q3
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
times =[]
# Process each dataset
for folder in folders:
    folder_path = os.path.join(datasets_dir, folder)

    if os.path.isdir(folder_path):
        # Get the file paths for the required files
        mask_tcl_v_file = os.path.join(folder_path, "mask_tcl.v")
        mask_tcl_raw_file = os.path.join(folder_path, "mask_tcl.raw")
        dissected_aorta_file = os.path.join(folder_path, "Dissected_Aorta.npy")
        full_aorta_file = os.path.join(folder_path, "Full_Aorta.npy")
        time_file = os.path.join(folder_path, "time.npy")
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
        Time = np.load(time_file)
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
        times.extend(Time)

        print(f'Done with {folder}')


#Calculate median and interquartile
median_dice_scores_full = np.median(dice_scores_full)
median_dice_scores_false_lumen = np.median(dice_scores_false_lumen)
median_dice_scores_true_lumen = np.median(dice_scores_true_lumen)
median_volume_Full_Aorta = np.median(volume_Full_Aorta)
median_volume_Full_Reference_Image = np.median(volume_Full_Reference_Image)
median_volume_false_lumen = np.median(volume_false_lumen)
median_volume_true_lumen = np.median(volume_true_lumen)
median_volume_true_lumen_mask = np.median(volume_true_lumen_mask)
median_volume_false_lumen_mask = np.median(volume_false_lumen_mask)

# Calculate the IQR for each variable
iqr_dice_scores_full = calculate_iqr(dice_scores_full)
iqr_dice_scores_false_lumen = calculate_iqr(dice_scores_false_lumen)
iqr_dice_scores_true_lumen = calculate_iqr(dice_scores_true_lumen)
iqr_volume_Full_Aorta = calculate_iqr(volume_Full_Aorta)
iqr_volume_Full_Reference_Image = calculate_iqr(volume_Full_Reference_Image)
iqr_volume_false_lumen = calculate_iqr(volume_false_lumen)
iqr_volume_true_lumen = calculate_iqr(volume_true_lumen)
iqr_volume_true_lumen_mask = calculate_iqr(volume_true_lumen_mask)
iqr_volume_false_lumen_mask = calculate_iqr(volume_false_lumen_mask)

# Calculate mean/median and IQR Time
median_time = np.median(times)
sorted_time = np.sort(times)
Q1 = np.percentile(sorted_time, 25)
Q3 = np.percentile(sorted_time, 75)
IQR = Q3-Q1
iqr_time = Q1, Q3
# Print mean and standard deviation
print(f"IQR dice_scores_full:", iqr_dice_scores_full)
print(f"IQR dice_scores_false_lumen:", iqr_dice_scores_false_lumen)
print(f"IQR dice_scores_true_lumen:", iqr_dice_scores_true_lumen)
print(f"IQR volume_Full_Aorta:", iqr_volume_Full_Aorta)
print(f"IQR volume_Full_Reference_Image:", iqr_volume_Full_Reference_Image)
print(f"IQR volume_false_lumen:", iqr_volume_false_lumen)
print(f"IQR volume_true_lumen:", iqr_volume_true_lumen)
print(f"IQR volume_true_lumen_mask:", iqr_volume_true_lumen_mask)
print(f"IQR volume_false_lumen_mask:", iqr_volume_false_lumen_mask)
print(f"Interquartile range for time is:", iqr_time)
print(f"Median time is: {median_time}")
print("Median Dice Scores (Full):", median_dice_scores_full)
print("Median Dice Scores (False Lumen):", median_dice_scores_false_lumen)
print("Median Dice Scores (True Lumen):", median_dice_scores_true_lumen)
print("Median Volume (Full Aorta):", median_volume_Full_Aorta)
print("Median Volume (Full Reference Image):", median_volume_Full_Reference_Image)
print("Median Volume (False Lumen):", median_volume_false_lumen)
print("Median Volume (True Lumen):", median_volume_true_lumen)
print("Median Volume (True Lumen Mask):", median_volume_true_lumen_mask)
print("Median Volume (False Lumen Mask):", median_volume_false_lumen_mask)
