import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import json
import sys
import skimage.measure
import time

patient_name = '12WE-BL'

# Function to overlay segmented mask onto the input image
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Function to show the input box
def show_box(box, ax):
    x_min, y_min = box[0]
    x_max, y_max = box[1]
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         fill=False, color='red', linewidth=2)
    ax.add_patch(rect)

# Function to show the points you placed
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]


# Importing the segment anything package
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

# Load the JSON file and extract image size and spacing
with open(f"C:\\Datasets\\{patient_name}\\data_tcl.v", "r") as json_file:
    data = json.load(json_file)

image_size = data["size"]
image_spacing = data["spacing"]
data_type = data['dataType']

#Load the JSON file and extract data type for reference data
with open(f"C:\\Datasets\{patient_name}\\mask_tcl.v", "r") as json_file:
    data_mask = json.load(json_file)
data_mask_type = data_mask['dataType']

# Open the binary file and display the image
raw_data = np.fromfile(f"C:\\Datasets\{patient_name}\\data_tcl.raw", dtype=data_type)

# Adjust the image size to match the available raw data
image_3D = np.reshape(raw_data, (image_size[2], image_size[0], image_size[1]))
desired_spacing= np.array([image_spacing[0],image_spacing[1],3]) #in mm (x,y,z)
factor=np.round(desired_spacing/image_spacing).astype(int)
factor=tuple(factor)
image_3D = skimage.measure.block_reduce(image_3D, block_size=(factor[2],factor[0],factor[1]), func=np.median)
#Load mask reference data
reference_data = np.fromfile(f"C:\\Datasets\\{patient_name}\\mask_tcl.raw",dtype = data_mask_type)
#Reshape mask reference data
reference_image = reference_data.reshape((image_size[2], image_size[0], image_size[1]))
reference_image = skimage.measure.block_reduce(reference_image,block_size = (factor[2], factor[0], factor[1]), func = np.median)

# Paths to the trained AI model
sam_checkpoint = "C:\\Aortic_Disection_Images/sam_vit_h_4b8939.pth"
# Model Type
model_type = "vit_h"

# Choice of device as processor, either CPU or CUDA
device = "cuda"

# Initializing the segment anything model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Initializing output masks variable
selected_masks = np.load(f'c:\\Datasets\\{patient_name}\\Full_Aorta.npy')
dissected_mask = np.load(f'c:\\Datasets\{patient_name}\\Dissected_Aorta.npy')
elapsed_times = np.load(f'c:\\Datasets\\{patient_name}\\time.npy').tolist()

# Color axis limits
cbar = [-100,400]

for i in range(69, image_3D.shape[0]):
    start_time = time.time()

    #Applying window function and converting to RGB
    image = (image_3D[i]-min(cbar))*(255/(max(cbar)-min(cbar)))
    image = np.clip(image, 0, 255)
    image = np.repeat(image[:,:,None],3, axis=2)
    image = image.astype(np.uint8)
   
    # Set the image for the predictor
    predictor.set_image(image)
    # Initializing user input prompt vectors
    input_box = np.zeros((2, 2))  # Initialize the input box
    points = np.empty((0, 2), dtype=np.float32)  # Preallocate memory for points
    labels = np.empty(0,dtype=int)

    # Showing ground-truth mask
    f = plt.figure(figsize=(10,10))
    f.canvas.manager.window.wm_geometry('+800+10')
    plt.imshow(image, cmap='gray', vmin=-100, vmax=400)
    show_mask(reference_image[i], plt.gca())
    plt.title("Reference Mask")
    plt.show(block=False)
    plt.axis('off')

    continue_loop = True
    #Loop to define snd refine input box coordinates
    while continue_loop:
        # Show the image
        plt.figure()
        plt.imshow(image, cmap='gray', vmin=-100, vmax=400)
        plt.title("Draw Box")
        # Draw box
        input_box = np.array(plt.ginput(2, timeout=-1))

        # Plot horizontal and vertical lines at the clicked points
        for point in input_box:
            plt.axvline(point[0], color='r', linestyle='--', linewidth=1)
            plt.axhline(point[1], color='r', linestyle='--', linewidth=1)

        # Pause the execution and wait for a key press to continue
        plt.pause(0.001)
        user_input = input("Type 'c' to continue or 'r' to redo the box: ")

        # Close the image
        plt.close()

        if user_input.lower() == "c":
            # Continue with the current box
            continue_loop = False
        elif user_input.lower() == "r":
            # Redo the box creation
            continue

    # AI model to derive segmentation frpm just box coordinates
    masks, score, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=True,
    )

    fig, axs = plt.subplots(1, masks.shape[0] , figsize=(15, 5))
    #For loop to show each masks and their corresponding score
    for j, ax in enumerate(axs):
        ax.imshow(image)
        show_mask(masks[j], ax)
        ax.set_title(f"Mask {j} (Score: {score[j]})", fontsize = 10)  
    plt.show(block=False)

    #User chooses best image
    user_choice = input("Please enter the number of the best image: " )
    #Makes sure the user choice number is within range
    user_choice = int(user_choice)
    plt.close()

    if 0 <= user_choice < masks.shape[0]:
        print(f"You chose Mask:{user_choice}" )
        #Saving the chosen subplot as another figure
        plt.figure(figsize=(10,10))
        plt.imshow(image, cmap='gray', vmin=-100, vmax=400)
        show_mask(masks[user_choice], plt.gca())
        plt.title("Your Chosen Mask")
        plt.show(block=False)
        plt.axis('off')
    else:
        print("Invalid Number.")

    # Ask the user if they like the model
    response = input("Do you like the model? (yes/no): ")
    plt.close()

    # Storing the compressed user selected mask
    mask_input = logits[user_choice,:,:]

    if response.lower() == "yes":
        # Save the model
        selected_masks[i] = masks[user_choice]
    elif response.lower() == "no":
        # Store the current segmented mask as the previous mask
        prev_mask = masks[user_choice]

        # Code2 - Interactive point selection
        while True:
            # Show the image
            plt.figure()
            plt.imshow(image, cmap='gray', vmin=-100, vmax=400)
            plt.title("Add Points")
            show_mask(prev_mask, plt.gca())
            # Show input clicks as points
            plt.scatter(points[:, 0], points[:, 1], color='blue', marker='o', s=50, edgecolor='white')
            pts = np.asarray(plt.ginput(-1, timeout=-1))
            # Close the image
            plt.close()

            # Append newly selected points to the existing points list
            points = np.concatenate((points, pts))
            # Input points and labels for segmentation
            labels = np.concatenate((labels,np.ones(len(pts), dtype=int)))
            
            refined_masks, score, logits = predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=input_box,
                mask_input = mask_input[None,:,:,],
                multimask_output=True,
            )

            fig, axs = plt.subplots(1, refined_masks.shape[0], figsize=(15, 5))
            #For loop to show each masks and their corresponding score
            for k, ax in enumerate(axs):
                ax.imshow(image)
                show_mask(refined_masks[k], ax)
                ax.set_title(f"Mask {k} (Score: {score[k]})", fontsize = 10)
            plt.show(block=False)  

            #User chooses best image
            user_choice = input("Please enter the number of the best image: " )

            #Makes sure the user choice number is within range
            user_choice = int(user_choice)
            plt.close()

            del mask_input
            mask_input = logits[user_choice,:,:]

            if 0 <= user_choice < refined_masks.shape[0]:
                print(f"You chose Mask:{user_choice}" )
                #Saving the chosen subplot as another figure
                plt.figure(figsize=(10,10))
                plt.imshow(image, cmap='gray', vmin=-100, vmax=400)
                show_mask(refined_masks[user_choice], plt.gca())
                plt.title("Your Chosen Mask")
                plt.axis('off')
                plt.show(block=False)
            else:
                print("Invalid Number.")  

            # Ask the user if they like the model
            response = input("Do you like the model? (yes/no): ")
            plt.close()

            if response.lower() == "yes":
                # Save the model and exit the loop
                selected_masks[i] = refined_masks[user_choice]
                break
            elif response.lower() == "no":
                prev_mask = refined_masks[user_choice]
                continue
    np.save(f'C:\\Datasets\\{patient_name}\\Full_Aorta.npy',selected_masks)

    remove_regions = input("would you like to remove regions of the mask?(yes/no):")
    if remove_regions.lower() == "yes":
        prev_dissected_mask=selected_masks[i]
        # Initialize an empty list to store the points for region removal
        while True:
            # Show the image
            plt.figure()
            plt.imshow(image, cmap='gray', vmin=-100, vmax=400)
            plt.title("Specify Region to Remove")
            # Show input clicks as points
            show_mask(prev_dissected_mask, plt.gca())
            remove_pts = np.asarray(plt.ginput(-1, timeout=-1))
            # Close the image
            plt.close()

            # Plot all points (input points + remove points) for visualization
            points = np.concatenate((points, remove_pts))
            labels = np.concatenate((labels, np.zeros(len(remove_pts), dtype=int)))

            cropped_masks, score, logits = predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=input_box,
                mask_input=mask_input[None, :, :],
                multimask_output=True,
            )
            
            fig, axs = plt.subplots(1, cropped_masks.shape[0], figsize=(15, 5))
            # For loop to show each mask and its corresponding score
            for k, ax in enumerate(axs):
                ax.imshow(image)
                show_mask(cropped_masks[k], ax)
                ax.set_title(f"Mask {k} (Score: {score[k]})", fontsize=10)
            plt.show(block=False)
            # User chooses the best image
            user_choice = input("Please enter the number of the best image: ")

            # Make sure the user choice number is within range
            user_choice = int(user_choice)
            plt.close()

            del mask_input
            mask_input=logits[user_choice]

            if 0 <= user_choice < cropped_masks.shape[0]:
                print(f"You chose Mask: {user_choice}")
                # Saving the chosen subplot as another figure
                plt.figure(figsize=(10, 10))
                plt.imshow(image, cmap='gray', vmin=-100, vmax=400)
                show_mask(cropped_masks[user_choice], plt.gca())
                plt.title("Your Chosen Mask")
                plt.axis('off')
                plt.show(block=False)
            else:
                print("Invalid Number.")

            # Ask the user if they like the model
            response = input("Do you like the model? (yes/no): ")
            plt.close()

            if response.lower() == "yes":
                # Save the model and exit the loop
                dissected_mask[i] = cropped_masks[user_choice]
                break
            elif response.lower() == "no":
                prev_dissected_mask = cropped_masks[user_choice]  # Store the current segmented mask as the previous mask
                continue
            else:
                print("ERROR")
        np.save(f'C:\\Datasets\\{patient_name}\\Dissected_Aorta.npy',dissected_mask)        
    elif remove_regions.lower() == 'no':
        dissected_mask[i]=selected_masks[i]
        np.save(f'C:\\Datasets\\{patient_name}\\Dissected_Aorta.npy',dissected_mask)
    plt.close('all')
    print("done with model", i+1)
    end_time = time.time()
    elapsed_time=(end_time-start_time)
    print(elapsed_time,"sec")
    elapsed_times.append(elapsed_time)
    # elapsed_times = np.array(elapsed_times)
    np.save(f'C:\\Datasets\\{patient_name}\\time.npy',elapsed_times)
