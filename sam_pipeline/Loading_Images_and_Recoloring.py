import numpy as np
import json
import skimage.measure
import matplotlib.pyplot as plt
patient_name = '10CR-FU3'

with open(f"C:\\Users\\bethz\\Desktop\\Python_Tutorials\\Datasets\\{patient_name}\\data_tcl.v", "r") as json_file:
    data = json.load(json_file)

image_size = data["size"]
image_spacing = data["spacing"]
data_type = data['dataType']

# Load the JSON file and extract data type for reference data
with open(f"C:\\Users\\bethz\\Desktop\\Python_Tutorials\\Datasets\\{patient_name}\\mask_tcl.v", "r") as json_file:
    data_mask = json.load(json_file)
data_mask_type = data_mask['dataType']

#Open the binary file and display the image
raw_data = np.fromfile(f"C:\\Users\\bethz\\Desktop\\Python_Tutorials\\Datasets\\{patient_name}\\data_tcl.raw", dtype=data_type)

# Adjust the image size to match the available raw data
image_3D = np.reshape(raw_data, (image_size[2], image_size[0], image_size[1]))

# Downsampling
desired_spacing = np.array([image_spacing[0], image_spacing[1], 3])  # 0.25x0.25x1
factor = np.round(desired_spacing / image_spacing).astype(int)
factor = tuple(factor)
image_3D = skimage.measure.block_reduce(image_3D, block_size=(factor[2], factor[0], factor[1]), func=np.median)

# # Load mask reference data
reference_data = np.fromfile(f"C:\\Users\\bethz\\Desktop\\Python_Tutorials\\Datasets\\{patient_name}\\mask_tcl.raw", dtype=data_mask_type)

# Reshape mask reference data
reference_image = reference_data.reshape((image_size[2], image_size[0], image_size[1]))
reference_image = skimage.measure.block_reduce(reference_image, block_size=(factor[2], factor[0], factor[1]), func=np.median)

selected_masks = np.load(f"C:\\Users\\bethz\\Desktop\\Python_Tutorials\\Datasets\\{patient_name}\\Full_Aorta.npy")
dissected_mask = np.load(f"C:\\Users\\bethz\\Desktop\\Python_Tutorials\\Datasets\\{patient_name}\\Dissected_Aorta.npy")
dissected_mask[dissected_mask != 1] = 0
True_Lumen = selected_masks - dissected_mask
# Define custom colors
true_lumen_color = np.array([32/255,144/255,140/255, 0.9])
selected_masks_color = np.array([253/255,231/255,36/255, 0.9])

# Ensure True_Lumen and selected_masks have compatible shapes
True_Lumen = True_Lumen[..., np.newaxis] * true_lumen_color
selected_masks = dissected_mask[..., np.newaxis] * selected_masks_color

# Display True Lumen in rgba(32,144,140,255) and selected_masks in rgba(253,231,36,255)
plt.imshow(image_3D[153], cmap='gray', vmin=-100, vmax=400)
plt.imshow(dissected_mask[153])
#plt.imshow(True_Lumen[59])
plt.show()

#reference_image[reference_image != 0] = 1
# Display reference_image
plt.imshow(reference_image[10:, :, 65])
plt.colorbar()
plt.show()
