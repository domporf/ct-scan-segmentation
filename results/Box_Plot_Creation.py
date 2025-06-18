import matplotlib.pyplot as plt

# Sample data for True Lumen, False Lumen, and Full Aorta for ground truth mask and SAM model
true_lumen_ground_truth = [40, 50, 55, 60, 65]
false_lumen_ground_truth = [20, 30, 35, 40, 45]
full_aorta_ground_truth = [70, 80, 85, 90, 95]

true_lumen_sam_model = [45, 55, 60, 65, 70]
false_lumen_sam_model = [30, 35, 40, 45, 50]
full_aorta_sam_model = [75, 85, 90, 95, 100]

# Data for the x-axis labels
categories = ['True Lumen', 'False Lumen', 'Full Aorta']

# Create a figure and axis
fig, ax = plt.subplots()

# Create boxplot for ground truth mask
bp_gt = ax.boxplot([true_lumen_ground_truth, false_lumen_ground_truth, full_aorta_ground_truth],
                   positions=[1, 2, 3], widths=0.4, patch_artist=True, medianprops=dict(color='black'),
                   whiskerprops=dict(color='black'), boxprops=dict(facecolor='lightgreen', edgecolor='black'))

# Create boxplot for SAM model
bp_sam = ax.boxplot([true_lumen_sam_model, false_lumen_sam_model, full_aorta_sam_model],
                    positions=[1.45, 2.45, 3.45], widths=0.4, patch_artist=True, medianprops=dict(color='black'),
                    whiskerprops=dict(color='black'), boxprops=dict(facecolor='lightcoral', edgecolor='black'))

# Set y-axis label
ax.set_ylabel('Volume (mL)', fontsize=12)

# Set x-axis tick positions and labels
ax.set_xticks([1.2, 2.2, 3.2])
ax.set_xticklabels(categories, fontsize=12)

# Add a legend
green_patch = plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='black', label='Ground Truth Mask')
red_patch = plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', edgecolor='black', label='SAM Model')
ax.legend(handles=[green_patch, red_patch], loc='upper left', fontsize=10)

# Customize plot aesthetics
plt.title('Volume Comparison', fontsize=14, fontweight='bold')
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='gray')
ax.set_axisbelow(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

# Apply background color to the plot
fig.patch.set_facecolor('white')

# Save the plot as an image
plt.savefig('box_plot.png', dpi=300)

# Display the plot
plt.show()
