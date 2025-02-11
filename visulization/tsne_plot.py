import numpy as np
from matplotlib import cm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# File paths
sample_names_1 = "tsne_outputs/goup_1_names.txt"
all_outputs_1 = "tsne_outputs/goup_1.py.npy"

sample_names_2 = "tsne_outputs/goup_2_names.txt"
all_outputs_2 = "tsne_outputs/goup_2.py.npy"

sample_names_3 = "tsne_outputs/goup_3_names.txt"
all_outputs_3 = "tsne_outputs/goup_3.py.npy"


# Ensure output directory exists
output_dir = "./tsne_outputs"
os.makedirs(output_dir, exist_ok=True)

# Load `sample_names` from text files and concatenate
with open(sample_names_1, "r") as file:
    sample_names_list_1 = [line.strip() for line in file.readlines()]
with open(sample_names_2, "r") as file:
    sample_names_list_2 = [line.strip() for line in file.readlines()]
with open(sample_names_3, "r") as file:
    sample_names_list_3 = [line.strip() for line in file.readlines()]

# Concatenate the lists
sample_names = sample_names_list_1 +  sample_names_list_2+ sample_names_list_3

# Load `all_outputs` from .npy files and concatenate
all_outputs_array_1 = np.load(all_outputs_1)
all_outputs_array_2 = np.load(all_outputs_2)
all_outputs_array_3 = np.load(all_outputs_3)

# Concatenate the arrays
all_outputs = np.vstack((all_outputs_array_1, all_outputs_array_2, all_outputs_array_3))

# Ensure shapes match
print(f"Loaded and concatenated sample_names: {len(sample_names)} names")
print(f"Loaded and concatenated all_outputs: {all_outputs.shape}")

# Ensure the number of `sample_names` matches the number of rows in `all_outputs`
assert len(sample_names) == all_outputs.shape[0], "Mismatch between sample names and all_outputs rows!"

# Generate unique colors for classes in `sample_names`
unique_names = list(set(sample_names))
unique_names = sorted(unique_names)
print("unique_names--------------------", unique_names)
color_map = {name: idx for idx, name in enumerate(unique_names)}
colors = [color_map[name] for name in sample_names]

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(all_outputs)

# Plot the t-SNE results (without the legend)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    tsne_results[:, 0], tsne_results[:, 1], 
    c=colors, cmap=cm.get_cmap('tab20', len(unique_names)), s=10
)
plt.title("t-SNE of Convert Layer Outputs")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
output_path_main = os.path.join(output_dir, "tsne_plot_without_legend.png")
plt.savefig(output_path_main, dpi=300, bbox_inches='tight')
plt.close()

# Generate a separate legend image
colormap = cm.get_cmap('tab20', len(unique_names))
color_values = [colormap(i / len(unique_names)) for i in range(len(unique_names))]

plt.figure(figsize=(4, 8))
handles = [
    plt.Line2D([0], [0], marker='o', color=color_values[idx], linestyle='', markersize=6)
    for idx in range(len(unique_names))
]
plt.legend(handles, unique_names, title="Classes", loc="center", fontsize='small', fancybox=True)
plt.axis('off')
output_path_legend = os.path.join(output_dir, "tsne_legend.png")
plt.savefig(output_path_legend, dpi=300, bbox_inches='tight')
plt.close()

print(f"Main plot saved as {output_path_main}")
print(f"Legend saved as {output_path_legend}")
