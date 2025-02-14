import json
import numpy as np
import logging
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.helpers

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
from utils.average_meter import AverageMeter
from transformers import AutoProcessor, FlavaModel
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
import torchvision.transforms as transforms
from PIL import Image
import os 
###############################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def get_volume_views(volume, save_path=None, elev=30, azim=45):
    """
    Generates a voxel plot from a 3D volume, rotates it, and optionally saves it.
    
    Args:
        volume (numpy.ndarray): A 3D numpy array representing the voxel grid.
        save_path (str, optional): If provided, saves the plot as an image. Default is None.
        elev (int, optional): Elevation angle for 3D rotation. Default is 30 degrees.
        azim (int, optional): Azimuth angle for 3D rotation. Default is 45 degrees.
    
    Returns:
        numpy.ndarray: The voxel plot as a NumPy array (H, W, 3).
    """
    volume = volume.squeeze() >= 0.5  # Apply thresholding if needed

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    # Remove background grid and lines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_frame_on(False)
    ax.grid(False)
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Plot the voxels
    ax.voxels(volume, edgecolor="k")

    # Rotate the shape
    ax.view_init(elev=elev, azim=azim)  # Rotate the view

    # Save the figure if save_path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
        plt.savefig(save_path + ".png", bbox_inches='tight', pad_inches=0, dpi=300)  # Save as PNG
        print(f"Saved voxel plot to {save_path}.png")

    # plt.show()  # Display the plot



def ensure_fixed_shape(tensor, target_length=202):
    """
    Ensures the sequence length of the tensor is fixed to `target_length`.
    Pads or truncates as needed.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, N, D].
        target_length (int): Desired sequence length.
    
    Returns:
        torch.Tensor: Tensor of shape [B, target_length, D].
    """
    batch_size, seq_length, feature_dim = tensor.shape

    if seq_length < target_length:
        # Pad the tensor to the right
        padding = torch.zeros((batch_size, target_length - seq_length, feature_dim), device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=1)
    elif seq_length > target_length:
        # Truncate the tensor
        tensor = tensor[:, :target_length, :]

    return tensor


def test_net_flava(cfg,vlm2pix, epoch_idx=-1, test_data_loader=None, test_writer=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                       batch_size=1,
                                                       num_workers=cfg.CONST.NUM_WORKER,
                                                       pin_memory=True,
                                                       shuffle=False)


    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    encoder_losses = AverageMeter()
    refiner_losses = AverageMeter()

    # Switch models to evaluation mode
    vlm2pix.eval()

    for sample_idx, (taxomony_class, taxonomy_id, sample_name, rendering_images, ground_truth_volume) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_volume = utils.helpers.var_or_cuda(ground_truth_volume)


            encoder_loss, refiner_loss, generated_volume = vlm2pix(rendering_images,ground_truth_volume, taxomony_class )


            # Append loss and accuracy to average metrics
            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou.append((intersection / union).item())

            # IoU per taxonomy
            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)
                    ## and sample_iou[2]>0.97
            if (cfg.TEST.VISUALIZATION == True and sample_name=="1a6f615e8b1b5ae4dbbc9440457e303e"  
            and taxomony_class[0].split()[-1]=="chair" ):
                print("generated_volume----------:", generated_volume.shape)
                print("rendering_images----------:", rendering_images.shape)
                print("sample_iou----------------:", sample_iou[2])
                print("sample_name---------------:",sample_name)
                print("taxonomy_id---------------:",taxonomy_id)
                print("taxomony_class------------:",taxomony_class[0].split()[-1])
                
                ## GT
                save_path =f"./visulization_output/{taxomony_class[0].split()[-1]}_{sample_name}_{round(sample_iou[2],3)}_Pred"
                get_volume_views(generated_volume.cpu().numpy(), save_path=save_path, elev=60, azim=30)
                
                ## predicted 
                save_path =f"./visulization_output/{taxomony_class[0].split()[-1]}_{sample_name}_{round(sample_iou[2],3)}_GT"
                get_volume_views(ground_truth_volume.cpu().numpy(), save_path=save_path,elev=60, azim=30)

                exit()

            # Append generated volumes to TensorBoard
            if test_writer and sample_idx < 3:
                # Volume Visualization
                rendering_views = utils.helpers.get_volume_views(generated_volume.cpu().numpy())
                rendering_views = rendering_views.transpose(2, 0, 1)  # Convert from (H, W, C) to (C, H, W)

                test_writer.add_image('Model%02d/Reconstructed' % sample_idx, rendering_views, epoch_idx)
                rendering_views = utils.helpers.get_volume_views(ground_truth_volume.cpu().numpy())
                rendering_views = rendering_views.transpose(2, 0, 1) 
                test_writer.add_image('Model%02d/GroundTruth' % sample_idx, rendering_views, epoch_idx)

            # Print sample loss and IoU
            logging.info('Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f RLoss = %.4f IoU = %s' %
                         (sample_idx + 1, n_samples, taxonomy_id, sample_name, encoder_loss.item(),
                          refiner_loss.item(), ['%.4f' % si for si in sample_iou]))
        # break
    # Output testing results
    mean_iou = []
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print header
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in test_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
        if 'baseline' in taxonomies[taxonomy_id]:
            print('%.4f' % taxonomies[taxonomy_id]['baseline']['%d-view' % cfg.CONST.N_VIEWS_RENDERING], end='\t\t')
        else:
            print('N/a', end='\t\t')

        for ti in test_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print()
    # Print mean IoU for each threshold
    print('Overall ', end='\t\t\t\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)
    if test_writer is not None:
        test_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx)
        # test_writer.add_scalar('Refiner/BatchLoss', refiner_losses.avg, epoch_idx)

        test_writer.add_scalar('Refiner/IoU', max_iou, epoch_idx)

    return max_iou





############################################### tsne model 

def test_tsne(cfg,vlm2pix, epoch_idx=-1, test_data_loader=None, test_writer=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                       batch_size=1,
                                                       num_workers=cfg.CONST.NUM_WORKER,
                                                       pin_memory=True,
                                                       shuffle=False)


    # # Set up loss functions
    # bce_loss = torch.nn.BCELoss()

    # Testing loop
    n_samples = len(test_data_loader)
    # test_iou = dict()
    # encoder_losses = AverageMeter()
    # refiner_losses = AverageMeter()

    # Switch models to evaluation mode
    vlm2pix.eval()

    # Initialize a list to track the sample indices
    sample_indices = []
    sample_names = []

    for sample_idx, (taxomony_class, taxonomy_id, sample_name, rendering_images, ground_truth_volume) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_volume = utils.helpers.var_or_cuda(ground_truth_volume)

            encoder_loss, refiner_loss, generated_volume  = vlm2pix(rendering_images,ground_truth_volume, taxomony_class )
            # print("output.shape---------------------------:",generated_volume.shape)

            # Register the hook
            hook = vlm2pix.convert_layer.module.register_forward_hook(hook_fn)

            # Run the forward pass
            output = vlm2pix(rendering_images,ground_truth_volume, taxomony_class)

            # Access the convert_layer output
            print("Convert Layer Output----------------------:", convert_layer_output.shape)
            print("sample_name, taxomony_class, taxonomy_id--------------:",sample_name, taxomony_class[0].split()[-1], taxonomy_id)
            # Remove the hook when done
            hook.remove()

            # Track the sample index
            sample_indices.append(convert_layer_output.view(convert_layer_output.size(0), -1).detach().cpu().numpy() )
            sample_names.append(taxomony_class[0].split()[-1])
            if sample_idx >= cfg.PLOT.NUM_SAMPLES:
                break  

    # Example: If `sample_names` contains class labels, map each unique name to a unique color
    unique_names = list(set(sample_names))
    color_map = {name: idx for idx, name in enumerate(unique_names)}

    # Map sample names to their corresponding colors
    colors = [color_map[name] for name in sample_names]

    # Stack all captured outputs into a single array
    all_outputs = np.vstack(sample_indices)  # Shape: [100, 12544] for 100 samples

    # Save the array
    np.save(f"./tsne_outputs/goup_{cfg.PLOT.GROUP}.py", all_outputs)

    file_path = f"./tsne_outputs/goup_{cfg.PLOT.GROUP}_names.txt"

    # Write the names to a text file
    with open(file_path, "w") as file:
        for name in sample_names:
            file.write(name + "\n")

    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    print("all_outputs------------------------", all_outputs.shape)
    tsne_results = tsne.fit_transform(all_outputs)
    print("tsne_results-----------------------", tsne_results.shape)


    # Plot the t-SNE results (without the legend)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        tsne_results[:, 0], tsne_results[:, 1], 
        c=colors, cmap=cm.get_cmap('tab20', len(unique_names)), s=10
    )
    plt.title("t-SNE of Convert Layer Outputs for First 100 Samples")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    output_path_main = "tsne_plot_without_legend.png"
    plt.savefig(output_path_main, dpi=300, bbox_inches='tight')
    plt.close()

    # Generate a separate legend image
    # Map colors to their actual RGB values using the colormap
    colormap = cm.get_cmap('tab20', len(unique_names))
    color_values = [colormap(i / len(unique_names)) for i in range(len(unique_names))]

    plt.figure(figsize=(4, 8))  # Adjust size for the legend
    handles = [
        plt.Line2D([0], [0], marker='o', color=color_values[idx], linestyle='', markersize=6) 
        for idx in range(len(unique_names))
    ]
    plt.legend(handles, unique_names, title="Classes", loc="center", fontsize='small', fancybox=True)
    plt.axis('off')  # Remove axes for legend-only image
    output_path_legend = "tsne_legend.png"
    plt.savefig(output_path_legend, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Main plot saved as {output_path_main}")
    print(f"Legend saved as {output_path_legend}")




  


# Define a hook function
def hook_fn(module, input, output):
    global convert_layer_output
    convert_layer_output = output

