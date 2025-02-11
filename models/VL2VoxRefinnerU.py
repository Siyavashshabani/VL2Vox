import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn
from transformers import AutoProcessor, FlavaModel

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import UNet3D # RefinerU
from models.merger import Merger
from utils.average_meter import AverageMeter

class ConvertLayer(nn.Module):
    def __init__(self, input_dim=(202, 768), output_dim=(1, 256, 7, 7), bottleneck_dim=512):
        super(ConvertLayer, self).__init__()
        self.input_size = input_dim[0] * input_dim[1]  # Flattened input size
        self.output_size = output_dim[0] * output_dim[1] * output_dim[2] * output_dim[3]  # Flattened output size
        
        # Bottleneck to reduce parameters drastically
        self.bottleneck = nn.Linear(self.input_size, bottleneck_dim)
        
        # Linear layer to map bottleneck to the final size
        self.fc = nn.Linear(bottleneck_dim, self.output_size)
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        # print("after view-----------------", x.shape)
        # Apply bottleneck
        x = self.bottleneck(x)
        # print("after btn-----------------", x.shape)
        # Map to final size
        x = self.fc(x)
        # print("after fc-----------------", x.shape)
       
        # Reshape to [B, 1, 256, 7, 7]
        x = x.view(x.size(0), 1, 256, 7, 7)
        return x

def load_state_dict_with_prefix(state_dict, submodule, prefix):
    filtered_state_dict = {f"module.{key[len(prefix):]}": value 
                           for key, value in state_dict.items() 
                           if key.startswith(prefix)}
    submodule.load_state_dict(filtered_state_dict)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten inputs and targets
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        # Compute the intersection and union
        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # Return Dice Loss
        return 1 - dice_score

class VL2VoxRefinnerU(nn.Module):
    def __init__(self, cfg,bce_loss):
        """
        Combined model with pretrained encoder and decoder, both frozen during training.

        Args:
            cfg: Configuration object containing model settings.
            encoder_checkpoint (str, optional): Path to pretrained encoder checkpoint.
            decoder_checkpoint (str, optional): Path to pretrained decoder checkpoint.
        """
        super(VL2VoxRefinnerU, self).__init__()
        self.cfg = cfg
        self.dice_loss = DiceLoss()
        self.bce_loss = bce_loss
        # Initialize and load pretrained encoder
        # self.encoder = torch.nn.DataParallel(Encoder(cfg) ).cuda()
        self.decoder = torch.nn.DataParallel(Decoder(cfg) ).cuda()
        self.refiner = torch.nn.DataParallel(UNet3D(in_channels=1, num_classes=1) ).cuda()
        self.merger =  torch.nn.DataParallel(Merger(cfg) ).cuda()
        self.flava_model = FlavaModel.from_pretrained("facebook/flava-full")
        self.flava_model = torch.nn.DataParallel(self.flava_model).cuda()  
        self.convert_layer =  torch.nn.DataParallel(ConvertLayer() ).cuda()
        self.flava_processor = AutoProcessor.from_pretrained("facebook/flava-full")

        # self.decoder = Decoder(cfg).cuda()
        # self.refiner = RefinerU(cfg).cuda()
        # self.merger =  Merger(cfg).cuda()
        # self.flava_model = FlavaModel.from_pretrained("facebook/flava-full")
        # self.flava_model = self.flava_model.cuda()  
        # self.convert_layer =  ConvertLayer().cuda()
        print("encoder_checkpoint---------------------------------------------------")
        # if checkpoint:
        
        if cfg.CONST.STATE=="Train":
            checkpoint = torch.load(cfg.CONST.WEIGHTS)
            # self.encoder = torch.nn.DataParallel(self.encoder)
            vlm2pix_state_dict = checkpoint["vlm2pix_state_dict"]    
            # for key in vlm2pix_state_dict.keys():
            #     print(key)     

            ################################################################################## Decoder   
            # self.decoder.load_state_dict(vlm2pix_state_dict["decoder"])
            load_state_dict_with_prefix(vlm2pix_state_dict, self.decoder, "decoder.module.")
            for param in self.decoder.parameters():
                param.requires_grad = False  

            ################################################################################## Refiner 
            # load_state_dict_with_prefix(vlm2pix_state_dict, self.refiner, "refiner.module.")
            # for param in self.refiner.parameters():
            #     param.requires_grad = True  
                
            ################################################################################## Merger
            if cfg.NETWORK.USE_MERGER:
                load_state_dict_with_prefix(vlm2pix_state_dict, self.merger, "merger.module.")
                for param in self.merger.parameters():
                    param.requires_grad = False  
                        
            ################################################################################## Initialize FLAVA model and processor
            load_state_dict_with_prefix(vlm2pix_state_dict, self.flava_model, "flava_model.module.")
            for param in self.flava_model.parameters():
                param.requires_grad = False  # Freeze FLAVA model

            ################################################################################## converter 
            load_state_dict_with_prefix(vlm2pix_state_dict, self.convert_layer, "convert_layer.module.")
            for param in self.convert_layer.parameters():
                param.requires_grad = False  
                
    def forward(self, rendering_images,ground_truth_volumes, taxonomy_class):
        """
        Forward pass for the CombinedModel.

        Args:
            rendering_images (torch.Tensor): Batch of input images.
            taxonomy_class (list[str]): List of text labels corresponding to the batch.

        Returns:
            torch.Tensor: The refined/generated volumes from the model.
        """
        # Process text inputs using FLAVA processor
        # print("taxonomy_class----------------------------------", taxonomy_class)
        # exit()        
        text = self.flava_processor(text=list(taxonomy_class), return_tensors="pt", padding=True, truncation=True, max_length=40) #
        # text = {key: val.cuda() for key, val in text.items()}  # Move to GPU if available
        
        # Forward pass through FLAVA model
        inputs_FLAVA = {
            "pixel_values": rendering_images.squeeze(1),
            "input_ids": text.input_ids.cuda(),
            "attention_mask": text.attention_mask.cuda()
        }
        flava_outputs = self.flava_model(**inputs_FLAVA)
        # print("flava_outputs------------------", flava_outputs.multimodal_embeddings.shape)
        # print("flava_outputs.multimodal_embeddings------------------------------------------", flava_outputs.multimodal_embeddings.shape)
        flava_embeddings = self.ensure_fixed_shape(flava_outputs.multimodal_embeddings, target_length=202)
        # print("flava_embeddings-------------------------------------------------------------", flava_embeddings.shape)
        flava_embeddings = flava_embeddings.unsqueeze(1)  
        flava_embeddings = self.convert_layer(flava_embeddings)
        # print("output of converter layer----------------------------------------------------", flava_embeddings.shape)
        # exit()
        # Forward pass through encoder and decoder
        # image_features = self.encoder(rendering_images)
        raw_features, generated_volumes = self.decoder(flava_embeddings) # image_features + 
        # print("raw_features-----------------------------------:", raw_features.shape)
        # print("generated_volumes------------------------------:", generated_volumes.shape)
        # Merge raw features if merger is enabled
        if self.cfg.NETWORK.USE_MERGER:
            generated_volumes = self.merger(raw_features, generated_volumes)
        else:
            generated_volumes = torch.mean(generated_volumes, dim=1)
        encoder_loss = self.bce_loss(generated_volumes, ground_truth_volumes) * 10

        if self.cfg.NETWORK.USE_REFINER:
            # print("generated volume before refiner---------------", generated_volumes.min().item(), generated_volumes.max().item())
            generated_volumes = self.refiner(generated_volumes)
            # print("generated_volumes shape-----------------------", generated_volumes.shape)
            # print("GT shape--------------------------------------", ground_truth_volumes.shape)
            # print("Generated Volumes (min, max):", generated_volumes.min().item(), generated_volumes.max().item())
            # print("Ground Truth Volumes (min, max):", ground_truth_volumes.min().item(), ground_truth_volumes.max().item())

            # refiner_loss = self.bce_loss(generated_volumes, ground_truth_volumes)* 10
            refiner_loss = self.dice_loss(generated_volumes, ground_truth_volumes)*10
        else:
            refiner_loss = encoder_loss

        return encoder_loss, refiner_loss, generated_volumes

    def ensure_fixed_shape(self, tensor, target_length=202):
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
        # print("tensor-------------------------------------", tensor.shape)
        return tensor


