import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn
from transformers import AutoProcessor, FlavaModel
from .losses import DiceLoss, CEDiceLoss, FocalLoss

from models.encoder import Encoder
from models.decoder import Decoder, VoxelDecoderTr
from models.refiner import Refiner
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

# class ConvertLayer(nn.Module):
#     def __init__(self):
#         super(ConvertLayer, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Halve the dimensions

#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
#         self.pool3 = nn.AdaptiveMaxPool2d((7, 7))  # Ensure final size is 7x7

#     def forward(self, x):
#         x = self.pool1(self.conv1(x))
#         x = self.pool2(self.conv2(x))
#         x = self.pool3(self.conv3(x))
#         return x.unsqueeze(1)
def load_state_dict_with_prefix(state_dict, submodule, prefix):
    filtered_state_dict = {f"module.{key[len(prefix):]}": value 
                           for key, value in state_dict.items() 
                           if key.startswith(prefix)}
    submodule.load_state_dict(filtered_state_dict)


class VL2Pix(nn.Module):
    def __init__(self, cfg,bce_loss):
        """
        Combined model with pretrained encoder and decoder, both frozen during training.

        Args:
            cfg: Configuration object containing model settings.
            encoder_checkpoint (str, optional): Path to pretrained encoder checkpoint.
            decoder_checkpoint (str, optional): Path to pretrained decoder checkpoint.
        """
        super(VL2Pix, self).__init__()
        self.cfg = cfg
        self.bce_loss = bce_loss
        self.loss_CEDice = CEDiceLoss()
        # Initialize and load pretrained encoder
        self.encoder = torch.nn.DataParallel(Encoder(cfg) ).cuda()

        ## Define the decoder
        if cfg.NETWORK.DECODER == "Conv":
            self.decoder = torch.nn.DataParallel(Decoder(cfg)).cuda()
        elif cfg.NETWORK.DECODER == "Tr":
            self.decoder = torch.nn.DataParallel(VoxelDecoderTr()).cuda()
        else:
            raise ValueError(f"Unknown decoder type: {cfg.NETWORK.DECODER}. Supported types are 'Conv' and 'Tr'.")   
                    
        self.refiner = torch.nn.DataParallel(Refiner(cfg) ).cuda()
        self.merger =  torch.nn.DataParallel(Merger(cfg) ).cuda()
        self.flava_model = FlavaModel.from_pretrained("facebook/flava-full")
        self.flava_model = torch.nn.DataParallel(self.flava_model).cuda()  
        self.convert_layer =  torch.nn.DataParallel(ConvertLayer() ).cuda()
        self.flava_processor = AutoProcessor.from_pretrained("facebook/flava-full")

        
        if cfg.CONST.STATE=="Train":
            checkpoint = torch.load(cfg.CONST.WEIGHTS)
            # self.encoder = torch.nn.DataParallel(self.encoder)
            ################################################################################## Converter
            for param in self.convert_layer.parameters():
                param.requires_grad = True  
                                     
            ################################################################################## Decoder
            for param in self.decoder.parameters():
                param.requires_grad = True  # Freeze encoder

            ################################################################################## Refiner 
            self.refiner.load_state_dict(checkpoint['refiner_state_dict'])
            if not self.cfg.Train_refiner:
                for param in self.refiner.parameters():
                    param.requires_grad = True  # Freeze encoder

            ################################################################################## Merger
            if cfg.NETWORK.USE_MERGER:
                self.merger.load_state_dict(checkpoint['merger_state_dict'])
                if not self.cfg.Train_merger:       
                    for param in self.merger.parameters():
                        param.requires_grad = False  # Freeze encoder
                        
            ################################################################################## Initialize FLAVA model and processor
            for param in self.flava_model.parameters():
                param.requires_grad = False  # Freeze FLAVA model

        elif cfg.CONST.STATE=="Train_resume":
            checkpoint = torch.load(cfg.CONST.WEIGHTS)
            # self.encoder = torch.nn.DataParallel(self.encoder)
            vlm2pix_state_dict = checkpoint["vlm2pix_state_dict"]    
            # for key in vlm2pix_state_dict.keys():
            #     print(key)     

            ################################################################################## Decoder   
            # self.decoder.load_state_dict(vlm2pix_state_dict["decoder"])
            load_state_dict_with_prefix(vlm2pix_state_dict, self.decoder, "decoder.module.")
            for param in self.decoder.parameters():
                param.requires_grad = True  

            ################################################################################## Refiner 
            load_state_dict_with_prefix(vlm2pix_state_dict, self.refiner, "refiner.module.")
            for param in self.refiner.parameters():
                param.requires_grad = True  
                
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
                param.requires_grad = True  


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
        text = self.flava_processor(text=list(taxonomy_class), return_tensors="pt", padding=True, truncation=True, max_length=40) #
        
        # Forward pass through FLAVA model
        inputs_FLAVA = {
            "pixel_values": rendering_images.squeeze(1),
            "input_ids": text.input_ids.cuda(),
            "attention_mask": text.attention_mask.cuda()
        }
        flava_outputs = self.flava_model(**inputs_FLAVA)
        # Forward pass through encoder and decoder
        if self.cfg.NETWORK.DECODER == "Conv":
            flava_embeddings = self.ensure_fixed_shape(flava_outputs.multimodal_embeddings, target_length=202)    
            flava_embeddings = flava_embeddings.unsqueeze(1) 
            # print("flava_embeddings shape-----------------------------------------", flava_embeddings.shape)

            flava_embeddings = self.convert_layer(flava_embeddings)
            # print("convert_layer shape-----------------------------------------", flava_embeddings.shape)

            raw_features, generated_volumes = self.decoder(flava_embeddings) # image_features + 
            # print("generated_volumes shape---------------------------", generated_volumes.shape)

        elif self.cfg.NETWORK.DECODER == "Tr":
            flava_embeddings = self.ensure_fixed_shape(flava_outputs.multimodal_embeddings, target_length=210)

            generated_volumes = self.decoder(flava_embeddings) 
            
        # Merge raw features if merger is enabled
        if self.cfg.NETWORK.USE_MERGER:
            generated_volumes = self.merger(raw_features, generated_volumes)
        else:
            if self.cfg.NETWORK.DECODER == "Conv":
                generated_volumes = torch.mean(generated_volumes, dim=1)
                encoder_loss = self.bce_loss(generated_volumes, ground_truth_volumes) * 10
            elif self.cfg.NETWORK.DECODER == "Tr":
                # print("generated_volumes shape---------------------------", generated_volumes.shape)
                # print("ground_truth_volumes shape------------------------", ground_truth_volumes.shape)
                encoder_loss = self.loss_CEDice(generated_volumes.squeeze(1),ground_truth_volumes )
                    
        if self.cfg.NETWORK.USE_REFINER:
            generated_volumes = self.refiner(generated_volumes)
            refiner_loss = self.bce_loss(generated_volumes, ground_truth_volumes) * 10
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


