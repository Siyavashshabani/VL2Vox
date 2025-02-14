import torch
import torch.nn as nn
from transformers import AutoProcessor, FlavaModel
from .losses import DiceLoss, CEDiceLoss, FocalLoss

from models.encoder import Encoder
from models.decoder import Decoder, VoxelDecoderTr
from models.refiner import Refiner
from models.merger import Merger
from utils.average_meter import AverageMeter
from models.fusion import Fusion, ConvFusion

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

class ConverterDino(nn.Module):
    def __init__(self):
        super(ConverterDino, self).__init__()
        self.layers = nn.Sequential(
            # First convolution: Reduce input channels and start reducing spatial dimensions
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3),  # [B, 64, 99, 99]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, 50, 50]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [B, 256, 25, 25]
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # [B, 256, 13, 13]
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # [B, 256, 7, 7]
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layers(x)  # Reduce to [B, 256, 7, 7]
        x = x.unsqueeze(1)  # Add the new dimension [B, 1, 256, 7, 7]
        return x



def load_state_dict_with_prefix(state_dict, submodule, prefix):
    filtered_state_dict = {f"module.{key[len(prefix):]}": value 
                           for key, value in state_dict.items() 
                           if key.startswith(prefix)}
    submodule.load_state_dict(filtered_state_dict)


class VL2VoxDino(nn.Module):
    def __init__(self, cfg,bce_loss):
        """
        Combined model with pretrained encoder and decoder, both frozen during training.

        Args:
            cfg: Configuration object containing model settings.
            encoder_checkpoint (str, optional): Path to pretrained encoder checkpoint.
            decoder_checkpoint (str, optional): Path to pretrained decoder checkpoint.
        """
        super(VL2VoxDino, self).__init__()
        self.cfg = cfg
        self.bce_loss = bce_loss
        self.loss_CEDice = CEDiceLoss()
        # Initialize and load pretrained encoder
        self.encoder = torch.nn.DataParallel(Encoder(cfg) ).cuda()

        ## Define the decoder
        if cfg.NETWORK.DECODER == "Conv":
            self.decoder = torch.nn.DataParallel(Decoder(cfg)).cuda()
            self.decoder_dino = torch.nn.DataParallel(Decoder(cfg)).cuda()
            
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

        ##################################### DINO modules  
        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.dino = torch.nn.DataParallel(self.dino).cuda()
        
        self.convert_dino = torch.nn.DataParallel(ConverterDino() ).cuda()
        self.fusion = torch.nn.DataParallel(ConvFusion(cfg) ).cuda() 
        
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


            ################################################################################## dino and its Converter and fusion
            for param in self.convert_dino.parameters():
                param.requires_grad = True 

            for param in self.dino.parameters():
                param.requires_grad = False  # Freeze FLAVA model

            for param in self.fusion.parameters():
                param.requires_grad = True  # Freeze FLAVA model


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
        
        # print("flava_embeddings-------------------------------------------------------------", flava_embeddings.shape)
         
        
        # print("output of converter layer----------------------------------------------------", flava_embeddings.shape)
        # exit()
        # Forward pass through encoder and decoder
        # image_features = self.encoder(rendering_images)
        if self.cfg.NETWORK.DECODER == "Conv":
            flava_embeddings = self.ensure_fixed_shape(flava_outputs.multimodal_embeddings, target_length=202)    
            flava_embeddings = flava_embeddings.unsqueeze(1) 
            flava_embeddings = self.convert_layer(flava_embeddings)
            raw_features, generated_volumes = self.decoder(flava_embeddings) 
            
            ############# dino midules 
            dino_output = self.dino.module.get_last_selfattention(rendering_images.squeeze(1))
            ## converter 
            dino_output = self.convert_dino(dino_output)
            ## dino decoder 
            raw_features_dino, generated_volumes_dino = self.decoder_dino(dino_output)
            ## fusion 
            final_generated_volumes = self.fusion(generated_volumes, generated_volumes_dino)
            # exit()

        elif self.cfg.NETWORK.DECODER == "Tr":
            flava_embeddings = self.ensure_fixed_shape(flava_outputs.multimodal_embeddings, target_length=210)
            generated_volumes = self.decoder(flava_embeddings) 
            
        # Merge raw features if merger is enabled
        if self.cfg.NETWORK.USE_MERGER:
            generated_volumes = self.merger(raw_features, generated_volumes)
        else:
            if self.cfg.NETWORK.DECODER == "Conv":
                generated_volumes      = torch.mean(generated_volumes, dim=1)
                generated_volumes_dino = torch.mean(generated_volumes_dino, dim=1)
                vl_encoder_loss = self.bce_loss(generated_volumes, ground_truth_volumes) * 10
                # print("generated_volumes_dino, ground_truth_volumes", generated_volumes_dino.shape, ground_truth_volumes.shape)
                dino_encoder_loss = self.bce_loss(generated_volumes_dino, ground_truth_volumes) * 10
                # print(final_generated_volumes.shape, ground_truth_volumes.shape)
                fusion_loss = self.bce_loss(final_generated_volumes.squeeze(1), ground_truth_volumes) * 10
                
        if self.cfg.NETWORK.USE_REFINER:
            generated_volumes_dino = self.refiner(generated_volumes_dino)
            # refiner_loss = self.bce_loss(final_generated_volumes, ground_truth_volumes) * 10
            # print("Use refiner-------------------------------------------------")
        else:
            refiner_loss = vl_encoder_loss
        # print("vl_encoder_loss, dino_encoder_loss, fusion_loss, refiner_loss")
        # exit()
        # return vl_encoder_loss, dino_encoder_loss, refiner_loss, generated_volumes_dino
        return vl_encoder_loss, dino_encoder_loss, refiner_loss,fusion_loss, final_generated_volumes

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


