# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import json
import numpy as np
import logging
import os
import random
import scipy.io
import scipy.ndimage
import sys
import torch.utils.data.dataset

from enum import Enum, unique

import utils.binvox_rw


@unique
class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


# //////////////////////////////// = End of DatasetType Class Definition = ///////////////////////////////// #


class ShapeNetDataset(torch.utils.data.dataset.Dataset):
    """ShapeNetDataset class used for PyTorch DataLoader"""
    def __init__(self, dataset_type, file_list, n_views_rendering, transforms=None):
        self.dataset_type = dataset_type
        self.file_list = file_list
        self.transforms = transforms
        self.n_views_rendering = n_views_rendering
        self.first_sentene = True
        self.category_mapping = {
            "02691156": "aeroplane",
            "02828884": "bench",
            "02933112": "cabinet",
            "02958343": "car",
            "03001627": "chair",
            "03211117": "display",
            "03636649": "lamp",
            "03691459": "speaker",
            "04090263": "rifle",
            "04256520": "sofa",
            "04379243": "table",
            "04401088": "telephone",
            "04530566": "watercraft",
        }
    def get_category_name(self, category_id):
        return self.category_mapping.get(category_id, "Unknown")        
        # Define the mapping
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # print("idx-------------------------------", idx)
        taxomony_class, taxonomy_name, sample_name, rendering_images, volume = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images)

        return taxomony_class, taxonomy_name, sample_name, rendering_images, volume

    def set_n_views_rendering(self, n_views_rendering):
        self.n_views_rendering = n_views_rendering

    def extract_first_sentence(self, text):
        # Iterate over characters and identify the first complete sentence
        sentence = []
        for char in text:
            if char == '.':  # Sentence terminator found
                return ''.join(sentence).strip()
            sentence.append(char)
        return ''.join(sentence).strip() 

    def get_datum(self, idx):
        # print("self.file_list[idx]-------------------", self.file_list[idx])
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_paths = self.file_list[idx]['rendering_images']
        volume_path = self.file_list[idx]['volume']
        text_path = self.file_list[idx]['text']

        # Get data of rendering images
        if self.dataset_type == DatasetType.TRAIN:
            selected_rendering_image_paths = [
                rendering_image_paths[i]
                for i in random.sample(range(len(rendering_image_paths)), self.n_views_rendering)
            ]
        else:
            selected_rendering_image_paths = [rendering_image_paths[i] for i in range(self.n_views_rendering)]

        rendering_images = []
        for image_path in selected_rendering_image_paths:
            rendering_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            if len(rendering_image.shape) < 3:
                logging.error('It seems that there is something wrong with the image file %s' % (image_path))
                sys.exit(2)

            rendering_images.append(rendering_image)

        # Get data of volume
        _, suffix = os.path.splitext(volume_path)

        if suffix == '.mat':
            volume = scipy.io.loadmat(volume_path)
            volume = volume['Volume'].astype(np.float32)
        elif suffix == '.binvox':
            with open(volume_path, 'rb') as f:
                volume = utils.binvox_rw.read_as_3d_array(f)
                volume = volume.data.astype(np.float32)

        ### load the .txt file 
        try:
            with open(text_path, "r") as file:
                text_content = file.read()  # Read the content of the file
                if self.first_sentene is not None:
                    text_content = self.extract_first_sentence(text_content) 
            # print(text_content)  # Print the content
        except FileNotFoundError:
            print(f"File not found: {text_path}")    
            # exit()  
        # print("taxonomy_name-------------------", taxonomy_name)
        # taxomony_class = self.get_category_name(taxonomy_name)
        # print("taxomony_class-------------------", taxomony_class)
        # print("volume_path-------------------", volume_path)
        # exit()
        # print("sample_name-------------------", sample_name)
        # print("taxomony_class-------------------", taxomony_class)
        # print("self.volume_path_template-------------------", self.volume_path_template)
        # if len(taxomony_class)!=1:
        #     taxomony_class = taxomony_class[0]
        # print("taxomony_class------------------------------", taxomony_class)
        return text_content, taxonomy_name, sample_name, np.asarray(rendering_images), volume


# //////////////////////////////// = End of ShapeNetDataset Class Definition = ///////////////////////////////// #


class ShapeNetDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        self.rendering_image_path_template = cfg.DATASETS.SHAPENET.RENDERING_PATH
        self.volume_path_template = cfg.DATASETS.SHAPENET.VOXEL_PATH
        self.text_data = pd.read_csv(cfg.DATASETS.SHAPENET.TEXT_PATH)
        self.text_data = self.text_data[self.text_data['source_dataset'] == "ShapeNet"]

        # Load all taxonomies of the dataset
        with open(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

        if cfg.DATASETS.SHAPETALK.SHAPE_TO_REMOVE is not None:
            # Remove the specified shapes
            self.dataset_taxonomy = [
                entry for entry in self.dataset_taxonomy if entry["taxonomy_name"] not in cfg.DATASETS.SHAPETALK.SHAPE_TO_REMOVE
            ]           
            remaining_taxonomy_names = [entry["taxonomy_name"] for entry in self.dataset_taxonomy]
            print("Remaining taxonomy names----------------------------:", remaining_taxonomy_names)
            

            
    def get_dataset(self, dataset_type, n_views_rendering, transforms=None):
        files = []

        # Load data for each category
        for taxonomy in self.dataset_taxonomy:
            taxonomy_folder_name = taxonomy['taxonomy_id']
            logging.info('Collecting files of Taxonomy[ID=%s, Name=%s]' %
                         (taxonomy['taxonomy_id'], taxonomy['taxonomy_name']))
            samples = []
            if dataset_type == DatasetType.TRAIN:
                samples = taxonomy['train']
            elif dataset_type == DatasetType.TEST:
                samples = taxonomy['test']
            elif dataset_type == DatasetType.VAL:
                samples = taxonomy['val']

            files.extend(self.get_files_of_taxonomy(taxonomy_folder_name, samples))

        logging.info('Complete collecting files of the dataset. Total files: %d.' % (len(files)))
        return ShapeNetDataset(dataset_type, files, n_views_rendering, transforms)

    def get_files_of_taxonomy(self, taxonomy_folder_name, samples):
        files_of_taxonomy = []


        for sample_idx, sample_name in enumerate(samples):
            # Get file path of volumes

            volume_file_path = self.volume_path_template % (taxonomy_folder_name, sample_name)

            # print("volume_file_path-------------------", volume_file_path)
            if not os.path.exists(volume_file_path):
                logging.warn('Ignore sample %s/%s since volume file not exists.' % (taxonomy_folder_name, sample_name))
                continue

            # Get file list of rendering images
            img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, 0)
            img_folder = os.path.dirname(img_file_path)
            total_views = len(os.listdir(img_folder))
            rendering_image_indexes = range(total_views)
            rendering_images_file_path = []
            for image_idx in rendering_image_indexes:
                img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, image_idx)
                if not os.path.exists(img_file_path):
                    continue

                rendering_images_file_path.append(img_file_path)

            if len(rendering_images_file_path) == 0:
                logging.warn('Ignore sample %s/%s since image files not exists.' % (taxonomy_folder_name, sample_name))
                continue


            ## create the .txt file 
            text_file_path = os.path.join(os.path.dirname(volume_file_path), "text.txt")

            # print("taxomony_class-------------------:", taxomony_class)
            # exit()
            ## extract the corresponding text
            # print("sample_name-----------------------------:", sample_name)
            # print("taxonomy_folder_name--------------------:", taxonomy_folder_name)
            
            
            # filtered_data = self.text_data[
            #     self.text_data['source_model_name'].str.contains(sample_name, na=False) ]
            # # print("len of filtered_data-----------------------------:", filtered_data["utterance"])
            # if filtered_data.empty or 'utterance' not in filtered_data.columns:
            #     continue
            
            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_folder_name,
                'sample_name': sample_name,
                'rendering_images': rendering_images_file_path,
                'volume': volume_file_path,
                'text': text_file_path,
            })
        #     print("sample_name------------------------:", sample_name)
            # print("volume_file_path----------------------:", volume_file_path)
        # exit()
        return files_of_taxonomy


# /////////////////////////////// = End of ShapeNetDataLoader Class Definition = /////////////////////////////// #


class Pascal3dDataset(torch.utils.data.dataset.Dataset):
    """Pascal3D class used for PyTorch DataLoader"""
    def __init__(self, file_list, transforms=None):
        self.file_list = file_list
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, volume, bounding_box = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images, bounding_box)

        return taxonomy_name, sample_name, rendering_images, volume

    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_path = self.file_list[idx]['rendering_image']
        bounding_box = self.file_list[idx]['bounding_box']
        volume_path = self.file_list[idx]['volume']

        # Get data of rendering images
        rendering_image = cv2.imread(rendering_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if len(rendering_image.shape) < 3:
            logging.warn('[WARN] %s It seems the image file %s is grayscale.' % (rendering_image_path))
            rendering_image = np.stack((rendering_image, ) * 3, -1)

        # Get data of volume
        with open(volume_path, 'rb') as f:
            volume = utils.binvox_rw.read_as_3d_array(f)
            volume = volume.data.astype(np.float32)

        return taxonomy_name, sample_name, np.asarray([rendering_image]), volume, bounding_box


# //////////////////////////////// = End of Pascal3dDataset Class Definition = ///////////////////////////////// #


class Pascal3dDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        self.volume_path_template = cfg.DATASETS.PASCAL3D.VOXEL_PATH
        self.annotation_path_template = cfg.DATASETS.PASCAL3D.ANNOTATION_PATH
        self.rendering_image_path_template = cfg.DATASETS.PASCAL3D.RENDERING_PATH

        # Load all taxonomies of the dataset
        with open(cfg.DATASETS.PASCAL3D.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

    def get_dataset(self, dataset_type, n_views_rendering, transforms=None):
        files = []

        # Load data for each category
        for taxonomy in self.dataset_taxonomy:
            taxonomy_name = taxonomy['taxonomy_name']
            logging.info('Collecting files of Taxonomy[Name=%s]' % (taxonomy_name))

            samples = []
            if dataset_type == DatasetType.TRAIN:
                samples = taxonomy['train']
            elif dataset_type == DatasetType.TEST:
                samples = taxonomy['test']
            elif dataset_type == DatasetType.VAL:
                samples = taxonomy['test']

            files.extend(self.get_files_of_taxonomy(taxonomy_name, samples))

        logging.info('Complete collecting files of the dataset. Total files: %d.' % (len(files)))
        return Pascal3dDataset(files, transforms)

    def get_files_of_taxonomy(self, taxonomy_name, samples):
        files_of_taxonomy = []

        for sample_idx, sample_name in enumerate(samples):
            # Get file list of rendering images
            rendering_image_file_path = self.rendering_image_path_template % (taxonomy_name, sample_name)
            # if not os.path.exists(rendering_image_file_path):
            #     continue

            # Get image annotations
            annotations_file_path = self.annotation_path_template % (taxonomy_name, sample_name)
            annotations_mat = scipy.io.loadmat(annotations_file_path, squeeze_me=True, struct_as_record=False)
            img_width, img_height, _ = annotations_mat['record'].imgsize
            annotations = annotations_mat['record'].objects

            cad_index = -1
            bbox = None
            if (type(annotations) == np.ndarray):
                max_bbox_aera = -1

                for i in range(len(annotations)):
                    _cad_index = annotations[i].cad_index
                    _bbox = annotations[i].__dict__['bbox']

                    bbox_xmin = _bbox[0]
                    bbox_ymin = _bbox[1]
                    bbox_xmax = _bbox[2]
                    bbox_ymax = _bbox[3]
                    _bbox_area = (bbox_xmax - bbox_xmin) * (bbox_ymax - bbox_ymin)

                    if _bbox_area > max_bbox_aera:
                        bbox = _bbox
                        cad_index = _cad_index
                        max_bbox_aera = _bbox_area
            else:
                cad_index = annotations.cad_index
                bbox = annotations.bbox

            # Convert the coordinates of bounding boxes to percentages
            bbox = [bbox[0] / img_width, bbox[1] / img_height, bbox[2] / img_width, bbox[3] / img_height]
            # Get file path of volumes
            volume_file_path = self.volume_path_template % (taxonomy_name, cad_index)
            if not os.path.exists(volume_file_path):
                logging.warn('Ignore sample %s/%s since volume file not exists.' % (taxonomy_name, sample_name))
                continue

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_name,
                'sample_name': sample_name,
                'rendering_image': rendering_image_file_path,
                'bounding_box': bbox,
                'volume': volume_file_path,
            })

        return files_of_taxonomy


# /////////////////////////////// = End of Pascal3dDataLoader Class Definition = /////////////////////////////// #


class Pix3dDataset(torch.utils.data.dataset.Dataset):
    """Pix3D class used for PyTorch DataLoader"""
    def __init__(self, file_list, transforms=None):
        self.file_list = file_list
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, volume, bounding_box = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images, bounding_box)

        return taxonomy_name, sample_name, rendering_images, volume

    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_path = self.file_list[idx]['rendering_image']
        bounding_box = self.file_list[idx]['bounding_box']
        volume_path = self.file_list[idx]['volume']

        # Get data of rendering images
        rendering_image = cv2.imread(rendering_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if len(rendering_image.shape) < 3:
            logging.warn('It seems the image file %s is grayscale.' % (rendering_image_path))
            rendering_image = np.stack((rendering_image, ) * 3, -1)

        # Get data of volume
        with open(volume_path, 'rb') as f:
            volume = utils.binvox_rw.read_as_3d_array(f)
            volume = volume.data.astype(np.float32)

        return taxonomy_name, sample_name, np.asarray([rendering_image]), volume, bounding_box


# //////////////////////////////// = End of Pascal3dDataset Class Definition = ///////////////////////////////// #


class Pix3dDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        self.annotations = dict()
        self.volume_path_template = cfg.DATASETS.PIX3D.VOXEL_PATH
        self.rendering_image_path_template = cfg.DATASETS.PIX3D.RENDERING_PATH

        # Load all taxonomies of the dataset
        with open(cfg.DATASETS.PIX3D.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

        # Load all annotations of the dataset
        _annotations = None
        with open(cfg.DATASETS.PIX3D.ANNOTATION_PATH, encoding='utf-8') as file:
            _annotations = json.loads(file.read())

        for anno in _annotations:
            filename, _ = os.path.splitext(anno['img'])
            anno_key = filename[4:]
            self.annotations[anno_key] = anno

    def get_dataset(self, dataset_type, n_views_rendering, transforms=None):
        files = []

        # Load data for each category
        for taxonomy in self.dataset_taxonomy:
            taxonomy_name = taxonomy['taxonomy_name']
            logging.info('Collecting files of Taxonomy[Name=%s]' % (taxonomy_name))

            samples = []
            if dataset_type == DatasetType.TRAIN:
                samples = taxonomy['train']
            elif dataset_type == DatasetType.TEST:
                samples = taxonomy['test']
            elif dataset_type == DatasetType.VAL:
                samples = taxonomy['test']

            files.extend(self.get_files_of_taxonomy(taxonomy_name, samples))

        logging.info('Complete collecting files of the dataset. Total files: %d.' % (len(files)))
        return Pix3dDataset(files, transforms)

    def get_files_of_taxonomy(self, taxonomy_name, samples):
        files_of_taxonomy = []

        for sample_idx, sample_name in enumerate(samples):
            # Get image annotations
            anno_key = '%s/%s' % (taxonomy_name, sample_name)
            annotations = self.annotations[anno_key]

            # Get file list of rendering images
            _, img_file_suffix = os.path.splitext(annotations['img'])
            rendering_image_file_path = self.rendering_image_path_template % (taxonomy_name, sample_name,
                                                                              img_file_suffix[1:])

            # Get the bounding box of the image
            img_width, img_height = annotations['img_size']
            bbox = [
                annotations['bbox'][0] / img_width,
                annotations['bbox'][1] / img_height,
                annotations['bbox'][2] / img_width,
                annotations['bbox'][3] / img_height
            ]  # yapf: disable
            model_name_parts = annotations['voxel'].split('/')
            model_name = model_name_parts[2]
            volume_file_name = model_name_parts[3][:-4].replace('voxel', 'model')

            # Get file path of volumes
            volume_file_path = self.volume_path_template % (taxonomy_name, model_name, volume_file_name)
            if not os.path.exists(volume_file_path):
                logging.warn('Ignore sample %s/%s since volume file not exists.' % (taxonomy_name, sample_name))
                continue

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_name,
                'sample_name': sample_name,
                'rendering_image': rendering_image_file_path,
                'bounding_box': bbox,
                'volume': volume_file_path,
            })

        return files_of_taxonomy


# /////////////////////////////// = End of Pascal3dDataLoader Class Definition = /////////////////////////////// #


class Things3DDataset(torch.utils.data.dataset.Dataset):
    """ShapeNetDataset class used for PyTorch DataLoader"""
    def __init__(self, dataset_type, file_list, n_views_rendering, transforms=None):
        self.dataset_type = dataset_type
        self.file_list = file_list
        self.transforms = transforms
        self.n_views_rendering = n_views_rendering

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, volume = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images)

        return taxonomy_name, sample_name, rendering_images, volume

    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        model_id = self.file_list[idx]['model_id']
        scene_id = self.file_list[idx]['scene_id']
        rendering_image_paths = self.file_list[idx]['rendering_images']
        volume_path = self.file_list[idx]['volume']

        # Get data of rendering images
        if self.dataset_type == DatasetType.TRAIN:
            selected_rendering_image_paths = [
                rendering_image_paths[i]
                for i in random.sample(range(len(rendering_image_paths)), self.n_views_rendering)
            ]
        else:
            selected_rendering_image_paths = [rendering_image_paths[i] for i in range(self.n_views_rendering)]

        rendering_images = []
        for image_path in selected_rendering_image_paths:
            rendering_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            if len(rendering_image.shape) < 3:
                logging.error('It seems that there is something wrong with the image file %s' % (image_path))
                sys.exit(2)

            rendering_images.append(rendering_image)

        # Get data of volume
        _, suffix = os.path.splitext(volume_path)

        if suffix == '.mat':
            volume = scipy.io.loadmat(volume_path)
            volume = volume['Volume'].astype(np.float32)
        elif suffix == '.binvox':
            with open(volume_path, 'rb') as f:
                volume = utils.binvox_rw.read_as_3d_array(f)
                volume = volume.data.astype(np.float32)

        _model_id = '%s-%s' % (model_id, scene_id)
        return taxonomy_name, _model_id, np.asarray(rendering_images), volume


# //////////////////////////////// = End of Things3DDataset Class Definition = ///////////////////////////////// #


class Things3DDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        self.rendering_image_path_template = cfg.DATASETS.THINGS3D.RENDERING_PATH
        self.volume_path_template = cfg.DATASETS.THINGS3D.VOXEL_PATH
        self.n_views_rendering = cfg.CONST.N_VIEWS_RENDERING

        # Load all taxonomies of the dataset
        with open(cfg.DATASETS.THINGS3D.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

    def get_dataset(self, dataset_type, n_views_rendering, transforms=None):
        files = []

        # Load data for each category
        for taxonomy in self.dataset_taxonomy:
            taxonomy_folder_name = taxonomy['taxonomy_id']
            logging.info('Collecting files of Taxonomy[ID=%s, Name=%s]' %
                         (taxonomy['taxonomy_id'], taxonomy['taxonomy_name']))
            models = []
            if dataset_type == DatasetType.TRAIN:
                models = taxonomy['train']
            elif dataset_type == DatasetType.TEST:
                models = taxonomy['test']
            elif dataset_type == DatasetType.VAL:
                models = taxonomy['val']

            files.extend(self.get_files_of_taxonomy(taxonomy_folder_name, models))

        logging.info('Complete collecting files of the dataset. Total files: %d.' % (len(files)))
        return Things3DDataset(dataset_type, files, n_views_rendering, transforms)

    def get_files_of_taxonomy(self, taxonomy_folder_name, models):
        files_of_taxonomy = []

        for model in models:
            model_id = model['model_id']
            scenes = model['scenes']

            # Get file path of volumes
            volume_file_path = self.volume_path_template % (taxonomy_folder_name, model_id)
            if not os.path.exists(volume_file_path):
                logging.warn('Ignore sample %s/%s since volume file not exists.' % (taxonomy_folder_name, model_id))
                continue

            # Get file list of rendering images
            for scene in scenes:
                scene_id = scene['scene_id']
                total_views = scene['n_renderings']

                if total_views < self.n_views_rendering:
                    continue

                rendering_image_indexes = range(total_views)
                rendering_images_file_path = []
                for image_idx in rendering_image_indexes:
                    img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, model_id, scene_id,
                                                                          image_idx)
                    rendering_images_file_path.append(img_file_path)

                # Append to the list of rendering images
                files_of_taxonomy.append({
                    'taxonomy_name': taxonomy_folder_name,
                    'model_id': model_id,
                    'scene_id': scene_id,
                    'rendering_images': rendering_images_file_path,
                    'volume': volume_file_path,
                })

        return files_of_taxonomy



# /////////////////////////////// = End of Things3DDataLoader Class Definition = /////////////////////////////// #
import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from torchvision import transforms
from pprint import pprint
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from transformers import AutoProcessor, FlavaModel


class ShapeTalkDataLoader(data.Dataset):
    def __init__(self, config):
        # pprint(config)
        self.data_root = config.DATASETS.SHAPETALK.DATA_ROOT
        self.pc_path = config.DATASETS.SHAPETALK.PC_PATH
        self.subset = config.DATASETS.SHAPETALK.subset
        self.npoints = config.DATASETS.SHAPETALK.N_POINTS
        self.img_path = config.DATASETS.SHAPETALK.IMG_PATH
        self.txt_apth = config.DATASETS.SHAPETALK.TXT_PATH
        # print("self.subset--------------------:", self.subset)
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        # print("self.data_list_file[0]---------",self.data_list_file )

        self.sample_points_num = self.npoints
        self.whole = config.get('whole')

        # FLAVA model initialization
        self.flava_processor = AutoProcessor.from_pretrained("facebook/flava-full")
        # self.flava_model = FlavaModel.from_pretrained("facebook/flava-full")
        # # Freeze FLAVA parameters
        # for param in self.flava_model.parameters():
        #     param.requires_grad = False
        self.text_data = pd.read_csv(self.txt_apth)
        self.text_data['combined_sentence'] = self.text_data[['source_object_class','utterance_0', 'utterance_1', 'utterance_2', 'utterance_3']].fillna('').agg(' '.join, axis=1)
        self.text_data['word_count'] = self.text_data['combined_sentence'].apply(lambda x: len(x.split()))


        # print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'ShapeTalk')
        # print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'ShapeTalk')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            # print_log(f'[DATASET] Open file {test_data_list_file}', logger = 'ShapeTalk')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'ShapeTalk')

        self.permutation = np.arange(self.npoints)
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        
    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Converts to [0, 1] and changes shape to [4, H, W]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing RGBA channels
    ])

    def text_input(self, sample):
        obj_cat = sample['model_id'].strip('/').split('/')[7]
        # print("sample['model_id']----------------------", sample['model_id'])
        # print("obj_cat---------------------------------", obj_cat)

        ## combine

        text_value = self.text_data.loc[self.text_data["source_model_name"] == obj_cat, "combined_sentence"].values #     source_object_class
        # print("text_value------------------------------------------:",text_value )
        # print("text_value[0]------------------------------------------:",text_value[0] )
        if text_value.size == 0:
            text_value = "empty"
            return text_value[0]
        else:
            # print("type(text_value[0]) -----------", type(text_value[0]) )
            # print("text_value[0]------------------------------------------:",text_value[0] )
            return text_value[0]

    def load_and_process_image(self, sample):
        """
        Constructs the target image path, loads the image, extracts RGB channels,
        and applies a transformation if specified.

        Parameters:
            base_path (str): The base path where images are stored.
            desired_section (str): The subdirectory and filename (without extension) needed for the path.
            transform (callable, optional): A transformation function to apply to the image.

        Returns:
            Image: The processed image with RGB channels.
        """
        desired_section = os.path.join(*sample['file_path'].split('/')[-3:-1], os.path.splitext(os.path.basename(sample['file_path']))[0])

        # Construct the full path to the target image
        target_img_path = os.path.join(self.img_path, f"{desired_section}.png")
        # print("target_img_path--------------------:", target_img_path)

        # Open the image and convert to RGBA, then extract RGB channels
        image = Image.open(target_img_path).convert("RGBA")
        r, g, b, _ = image.split()  # Discard the alpha channel
        image = Image.merge("RGB", (r, g, b))

        # Apply transformation if provided
        image = self.transform(image)
        image = torch.clamp(image, 0, 1)
        # print("image--------------------:", image.shape)
        return image

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        # print("Start the ShapeTalk __getitem__-------------------------------------------------------------------")
        
        ## read the clould points         
        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)

        ## read the images 
        image =self.load_and_process_image(sample)


        ## read the text file
        text = self.text_input(sample)
        # print("text_value------------------------:", text)

        # print("shape before rand sample-------------------------------", data.shape)
        data = self.random_sample(data, self.sample_points_num)
        # print("shape after rand sample-------------------------------", data.shape)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        # print("sample['taxonomy_id'], sample['model_id']", sample['taxonomy_id'], sample['model_id'], data.shape)
        # text = self.flava_processor(text=text, return_tensors="pt", padding=True, truncation=True)
        # print("text after flava_processor------------------", text)
        # print("")
                # Append to the list of rendering images
        # files_of_taxonomy.append({
        #     'taxonomy_name': sample['taxonomy_id'],
        #     'model_id': sample['model_id'],
        #     'scene_id': scene_id,
        #     'rendering_images': image,
        #     'volume': data,
        #     'text': text,
        # })
        # return sample['taxonomy_id'], sample['model_id'], data , image, text
        
        return sample['taxonomy_id'], sample['model_id'], data , image, text
    
    def __len__(self):
        return len(self.file_list)



# /////////////////////////////// = End of Things3DDataLoader Class Definition = /////////////////////////////// #

DATASET_LOADER_MAPPING = {
    'ShapeNet': ShapeNetDataLoader,
    'Pascal3D': Pascal3dDataLoader,
    'Pix3D': Pix3dDataLoader,
    'Things3D': Things3DDataLoader,
    'ShapeTalk': ShapeTalkDataLoader,
}  # yapf: disable
