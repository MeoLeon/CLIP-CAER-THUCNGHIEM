# builders.py

import argparse
from typing import Tuple
import os
import torch
import torch.utils.data
from models.clip import clip

from dataloader.video_dataloader import train_data_loader, test_data_loader
from models.Generate_Model import GenerateModel
#from models.Text import *
from utils.utils import *


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    print("Loading pretrained CLIP model...")
    CLIP_model, _ = clip.load(args.clip_path, device='cpu')
    CLIP_visual = CLIP_model.visual

    print("\nInput Text Prompts:")


    print("\nInstantiating GenerateModel...")
    #model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)
    model = GenerateModel(clip_model=CLIP_visual, args=args)

    for name, param in model.named_parameters():
        param.requires_grad = False

    trainable_params_keywords = ["image_encoder", "temporal_net", "temporal_net_body", "project_fc"]
    print('\nTrainable parameters:')
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trainable_params_keywords):
            param.requires_grad = True
            print(f"- {name}")
    print('************************\n')

    return model


def get_class_info(args: argparse.Namespace) -> Tuple[list, list]:
    """
    根据数据集和文本类型获取 class_names 和 input_text（用于生成 CLIP 模型文本输入）。

    Returns:
        class_names: 类别名称，用于混淆矩阵等
        input_text: 输入文本，用于传入模型
    """
    if args.dataset == "RAER":
        class_names = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction.']
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not implemented yet.")

    return class_names



def build_dataloaders(args: argparse.Namespace) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: 
    train_annotation_file_path = os.path.join(args.root_dir, args.train_annotation)
    test_annotation_file_path = os.path.join(args.root_dir, args.test_annotation)
    
    print("Loading train data...")
    train_data = train_data_loader(
        list_file=train_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size,dataset_name=args.dataset,
        bounding_box_face=args.bounding_box_face,bounding_box_body=args.bounding_box_body
    )
    
    print("Loading test data...")
    test_data = test_data_loader(
        list_file=test_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size,
        bounding_box_face=args.bounding_box_face,bounding_box_body=args.bounding_box_body
    )

    print("Creating DataLoader instances...")
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    return train_loader, val_loader