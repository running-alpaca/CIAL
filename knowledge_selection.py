import os
import json
import argparse
import datetime

import torch
import numpy as np

from models import get_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import disable_caching
from PIL import Image
disable_caching()


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_name", type=str, default="LLaVA-1.5-7b")
    parser.add_argument("--type", type=str, default="Activity,Forecast,Localization/location,Localization/spatial,Object/affordance,Object/attribute,Object/existence,Planning/assistance,Planning/navigation,Reasoning/comparing,Reasoning/counting,Reasoning/situated")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()
    return args


def remove_clip0(captions):
    caption=captions[2]
    return caption


def has_clip0(captions):
    caption=captions[1]
    return caption


args = parse_args()
types = list(args.type.split(',')) 
for type in types: 
    caption_path = "./caption/"+type+"/image_captions/caption.jsonl"
    clip_path_main = "./clip_ego/"+type
    model_answers = []
    with open(caption_path, 'r') as caption_file:
        for line in caption_file:
            json_line = json.loads(line)
            img_id=json_line["img_id"]
            captions=[json_line["caption"],json_line["clip0"],json_line["clip1"],json_line["clip2"]]
            clip_json_path=clip_path_main+"/"+str(img_id)+"/clip_results.json"
            image_path = "./clip_ego/"+type+"/"+str(img_id)+"/clip0.jpg"  
            with Image.open(image_path) as img:
                width, height = img.size
                area = width * height
            if area<32000:
                caption_prompt=remove_clip0(captions)
                caption_num=3           
            elif os.path.exists(clip_json_path):
                with open(clip_json_path, "r") as clip_file:
                    clip = json.load(clip_file)
                    annotation = clip["annotations"]
                    if len(annotation)==1:
                        caption_prompt=remove_clip0(captions)
                        caption_num=3                     
                    else:
                        caption_prompt=has_clip0(captions)
                        caption_num=4       
            else:
                caption_prompt=has_clip0(captions)
                caption_num=4                        
            
            model_answers.append({
                'img_id': img_id,
                'caption_num':caption_num,
                'caption': caption_prompt,
            })        
            result_folder = "./selected_knowledge/"+type
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)

            model_answer_folder = os.path.join(result_folder)
            with open(os.path.join(model_answer_folder, f"caption.jsonl"), 'w') as f:
                for pred in model_answers:
                    f.write(json.dumps(pred) + '\n')