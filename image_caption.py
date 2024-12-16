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
disable_caching()

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_name", type=str, default="BLIP2")
    parser.add_argument("--type", type=str, default="Activity")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()
    return args



def load_dataset(args):
    annotation_path = "./data/"+args.type+"/annotations.json"
    with open(annotation_path, 'r') as f:
        dataset = json.load(f)
    for i, d in enumerate(dataset):
        image_filename = d['image_path'][0].split('/')[-1]
        dataset[i]['images'] = os.path.join(os.path.dirname(annotation_path), 'images', image_filename)
        img_id = os.path.splitext(image_filename)[0]
        data_dir="./clip_ego/"+args.type+"/"+img_id+"/"
        dataset[i]['images_with_annotation'] = os.path.join(os.path.dirname(data_dir), image_filename)
        dataset[i]['clip0'] = os.path.join(os.path.dirname(data_dir), "clip0.jpg")
        dataset[i]['clip1'] = os.path.join(os.path.dirname(data_dir), "clip1.jpg")
        dataset[i]['clip2'] = os.path.join(os.path.dirname(data_dir), "clip2.jpg")


    return dataset

def get_generation_args(dataset_name):
    if "Planning" in dataset_name or "planning" in dataset_name:
        return {
            'planning': True
        }
    else:
        return {
            'planning': False
        }

def remove_after_last_period(sentence):
    last_period_pos = sentence.rfind('.')
    if last_period_pos != -1:
        return sentence[:last_period_pos + 1]
    else:
        return sentence


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model_names = args.model_name.split(',')  
    time = datetime.datetime.now().strftime("%m%d-%H%M")
    annotation_path="./data/"+args.type+"/annotations.json"
    dataset_name = args.type

    for model_name in model_names:
        print(f"Running inference: {model_name}")

        if 'blip2' in model_name.lower() or 'llava' in model_name.lower():
            batch_size = 1
        else:
            batch_size = args.batch_size
        model = get_model(model_name, device=torch.device('cuda'))
        
        dataset = load_dataset(args)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})

        model_answers = []
        ref_answers = []
        question_files = []
        q_id = 1
        for batch in tqdm(dataloader, desc=f"Running inference: {model_name} on  {dataset_name}"):
            questions = batch['question']
            print(questions)
            images = batch['images']
            clip0 = batch['clip0']
            clip1 = batch['clip1']
            clip2 = batch['clip2']

            print("1:",batch['images'])
            print("2:",batch['clip0'])
            print("3:",batch['clip1'])
            print("4:",batch['clip2'])


            if args.batch_size == 1:
                output = model.generate(images[0], questions[0], max_new_tokens=100 ,**get_generation_args(dataset_name))
                output=remove_after_last_period(output)
                outputs = [output]
                print("output:",output)                
                clip0_output = model.generate(clip0[0], questions[0], max_new_tokens=60 ,**get_generation_args(dataset_name))            
                clip0_output=remove_after_last_period(clip0_output)
                print("clip0_output:",clip0_output)                  
                clip1_output = model.generate(clip1[0], questions[0], max_new_tokens=60 ,**get_generation_args(dataset_name))
                clip1_output=remove_after_last_period(clip1_output)
                print("clip1_output:",clip1_output)                
                clip2_output = model.generate(clip2[0], questions[0], max_new_tokens=60 ,**get_generation_args(dataset_name))
                clip2_output=remove_after_last_period(clip2_output)
                print("clip2_output:",clip2_output)
            else:
                print("batch generate")
                outputs = model.batch_generate(images, questions, **get_generation_args(dataset_name))
            for i, (question, answer, pred) in enumerate(zip(batch['question'], batch['answer'], outputs)):

                model_answers.append({
                    'img_id': q_id,
                    'model_id': model_name,
                    'caption': pred,
                    'clip0': clip0_output,
                    'clip1': clip1_output,
                    'clip2': clip2_output,
                })
                q_id += 1


        torch.cuda.empty_cache()
        del model



        result_folder = "./caption/"+args.type
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        model_answer_folder = os.path.join(result_folder, 'image_captions')
        if not os.path.exists(model_answer_folder):
            os.makedirs(model_answer_folder)
        with open(os.path.join(model_answer_folder, "caption.jsonl"), 'w') as f:
            for pred in model_answers:
                f.write(json.dumps(pred) + '\n')
        

        

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)


