import os
from PIL import Image
import json
import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
# from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
def load_dataset(folder):
    dataset=[]
    with open("./labels/caption.jsonl", 'r') as f:
        for line in f:
            data = json.loads(line)        
            dataset.append(data)
    for i, d in enumerate(dataset):
        img_id=d['img_id']
        image_filename=str(img_id)+'.jpg'
        dataset[i]['images'] = os.path.join(folder,image_filename)
    return dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)   


folder = './all_image/'
dataset = load_dataset(folder)


def load_image(image_path):
    image = Image.open(image_path)
    return preprocess(image).unsqueeze(0).to(device)

def get_image_features(image_path):
    image = load_image(image_path)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features / image_features.norm(dim=-1, keepdim=True)

def cosine_similarity(a, b):
    return (a @ b.T).cpu().item()  


def find_top_k_similar_images(query_image_path, features, k):
    query_features = get_image_features(query_image_path)
    similarities = []
    for fe , image_name in features:
        similarity = cosine_similarity(query_features, fe)
        similarities.append((similarity, image_name))
    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[1:k]

folder_path = ".all_image"

features=[]
tmp = 700
progress_bar = tqdm(total=tmp,desc='Processing')
for i in range(1,701):
    image_name=str(i)+".jpg"
    image_path = os.path.join(folder_path, image_name)
    if os.path.isfile(image_path):
        image_features = get_image_features(image_path)
        features.append((image_features,image_name))
    progress_bar.update()

progress_bar.close()


types="Activity,Forecast,Localization/location,Localization/spatial,Object/affordance,Object/attribute,Object/existence,Reasoning/comparing,Reasoning/counting,Reasoning/situated,Planning/assistance,Planning/navigation"

types = list(types.split(','))  
for type in types:
    results=[]
    data_path=os.path.join("./data",type,"images")
    print("data_path:",data_path)
    items = os.listdir(data_path)
    num=len(items)
    print("num:",num)
    for i in range(1,num+1):
        image_title=str(i)+'.jpg'
        query_image = os.path.join(data_path, image_title)    
        top_k_similar_images = find_top_k_similar_images(query_image, features, k=4)
        top_k=[]
        for similarity, image_name in top_k_similar_images:
            if image_name==i:
                continue
            img_id=int(image_name.replace('.jpg',''))
            caption=dataset[int(img_id-1)]['caption']
            top_k.append({"img_id": img_id,"similarity":similarity, "caption": caption})
        results.append(top_k)

    result_folder = os.path.join("./labels",type)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    with open(os.path.join(result_folder, "similar3.jsonl"), 'w') as f:
        for pred in results:
            f.write(json.dumps(pred) + '\n')






