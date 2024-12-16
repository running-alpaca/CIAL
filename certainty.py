import torch
import clip
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from PIL import Image
import os
from tqdm import tqdm
# Load the CLIP model and processor
model = CLIPModel.from_pretrained("your path to clip-vit-large-patch14-336")
processor = CLIPProcessor.from_pretrained("your path to clip-vit-large-patch14-336")

types="Activity,Forecast,Localization/location,Localization/spatial,Object/affordance,Object/attribute,Object/existence,Planning/assistance,Planning/navigation,Reasoning/comparing,Reasoning/counting,Reasoning/situated"

# model which generating answer
model_name="LLaVA-1.5-7b"
types = list(types.split(','))  
print(types)

for type in types:
    # The sentences to compare
    answer_path="./results/"+type+"/model_answer/"+model_name+".jsonl"
    model_answers=[]
    if type in ['Activity','Forecast']:
        tmp=100
    else:
        tmp=50
    with open(answer_path, 'r') as answer_file:
        progress_bar = tqdm(total=tmp,desc='Processing')
        for line in answer_file:
            # print("*"*50)
            json_line = json.loads(line)
            answers=json_line["choices"][0]["turns"]
            answer_number=json_line["choices"][0]["index"]
            img_id=json_line["question_id"]+1
            image_path = "./data/"+type+"/images/"+str(img_id)+".jpg"
            image = Image.open(image_path)
            if answer_number>1:
                # Preprocess the image
                image_inputs = processor(images=image, return_tensors="pt")

                # Preprocess the sentences
                text_inputs = processor(text=answers, return_tensors="pt", padding=True, truncation=True)

                # Extract image features
                with torch.no_grad():
                    image_features = model.get_image_features(pixel_values=image_inputs['pixel_values'])

                # Extract text features
                with torch.no_grad():
                    text_features = model.get_text_features(input_ids=text_inputs['input_ids'], attention_mask=text_inputs['attention_mask'])

                # Convert the text features to numpy arrays
                text_features = text_features.numpy()
                image_features = image_features.numpy()
                image_features = np.repeat(image_features, answer_number, axis=0)

                # Compute the cosine similarities
                text_similarities = cosine_similarity(text_features)
                for i in range(text_similarities.shape[0]):
                    text_similarities[i, i] = 0
                score1=text_similarities.sum(axis=0)
                print("score1:",score1)
                
                img_similarities = cosine_similarity(text_features, image_features)
                score2=img_similarities.sum(axis=0)
                print("score2:",score2)

                score = [a + b for a, b in zip(score1, score2)]
                print("score:",score)
                max_index = score.index(max(score))
                print("max_index:",max_index)
                final_answer=answers[max_index]
            else:
                final_answer=answers[0]
            model_answers.append({
                'question_id': json_line["question_id"],
                'model_id': model_name,
                'choices':[{'index': 0, "turns": final_answer}]
            })
            progress_bar.update()

        progress_bar.close()


        result_folder = "./results/"+type
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        model_answer_folder = os.path.join(result_folder, 'final_answer')
        if not os.path.exists(model_answer_folder):
            os.makedirs(model_answer_folder)
        with open(os.path.join(model_answer_folder, f"{model_name}.jsonl"), 'w') as f:
            for pred in model_answers:
                f.write(json.dumps(pred) + '\n')

