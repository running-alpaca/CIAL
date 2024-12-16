import os
import shutil
import json
def collect_images(types, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    image_count = 1

    for type in types:
        folder = './data/'+type+'/images'
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    source_path = os.path.join(root, file)
                    destination_path = os.path.join(destination_folder, f'{image_count}.jpg')
                    shutil.copy(source_path, destination_path)
                    image_count += 1     

def collect_captions(types, output_file):
    captions = []
    img_id = 1
    destination_folder='./all_image'
    for type in types:
        print("type:",type)
        print("captions",len(captions))
        folder="./caption/"+type+"/image_captions"
        img_folder='./data/'+type+'/images'
        for root, _, files in os.walk(folder):
            for file in files:
                if file==('caption.jsonl'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line)
                            img_name=data["img_id"]
                            file_name=str(img_name)+'.jpg'
                            source_path = os.path.join(img_folder, file_name)
                            destination_path = os.path.join(destination_folder, f'{img_id}.jpg')                                  
                            shutil.copy(source_path, destination_path)                                                 
                            if 'caption' in data:
                                captions.append({"img_id": img_id, "caption": data["caption"]})
                                img_id += 1

    
    with open(output_file, 'w', encoding='utf-8') as f:
        for caption in captions:
            f.write(json.dumps(caption) + '\n')

if __name__ == "__main__":
    types = ['Activity','Forecast','Localization/location','Localization/spatial','Object/affordance','Object/attribute','Object/existence','Planning/assistance','Planning/navigation','Reasoning/comparing','Reasoning/counting','Reasoning/situated'] 
    destination_folder = './all_image'
    output_file = './labels/captions.json'
    collect_captions(types, output_file)
    print(f'Captions collected and stored in {output_file}.')
