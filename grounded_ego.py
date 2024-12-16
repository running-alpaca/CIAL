import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--type", type=str, default="Activity")

    args = parser.parse_args()
    return args

args = parse_args()
"""
Hyper parameters
"""



TEXT_PROMPT = "hand." 
data_type = args.type
IMG_PATH = "./data/"+data_type+"/images/"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./clip_ego/"+data_type+"/"
DUMP_JSON_RESULTS = True


def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle




# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT
folder_path = IMG_PATH

for img_path in os.listdir(folder_path):
    img_path = IMG_PATH+img_path
    print("image_path:",img_path)
    filename = os.path.basename(img_path)
    print("image_name:",filename)
    img_id = os.path.splitext(filename)[0]
    output_dir = Path(OUTPUT_DIR+img_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_source, image = load_image(img_path)

    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


    # FIXME: figure how does this influence the G-DINO model
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
    # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__() 

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    

    if len(input_boxes)>0:      
        print("input_boxes:",input_boxes)
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)


        confidences = confidences.numpy().tolist()
        class_names = labels

        class_ids = np.array(list(range(len(class_names))))

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]

        """
        Visualize image with supervision useful API
        """
        img = cv2.imread(img_path)
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        cv2.imwrite(os.path.join(output_dir, filename), annotated_frame)
        DUMP_JSON_RESULTS=True
    
    else:
        img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(output_dir, filename), img)
        DUMP_JSON_RESULTS=False

    """
    Dump the results in standard format and save as json files
    """


    if DUMP_JSON_RESULTS:
        # convert mask into rle format
        mask_rles = [single_mask_to_rle(mask) for mask in masks]

        input_boxes = input_boxes.tolist()
        scores = scores.tolist()
        # save the results in standard format
        results = {
            "image_path": img_path,
            "annotations" : [
                {
                    "class_name": class_name,
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                }
                for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }
        
        with open(os.path.join(output_dir, "clip_results.json"), "w") as f:
            json.dump(results, f, indent=4)

        mins=min(scores)
        min_index = scores.index(mins)
        sort_scores=sorted(scores)
        print(sort_scores)
        if len(input_boxes)>2:
            if (sort_scores[1][0]-sort_scores[0][0])>=0.4:
                del input_boxes[min_index]
    if len(input_boxes)==2: 
        box1=input_boxes[0]
        box2=input_boxes[1]
        x=[box1[0],box1[2],box2[0],box2[2]]
        y=[box1[1],box1[3],box2[1],box2[3]]
        box=[min(x),min(y),max(x),max(y)]
    elif len(input_boxes)==1: 
        box=input_boxes[0]
    elif len(input_boxes)==0:   
        x1=w/6
        x2=w-x1
        y2=h
        y1=h/3
        box=[x1,y1,x2,y2]
    else: 
        max_y=0
        for b in input_boxes:
           if b[3]>max_y:
               box=b
               max_y=b[3]
        if max_y==0:
            x1=w/6
            x2=w-x1
            y2=h
            y1=h/3
            box=[x1,y1,x2,y2]

    box = [int(coord) for coord in box]
    print("box:",box)

    img_without_clip = cv2.imread(img_path)



    cropped_image = img_without_clip[box[1]:box[3], box[0]:box[2]]

    cv2.imwrite(os.path.join(output_dir, "clip0.jpg"), cropped_image)


    x1_1=box[0]-box[0]/3
    x2_1=(w-box[2])/3+box[2]
    if x2_1>w:
        x2_1=w
    y1_1=box[1]-box[1]/3
    y2_1=(h-box[3])/3+box[3]
    if y2_1>h:
        y2_1=h
    box1=[x1_1,y1_1,x2_1,y2_1]
    box1 = [int(coord) for coord in box1]
    print("box1:",box1)
    cropped_image1 = img_without_clip[box1[1]:box1[3], box1[0]:box1[2]]

    cv2.imwrite(os.path.join(output_dir, "clip1.jpg"), cropped_image1)    



    x1_2=box[0]/2
    x2_2=(w-box[2])/2+box[2]
    if x2_2>w:
        x2_2=w
    y1_2=box[1]/2
    y2_2=(h-box[3])/2+box[3]
    if y2_2>h:
        y2_2=h
    box2=[x1_2,y1_2,x2_2,y2_2]
    box2 = [int(coord) for coord in box2]
    print("box2:",box2)


    cropped_image2 = img_without_clip[box2[1]:box2[3], box2[0]:box2[2]]

    cv2.imwrite(os.path.join(output_dir, "clip2.jpg"), cropped_image2)        




