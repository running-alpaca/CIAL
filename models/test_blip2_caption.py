import torch
import contextlib
from types import MethodType
from lavis.models import load_model_and_preprocess
from . import get_image
from lavis import registry
import requests
from PIL import Image
from io import BytesIO

def image_parser(image_file):
    out = image_file.split(',')
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def new_maybe_autocast(self, dtype=None):
    enable_autocast = self.device != torch.device("cpu")
    if not enable_autocast:
        return contextlib.nullcontext()
    elif dtype is torch.bfloat16 and torch.cuda.is_bf16_supported():
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return contextlib.nullcontext()


def postprocess(output: str):
    output = output.strip().split('\n')[0].strip()
    return output.replace('<s>', '').strip()

def load_offline(model_name, model_type, model_config_path):
    model_class = registry.get_model_class(model_name)
    model_class.PRETRAINED_MODEL_CONFIG_DICT[model_type] = model_config_path
    model, image_processor, text_processor = load_model_and_preprocess(name=model_name, model_type=model_type, device='cpu', is_eval=True)

    return model, image_processor, text_processor


def format_question(caption, question, planning=False):
    if planning:
        return f"Please answer the following question in a detailed and helpful way. List steps to follow if needed. \nQuestion: {question} \nHere is the caption of the image:{caption} \nAnswer:"
    return f"Please answer the following question in a few words as short as possible. \nHere is the caption of the image:{caption} \nQuestion: {question} \nAnswer:"

class TestBlip2:
    def __init__(self, name='blip2_opt', model_type='pretrain_opt6.7b', config_path=None, device=None) -> None:
        # self.model, self.vis_processors, _ = load_model_and_preprocess(
        #     name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device='cpu'
        # )
        if config_path:
            self.model, self.vis_processors, _ = load_offline(
                model_name=name, model_type=model_type, model_config_path=config_path
            )
        else:
            self.model, self.vis_processors, _ = load_model_and_preprocess(
                name=name, model_type=model_type, is_eval=True, device='cpu'
            )
        self.model.maybe_autocast = MethodType(new_maybe_autocast, self.model)

        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float32 # torch.float16
            self.device = device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
            self.model.visual_encoder = self.model.visual_encoder.to(self.device, dtype=self.dtype)
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def generate(self, image, question, caption, max_new_tokens=300, planning=False):
        image = get_image(image)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device, dtype=self.dtype)
        prompt= format_question(caption, question, planning=planning)
        answer = self.model.generate({
            "image": image, "prompt":prompt
        }, max_length=max_new_tokens)
        if planning:
            output=answer[0]
        else:
            output=postprocess(answer[0])
        return output


    @torch.no_grad()
    def pure_generate(self, image, question, max_new_tokens=30):
        image = get_image(image)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device, dtype=self.dtype)
        answer = self.model.generate({
            "image": image, "prompt": question
        }, max_length=max_new_tokens)

        return answer[0]
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=30):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.vis_processors["eval"](x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device, dtype=self.dtype)
        prompts = [format_question(question) for question in question_list]
        # import pdb; pdb.set_trace()
        output = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)
        # import pdb; pdb.set_trace()
        output = [postprocess(x) for x in output]
        return output
    