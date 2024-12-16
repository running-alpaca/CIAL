import torch
import transformers
from PIL import Image
# import requests
# import BytesIO
# from .otter_image.modeling_otter import OtterForConditionalGeneration
from otter_ai import OtterForConditionalGeneration
def get_image(image):
    if type(image) is str:
        try:
            return Image.open(image).convert("RGB")
        except Exception as e:
            print(f"Fail to read image: {image}")
            exit(-1)
    elif type(image) is Image.Image:
        return image
    else:
        raise NotImplementedError(f"Invalid type of Image: {type(image)}")


def get_formatted_prompt(prompt: str, in_context_prompts: list = []) -> str:
    in_context_string = ""
    for in_context_prompt, in_context_answer in in_context_prompts:
        in_context_string += f"<image>User: {in_context_prompt} GPT:<answer> {in_context_answer}<|endofchunk|>"
    return f"{in_context_string}<image>User: {prompt} GPT:<answer>"


def prompt_template_planning(prompt,caption):
    return f"User: You are a person in the situation shown in the image. Answer your question in a detailed and helpful way.\nHere are two images, an original image and a clip of the hands area.<image><image> \nHere is the caption of the hands area:{caption}  \nQuestion: {prompt}  GPT:<answer>"

def prompt_template(prompt,caption):
    return f"User: You are a person in the situation shown in the image. Answer the following question shortly and accurately! Keep your answer as short as possible! \nHere are two images, an original image and a clip of the hands area.<image><image> \nHere is the caption of the hands area:{caption}  \nQuestion: {prompt}  GPT:<answer>"

def prompt_template_planning_ret(prompt,caption, ret_caption):
    return f"User: You are a person in the situation shown in the image. Answer your question in a detailed and helpful way.\n \nHere are two images, an original image and a clip of the hands area.<image><image> \nHere is the caption of the hands area:{caption} \nHere is the image-caption pair similar to the test image:<image>{ret_caption}\nQuestion: {prompt}  GPT:<answer>"

def prompt_template_ret(prompt,caption, ret_caption):
    return f"User: You are a person in the situation shown in the image. Answer the following question shortly and accurately! Keep your answer as short as possible!\nHere are two images, an original image and a clip of the hands area.<image><image> \nHere is the caption of the hands area:{caption} \nHere is the image-caption pair similar to the test image:<image>{ret_caption} \nQuestion: {prompt}  GPT:<answer>"

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


class TestOtterImage:
    def __init__(self, name='image', device=None) -> None:
        if name == 'image':
            model_path = r'./VLMs/otter/OTTER-Image-MPT7B'
        elif name == 'video':
            model_path = r'./VLMs/otter/OTTER-Video-LLaMA7B-DenseCaption'

        model = OtterForConditionalGeneration.from_pretrained(model_path).to(device)
        model.text_tokenizer.padding_side = "left"
        image_processor = transformers.CLIPImageProcessor()
        model.eval()
        self.model = model
        self.image_processor = image_processor

    def move_to_device(self, device):
        pass

    @torch.no_grad()
    def generate(self, raw_image, question,caption,ret_img, ret_caption, max_new_tokens=30, planning=False):
        if ret_img=='':
            if planning:
                question = prompt_template_planning(question,caption)
            else:
                question = prompt_template(question,caption)
        else:
            raw_image=raw_image+","+ret_img
            if planning:
                question = prompt_template_planning_ret(question,caption, ret_caption)
            else:
                question = prompt_template_ret(question,caption, ret_caption)      

        image_files = image_parser(raw_image)
        images = load_images(image_files)
        vision_x = self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
      
        lang_x = self.model.text_tokenizer(
            [
                question
            ],
            return_tensors="pt",
        )
        bad_words_id = self.model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
        # import pdb; pdb.set_trace()
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device),
            max_new_tokens=max_new_tokens,
            bad_words_ids=bad_words_id,
            do_sample=False,
            temperature=0,
        )
        parsed_output = (
            self.model.text_tokenizer.decode(generated_text[0])
            .split("<answer>")[-1]
            .lstrip()
            .rstrip()
            .split("<|endofchunk|>")[0]
            .lstrip()
            .rstrip()
            .lstrip('"')
            .rstrip('"')
        )
        
        return parsed_output

    @torch.no_grad()
    def pure_generate(self, raw_image, question, max_new_tokens=30):
        raw_image = get_image(raw_image)
        vision_x = self.image_processor.preprocess([raw_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        lang_x = self.model.text_tokenizer([question], return_tensors="pt")
        bad_words_id = self.model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device),
            max_new_tokens=max_new_tokens,
            bad_words_ids=bad_words_id,
            do_sample=False,
            temperature=0,
        )
        parsed_output = (
            self.model.text_tokenizer.decode(generated_text[0])
            .split("<answer>")[-1]
            .lstrip()
            .rstrip()
            .split("<|endofchunk|>")[0]
            .lstrip()
            .rstrip()
            .lstrip('"')
            .rstrip('"')
        )
        
        return parsed_output

    def batch_generate(self, image_list, question_list, max_new_tokens=30):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.image_processor.preprocess([x], return_tensors="pt")["pixel_values"].unsqueeze(0) for x in imgs]
        vision_x = (torch.stack(imgs, dim=0))
        prompts = [get_formatted_prompt(question, []) for question in question_list]
        lang_x = self.model.text_tokenizer(prompts, return_tensors="pt", padding=True)
        bad_words_id = self.model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device),
            max_new_tokens=max_new_tokens,
            bad_words_ids=bad_words_id,
            do_sample=False,
            temperature=0,
        )
        total_output = []
        for i in range(len(generated_text)):
            parsed_output = (
                self.model.text_tokenizer.decode(generated_text[i])
                .split("<answer>")[-1]
                .lstrip()
                .rstrip()
                .split("<|endofchunk|>")[0]
                .lstrip()
                .rstrip()
                .lstrip('"')
                .rstrip('"')
            )
            total_output.append(parsed_output)

        return total_output
if __name__ == "__main__":
    model = TestOtterImage(device='cuda:0')
    image = "/data/Activity/images/3_153.jpg"
    qs = "What am I doing?"
    output = model.generate(image, qs)
    print(output)