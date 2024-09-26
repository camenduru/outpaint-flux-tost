import os, json, requests, random, time, runpod

import torch
from diffusers.utils import load_image
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from PIL import Image, ImageOps
import numpy as np
from torch import nn
import torch.amp.autocast_mode
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM

class ImageAdapter(nn.Module):
	def __init__(self, input_features: int, output_features: int):
		super().__init__()
		self.linear1 = nn.Linear(input_features, output_features)
		self.activation = nn.GELU()
		self.linear2 = nn.Linear(output_features, output_features)

	def forward(self, vision_outputs: torch.Tensor):
		x = self.linear1(vision_outputs)
		x = self.activation(x)
		x = self.linear2(x)
		return x

CLIP_PATH = "/content/siglip"
MODEL_PATH = "/content/llama"

with torch.inference_mode():
    clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
    clip_model = AutoModel.from_pretrained(CLIP_PATH)
    clip_model = clip_model.vision_model
    clip_model.eval()
    clip_model.requires_grad_(False)
    clip_model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    text_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)
    text_model.eval()
    image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size)
    image_adapter.load_state_dict(torch.load("/content/adapter/image_adapter.pt", map_location="cpu"))
    image_adapter.eval()
    image_adapter.to("cuda")

with torch.inference_mode():
    controlnet = FluxControlNetModel.from_pretrained("/content/controlnet", torch_dtype=torch.bfloat16)
    transformer = FluxTransformer2DModel.from_pretrained("/content/model", subfolder='transformer', torch_dytpe=torch.bfloat16)
    pipe = FluxControlNetInpaintingPipeline.from_pretrained("/content/model", controlnet=controlnet, transformer=transformer, torch_dtype=torch.bfloat16).to("cuda")
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)

def expand_image(image_path, left_percent=0, right_percent=0, top_percent=0, bottom_percent=0, 
                 aspect_ratio_width=None, aspect_ratio_height=None, aspect_ratio_toggle=False, 
                 output_path="expanded_image.png", mask_output_path="expanded_mask_image.png"):
    img = Image.open(image_path)
    width, height = img.size
    left_expansion = int(width * (left_percent / 100))
    right_expansion = int(width * (right_percent / 100))
    top_expansion = int(height * (top_percent / 100))
    bottom_expansion = int(height * (bottom_percent / 100))
    new_width = width + left_expansion + right_expansion
    new_height = height + top_expansion + bottom_expansion
    expanded_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    expanded_img.paste(img, (left_expansion, top_expansion))
    mask_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    mask_img.paste((0, 0, 0), [left_expansion, top_expansion, left_expansion + width, top_expansion + height])
    if aspect_ratio_toggle and aspect_ratio_width and aspect_ratio_height:
        aspect_ratio_value = aspect_ratio_width / aspect_ratio_height
        if new_width / new_height > aspect_ratio_value:
            final_width = new_width
            final_height = int(new_width / aspect_ratio_value)
        else:
            final_height = new_height
            final_width = int(new_height * aspect_ratio_value)
        aspect_img = Image.new('RGB', (final_width, final_height), (255, 255, 255))
        x_offset = (final_width - new_width) // 2
        y_offset = (final_height - new_height) // 2
        aspect_img.paste(expanded_img, (x_offset, y_offset))
        expanded_img = aspect_img
        aspect_mask = Image.new('RGB', (final_width, final_height), (255, 255, 255))
        aspect_mask.paste((0, 0, 0), [x_offset + left_expansion, y_offset + top_expansion,
                                      x_offset + left_expansion + width, y_offset + top_expansion + height])
        mask_img = aspect_mask
        new_width, new_height = final_width, final_height
    if new_width != 768 and new_height != 768:
        if new_width < new_height:
            scaling_factor = 768 / new_width
            final_width = 768
            final_height = int(new_height * scaling_factor)
        else:
            scaling_factor = 768 / new_height
            final_height = 768
            final_width = int(new_width * scaling_factor)
        if final_height == 768:
            final_width = (final_width // 16) * 16
        elif final_width == 768:
            final_height = (final_height // 16) * 16
        expanded_img = expanded_img.resize((final_width, final_height), resample=Image.LANCZOS)
        mask_img = mask_img.resize((final_width, final_height), resample=Image.LANCZOS)
    expanded_img.save(output_path)
    mask_img.save(mask_output_path)
    return final_width, final_height

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file_name)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image = values['input_image']
    input_image = download_file(url=input_image, save_dir='/content', file_name='input_image')
    left_percent = values['left_percent']
    right_percent = values['right_percent']
    top_percent = values['top_percent']
    bottom_percent = values['bottom_percent']
    aspect_ratio_width = values['aspect_ratio_width']
    aspect_ratio_height = values['aspect_ratio_height']
    aspect_ratio_toggle = values['aspect_ratio_toggle']
    seed = values['seed']
    prompt_toggle = values['prompt_toggle']
    positive_prompt = values['positive_prompt']
    negative_prompt = values['negative_prompt']
    num_inference_steps = values['num_inference_steps']
    controlnet_conditioning_scale = values['controlnet_conditioning_scale']
    guidance_scale = values['guidance_scale']
    true_guidance_scale = values['true_guidance_scale']
    vlm_prompt = values['vlm_prompt']
    max_new_tokens = values['max_new_tokens']
    top_k = values['top_k']
    temperature = values['temperature']

    if prompt_toggle:
        joy_input_image = Image.open(input_image).convert("RGB")
        image = clip_processor(images=joy_input_image, return_tensors='pt').pixel_values
        image = image.to('cuda')
        joy_prompt = tokenizer.encode(vlm_prompt, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)
        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = clip_model(pixel_values=image, output_hidden_states=True)
            image_features = vision_outputs.hidden_states[-2]
            embedded_images = image_adapter(image_features)
            embedded_images = embedded_images.to('cuda')
        prompt_embeds = text_model.model.embed_tokens(joy_prompt.to('cuda'))
        embedded_bos = text_model.model.embed_tokens(torch.tensor([[tokenizer.bos_token_id]], device=text_model.device, dtype=torch.int64))
        inputs_embeds = torch.cat([
            embedded_bos.expand(embedded_images.shape[0], -1, -1),
            embedded_images.to(dtype=embedded_bos.dtype),
            prompt_embeds.expand(embedded_images.shape[0], -1, -1),
        ], dim=1)
        input_ids = torch.cat([
            torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
            joy_prompt,
        ], dim=1).to('cuda')
        attention_mask = torch.ones_like(input_ids)
        generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=True, top_k=top_k, temperature=temperature, suppress_tokens=None)
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        if generate_ids[0][-1] == tokenizer.eos_token_id:
            generate_ids = generate_ids[:, :-1]
        positive_prompt = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        print(positive_prompt)

    final_width, final_height = expand_image(input_image, left_percent=left_percent, right_percent=right_percent, top_percent=top_percent, bottom_percent=bottom_percent, aspect_ratio_width=aspect_ratio_width,
                                            aspect_ratio_height=aspect_ratio_height, aspect_ratio_toggle=aspect_ratio_toggle, output_path='/content/expanded_image.png', mask_output_path="/content/expanded_mask_image.png")
    image = load_image('/content/expanded_image.png').convert("RGB")
    mask = load_image('/content/expanded_mask_image.png').convert("RGB")
    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, np.iinfo(np.int32).max)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    final_width = closestNumber(final_width, 16)
    final_height = closestNumber(final_height, 16)
    outpaint = pipe(prompt=positive_prompt, height=final_height, width=final_width, control_image=image, control_mask=mask, num_inference_steps=num_inference_steps, 
                generator=generator, controlnet_conditioning_scale=controlnet_conditioning_scale, guidance_scale=guidance_scale, negative_prompt=negative_prompt, true_guidance_scale=true_guidance_scale).images[0]
    outpaint.save(f"/content/outpaint-flux--{final_width}x{final_height}--tost.png")

    result = f"/content/outpaint-flux--{final_width}x{final_height}--tost.png"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)
        if os.path.exists(input_image):
            os.remove(input_image)

runpod.serverless.start({"handler": generate})