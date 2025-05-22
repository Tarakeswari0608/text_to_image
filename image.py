from flask import Flask, render_template, request, send_file, send_from_directory from image_generator import generate_image
import os
image = Flask( name ) UPLOAD_FOLDER = 'uploads'
 
image.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER os.makedirs(image.config['UPLOAD_FOLDER'], exist_ok=True) @image.route('/')
def home():
return render_template('index.html') @image.route('/generate_image', methods=['POST']) def generate_and_serve_image():
text_prompt = request.form['text_prompt'] generated_image = generate_image(text_prompt)
image_path=os.path.join(image.config['UPLOAD_FOLDER'], 'generated_image.png')
generated_image.save(image_path)
return send_file(image_path, mimetype='image/png') @image.route('/uploads/<filename>')
def uploaded_file(filename):
return send_from_directory(image.config['UPLOAD_FOLDER'], filename) if  name	== ' main ':
image.run(debug=True) image_generator.py import model_loader import pipeline
from PIL import Image
from transformers import CLIPTokenizer import torch
def generate_image(prompt): DEVICE = "cpu" ALLOW_CUDA = True
ALLOW_MPS = False
if torch.cuda.is_available() and ALLOW_CUDA: DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS: DEVICE = "mps"
print(f"Using device: {DEVICE}")
tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
 
model_file = "../data/v1-5-pruned-emaonly.ckpt" models=model_loader.preload_models_from_standard_weights(model_file,
DEVICE)
uncond_prompt = "" do_cfg = True cfg_scale = 8 input_image = None
image_path = "../images/dog.jpg" strength = 0.9
sampler = "ddpm" num_inference_steps = 50
seed = 42
# Generate image
output_image = pipeline.generate( prompt=prompt, uncond_prompt=uncond_prompt, input_image=input_image, strength=strength, do_cfg=do_cfg, cfg_scale=cfg_scale, sampler_name=sampler,
n_inference_steps=num_inference_steps, seed=seed,
models=models, device=DEVICE, idle_device="cpu", tokenizer=tokenizer, )
output_image_pil = Image.fromarray(output_image) return output_image_pil
