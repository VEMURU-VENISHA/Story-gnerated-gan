import torch
from diffusers import StableDiffusionPipeline
from flask import Flask, render_template, request
import pyttsx3
import os
import time
app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")
model_id = "stabilityai/sd-turbo"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
print("⏳ Loading turbo diffusion model...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch_dtype
).to(device)
pipe.enable_attention_slicing()
print("✅ Model loaded successfully.\n")
def generate_image(prompt, index):
    print(f"🎨 Generating image for: {prompt}")
    start_time = time.time()
    with torch.no_grad():
        if device == "cuda":
            with torch.autocast("cuda"):
                image = pipe(prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
        else:
            image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
    os.makedirs("static/generated", exist_ok=True)
    img_path = f"static/generated/scene_{index+1}.png"
    image.save(img_path)
    print(f"💾 Saved: {img_path} ({time.time() - start_time:.2f}s)")
    return img_path
def split_story_into_scenes(story_text):
    story_text = story_text.replace(" and then ", ". ")
    scenes = [s.strip() for s in story_text.split('.') if s.strip()]
    return scenes[:3]
def generate_audio(story_text):
    audio_path = "static/generated/story_audio.mp3"
    engine = pyttsx3.init()
    engine.save_to_file(story_text, audio_path)
    engine.runAndWait()
    return audio_path
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/generate', methods=['POST'])
def generate():
    story = request.form['story'].strip()
    if not story:
        return render_template('index.html', error="Please enter a story.")
    scenes = split_story_into_scenes(story)
    print(f"\n📖 Your story has {len(scenes)} scenes.\n")
    image_paths = []
    for i, scene in enumerate(scenes):
        path = generate_image(scene, i)
        image_paths.append(path)
    audio_path = generate_audio(story)
    print(f"🔊 Audio saved at: {audio_path}")
    return render_template('result.html', story=story, images=image_paths, audio=audio_path)
if __name__ == '__main__':
    app.run(debug=True)
