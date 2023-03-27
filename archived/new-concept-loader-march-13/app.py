# !pip install -qq diffusers==0.11.1 transformers ftfy accelerate
#@title Import required libraries
import os
import torch

import PIL
from PIL import Image

from diffusers import StableDiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


#@title Login to the Hugging Face Hub
from huggingface_hub import notebook_login
# hf_token_write = "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" # 🤫
hf_token_write = os.environ['api_key']

# notebook_login()

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

## Run Stable Diffusion with pre-trained Learned Concepts

pretrained_model_name_or_path = "stabilityai/stable-diffusion-2"



from IPython.display import Markdown
from huggingface_hub import hf_hub_download

#@title Concept Pipe Function

def create_concept_pipe(model_name):
  # 1. Load Concept
  repo_id_embeds = f"sd-concepts-library/{model_name}" # <-------- CONCEPT NAME

  embeds_url = "" #Add the URL or path to a learned_embeds.bin file in case you have one
  placeholder_token_string = "" #Add what is the token string in case you are uploading your own embed

  downloaded_embedding_folder = "./downloaded_embedding"
  if not os.path.exists(downloaded_embedding_folder):
    os.mkdir(downloaded_embedding_folder)
  if(not embeds_url):
    embeds_path = hf_hub_download(repo_id=repo_id_embeds, filename="learned_embeds.bin")
    token_path = hf_hub_download(repo_id=repo_id_embeds, filename="token_identifier.txt")
    !cp $embeds_path $downloaded_embedding_folder
    !cp $token_path $downloaded_embedding_folder
    with open(f'{downloaded_embedding_folder}/token_identifier.txt', 'r') as file:
      placeholder_token_string = file.read()
  else:
    !wget -q -O $downloaded_embedding_folder/learned_embeds.bin $embeds_url

  learned_embeds_path = f"{downloaded_embedding_folder}/learned_embeds.bin"

#   display (Markdown("## The placeholder token for your concept is `%s`"%(placeholder_token_string)))



  # 2. Set up the Tokenizer and the Text Encoder
  tokenizer = CLIPTokenizer.from_pretrained(
      pretrained_model_name_or_path,
      subfolder="tokenizer",
  )
  text_encoder = CLIPTextModel.from_pretrained(
      pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch.float16
  )



  # 3. Load the newly learned embeddings into CLIP
  def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
    
    # separate token and the embeds
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
      raise ValueError(f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")
    
    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds
  load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer)



  # 4. Load the Stable Diffusion pipeline
  pipe = StableDiffusionPipeline.from_pretrained(
      pretrained_model_name_or_path,
      torch_dtype=torch.float16,
      text_encoder=text_encoder,
      tokenizer=tokenizer,
  ).to("cuda")

  return pipe

# Load All Concept Pipes

models_to_load = [
    # "ahx-model-3",
    # "ahx-model-5",
    # "ahx-model-6",
    # "ahx-model-7",
    # "ahx-model-8",
    # "ahx-model-9",
    # "ahx-model-10",
    "ahx-model-11",
    "ahx-model-12",
    # "ahx-model-13",
    # "ahx-model-14",
]

completed_concept_pipes = {}

for model in models_to_load:
  completed_concept_pipes[model] = create_concept_pipe(model)

# Test Concept Pipes

#@title Create Image Function

import random 

def random_seed():
  return random.randint(0, 18446744073709551615)

def create_image(concept="", prompt="", height=768, width=768, steps=30, guidance=7.5, seed=None):
  complete_prompt = f"{prompt} in the style of \u003C{concept}>"

  if seed is None:
    seed = random_seed()

  num_samples = 1
  num_rows = 1

  all_images = [] 
  for _ in range(num_rows):
      pipe = completed_concept_pipes[concept]

      generator = torch.Generator(device="cuda").manual_seed(seed)

      images = pipe(complete_prompt, num_images_per_prompt=num_samples, num_inference_steps=steps, guidance_scale=guidance, height=int((height // 8) * 8), width=int((width // 8) * 8), generator=generator).images

      # images = pipe(complete_prompt, num_images_per_prompt=num_samples, height=height, width=width, num_inference_steps=30, guidance_scale=7.5).images
      all_images.extend(images)

  grid = image_grid(all_images, num_samples, num_rows)

  return {
      "complete_prompt": complete_prompt,
      "seed": seed,
      "guidance": guidance,
      "inf_steps": steps,
      "grid": grid,
  }

#@title Test Text-to-Image Functionality

concept = "ahx-model-11" #@param {type:"string"}
prompt = "forgotten" #@param {type:"string"}

height = 525 #@param {type:"integer"}
width = 1700 #@param {type:"integer"}
# max square --> 983 x 983 --> 966,289 px^2
# default good square --> 768 x 768

guidance = 7.5 #@param {type:"number"}
steps = 30 #@param {type:"integer"}
seed = None #@param {type:"integer"}

image_obj = create_image(concept, prompt, steps=steps, guidance=guidance, height=height, width=width, seed=seed)

print(image_obj)
image_obj["grid"]

# Create Gradio Interface

# !pip install gradio
import gradio as gr

#@title Gradio Concept Loader
DROPDOWNS = {}

# images = pipe(complete_prompt, num_images_per_prompt=num_samples, num_inference_steps=steps, guidance_scale=guidance, height=int((height // 8) * 8), width=int((width // 8) * 8), generator=generator).images

for model in models_to_load:
  # token = model.split("/")[1]
  DROPDOWNS[model] = f" in the style of <{model}>"

if "sd-concepts-library/ahx-model-5" in DROPDOWNS:
  DROPDOWNS["sd-concepts-library/ahx-model-5"] = f"{prompt} in the style of <ahx-model-4>"

def image_prompt(prompt, dropdown, guidance, steps, seed, height, width):
# def image_prompt(prompt, dropdown, seed):
  prompt = prompt + DROPDOWNS[dropdown]
  pipe = completed_concept_pipes[dropdown]
  generator = torch.Generator(device="cuda").manual_seed(int(seed))
  return (
      pipe(prompt=prompt, guidance_scale=guidance, num_inference_steps=steps, generator=generator, height=int((height // 8) * 8), width=int((width // 8) * 8)).images[0], 
      f"prompt = '{prompt}'\nseed = {int(seed)}\nguidance_scale = {guidance}\ninference steps = {steps}\nheight = {int((height // 8) * 8)}\nwidth = {int((width // 8) * 8)}"
      )
  # images = pipe(complete_prompt, num_images_per_prompt=num_samples, num_inference_steps=steps, guidance_scale=guidance, height=int((height // 8) * 8), width=int((width // 8) * 8), generator=generator).images
  # return pipe(prompt=prompt, height=768, width=768, generator=generator).images[0]


def default_guidance():
  return 7.5

def default_steps():
  return 30

def default_pixel():
  return 768

def random_seed():
  return random.randint(0, 99999999999999) # <-- this is a random gradio limit, the seed range seems to actually be 0-18446744073709551615

# with gr.Blocks(css=gradio_css) as demo:
with gr.Blocks(css=".gradio-container {max-width: 650px}") as demo:
  dropdown = gr.Dropdown(list(DROPDOWNS), label="choose style...")
  gr.Markdown("<u>styles</u>: check out examples of these at https://www.astronaut.horse/collaborations")
  prompt = gr.Textbox(label="image prompt...", elem_id="input-text")
  seed = gr.Slider(0, 99999999999999, label="seed", dtype=int, value=random_seed, interactive=True, step=1)
  with gr.Row():
    with gr.Column():
      guidance = gr.Slider(0, 10, label="guidance", dtype=float, value=default_guidance, step=0.1, interactive=True)
    with gr.Column():
      steps = gr.Slider(1, 100, label="inference steps", dtype=int, value=default_steps, step=1, interactive=True)
  with gr.Row():
    with gr.Column():
      height = gr.Slider(50, 3500, label="height", dtype=int, value=default_pixel, step=1, interactive=True)
    with gr.Column():
      width = gr.Slider(50, 3500, label="width", dtype=int, value=default_pixel, step=1, interactive=True)
  gr.Markdown("<u>heads-up</u>: height multiplied by width should not exceed about 195,000 or an error will occur so don't go too nuts")

  # seed = gr.Slider(0, 18446744073709551615, label="seed", dtype=int, value=random_seed, interactive=True, step=1)
  # seed = gr.Slider(0, 18446744073709550591, label="seed", dtype=int, value=random_seed, interactive=True, step=1)
  # seed = gr.Slider(0, 18446744073709550, label="seed", dtype=int, value=random_seed, interactive=True, step=1)
  # seed = gr.Slider(0, 85835103557872, label="seed", dtype=int, value=random_seed, interactive=True, step=1)

  output = gr.Image(elem_id="output-image")
  output_text = gr.Text(elem_id="output-text")
  go_button = gr.Button("draw it!", elem_id="go-button")
  go_button.click(fn=image_prompt, inputs=[prompt, dropdown, guidance, steps, seed, height, width], outputs=[output, output_text])
  # go_button.click(fn=image_prompt, inputs=[prompt, dropdown, seed], outputs=output)

#@title Create Gradio Tab Interface

tabbed_interface = gr.TabbedInterface([demo], ["Concept Loader"])
tabbed_interface.launch(share=True)