

#@title 1. General Setup

# FOR DEPLOYMENT:
# !pip install -qq diffusers==0.11.1 transformers ftfy accelerate
# !pip install -Uq diffusers transformers
# !pip install -Uq gradio
# !pip install -Uq accelerate

from diffusers import StableDiffusionPipeline
pipeline = StableDiffusionPipeline

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from accelerate import init_empty_weights
import gradio
import torch
import os

# UNDER CONSTRUCTION ---{{{
import subprocess
# }}}---

# FOR DEPLOYMENT: uncomment these and delete the notebook_login() below
api_key = os.environ['api_key']
my_token = api_key

# from huggingface_hub import notebook_login
# notebook_login()

# NOT NEEDED FOR DEPLOYMENT ---{{{
# import PIL
# from PIL import Image

# def image_grid(imgs, rows, cols):
#     assert len(imgs) == rows*cols

#     w, h = imgs[0].size
#     grid = Image.new('RGB', size=(cols*w, rows*h))
#     grid_w, grid_h = grid.size
    
#     for i, img in enumerate(imgs):
#         grid.paste(img, box=(i%cols*w, i//cols*h))
#     return grid
# }}}---

pretrained_model_name_or_path = "stabilityai/stable-diffusion-2"

# from IPython.display import Markdown
from huggingface_hub import hf_hub_download
     

#@title 2. Tell it What Concepts to Load

models_to_load = [
    "ahx-model-3",
    "ahx-model-5",
    "ahx-model-6",
    "ahx-model-7",
    "ahx-model-8",
    "ahx-model-9",
    "ahx-model-10",
    "ahx-model-11",
]

models_to_load = [f"sd-concepts-library/{model}" for model in models_to_load]
completed_concept_pipes = {}
     

#@title 3. Load the Concepts as Distinct Pipes

for repo_id_embeds in models_to_load:
  print(f"loading {repo_id_embeds}")
  print("----------------------")
  # repo_id_embeds = "sd-concepts-library/ahx-model-3"

  embeds_url = "" #Add the URL or path to a learned_embeds.bin file in case you have one
  placeholder_token_string = "" #Add what is the token string in case you are uploading your own embed

  downloaded_embedding_folder = "./downloaded_embedding"
  if not os.path.exists(downloaded_embedding_folder):
    os.mkdir(downloaded_embedding_folder)
  if(not embeds_url):
    embeds_path = hf_hub_download(repo_id=repo_id_embeds, filename="learned_embeds.bin")
    token_path = hf_hub_download(repo_id=repo_id_embeds, filename="token_identifier.txt")
    # FOR DEPLOYMENT: address file system use
    #!cp downloaded_embedding_folder
    #!cp downloaded_embedding_folder

    # UNDER CONSTRUCTION ---{{{
    subprocess.call([f"cp {embeds_path} {downloaded_embedding_folder}"])
    subprocess.call([f"cp {token_path} {downloaded_embedding_folder}"])
    # }}}---

    with open(f'{downloaded_embedding_folder}/token_identifier.txt', 'r') as file:
      placeholder_token_string = file.read()
  else:
    # FOR DEPLOYMENT: address file system use
    #!wget -q -O $downloaded_embedding_folder/learned_embeds.bin $embeds_url

    # UNDER CONSTRUCTION ---{{{
    subprocess.call([f"wget -q -O {downloaded_embedding_folder}/learned_embeds.bin {embeds_url}"])
    # }}}---

  learned_embeds_path = f"{downloaded_embedding_folder}/learned_embeds.bin"

  # ----

  tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
  )
  text_encoder = CLIPTextModel.from_pretrained(
      pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch.float16
  )

  # ----

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

  # FOR DEPLOYMENT: add use_auth_token=my_token to pipe keyword args
    # ie --> pipe = pipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=my_token).to("cuda")
  pipe = StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path,
    torch_dtype=torch.float16,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    use_auth_token=my_token,
  ).to("cuda")

  completed_concept_pipes[repo_id_embeds] = pipe
  print("--> complete !")
  print("----------------------")


     

#@title 4. Print Available Concept Strings

# NOT NEEDED FOR DEPLOYMENT ---{{{
# print("AVAILABLE CONCEPTS TO SELECT FROM")
# print("copy one and paste below under 'model'")
# print("------------------------------------------------------")
# # list(completed_concept_pipes)
# for model in completed_concept_pipes:
#   print(f"{model}")
# }}}---

#@title 5. Optionally Test without Gradio

# NOT NEEDED FOR DEPLOYMENT ---{{{
# model = "" #@param {type: "string"}
# prompt = "" #@param {type:"string"}

# if prompt and model:
#   if model not in completed_concept_pipes:
#     raise ValueError("Invalid Model Name")

#   model_token = model.split("/")[1]
#   prompt = f"{prompt} in the style of <{model_token}>"

#   if model == "sd-concepts-library/ahx-model-5":
#     prompt = f"{prompt} in the style of "

#   num_samples = 1
#   num_rows = 1

#   all_images = [] 
#   pipe = completed_concept_pipes[model]

#   for _ in range(num_rows):
#       images = pipe(prompt, num_images_per_prompt=num_samples, height=512, width=512, num_inference_steps=30, guidance_scale=7.5).images
#       all_images.extend(images)

#   grid = image_grid(all_images, num_samples, num_rows)
#   grid
# }}}---
     

#@title 6. Define Custom CSS for Gradio

use_custom_css = True

gradio_css = """
  #output-image {
    border: 1px solid black;
    background-color: white;
    width: 500px;
    display: block;
    margin-left: auto;
    margin-right: auto;
  }
"""

gradio_css_alternative = """
  #go-button {
    background-color: white;
    border-radius: 0;
    border: none;
    font-family: serif;
    background-image: none;
    font-weight: 100;
    width: fit-content;
    display: block;
    margin-left: auto;
    margin-right: auto;
    text-decoration: underline;
    box-shadow: none;
    color: blue;
  }
  .rounded-lg {
    border: none;
  }
  .gr-box {
    border-radius: 0;
    border: 1px solid black;
  }
  .text-gray-500 {
    color: black;
    font-family: serif;
    font-size: 15px;
  }
  .border-gray-200 {
    border: 1px solid black;
  }
  .bg-gray-200 {
    background-color: white;
    --tw-bg-opacity: 0;
  }
  footer {
    display: none;
  }
  footer {
    opacity: 0;
  }
  #output-image {
    border: 1px solid black;
    background-color: white;
    width: 500px;
    display: block;
    margin-left: auto;
    margin-right: auto;
  }
  .absolute {
    display: none;
  }
  #input-text {
    width: 500px;
    display: block;
    margin-left: auto;
    margin-right: auto;
    padding: 0 0 0 0;
  }
  .py-6 {
    padding-top: 0;
    padding-bottom: 0;
  }
  .px-4 {
    padding-left: 0;
    padding-right: 0;
  }
  .rounded-lg {
    border-radius: 0;
  }
  .gr-padded {
    padding: 0 0;
    margin-bottom: 12.5px;
  }
  .col > *, .col > .gr-form > * {
    width: 500px;
    margin-left: auto;
    margin-right: auto;
  }
"""
     

#@title 7. Build and Launch the Gradio Interface

DROPDOWNS = {}

for model in models_to_load:
  token = model.split("/")[1]
  DROPDOWNS[model] = f" in the style of <{token}>"

if "sd-concepts-library/ahx-model-5" in DROPDOWNS:
  DROPDOWNS["sd-concepts-library/ahx-model-5"] = f"{prompt} in the style of "

def image_prompt(prompt, dropdown):
  prompt = prompt + DROPDOWNS[dropdown]
  pipe = completed_concept_pipes[dropdown]
  return pipe(prompt=prompt, height=512, width=512).images[0]

with gradio.Blocks(css=gradio_css if use_custom_css else "") as demo:
  dropdown = gradio.Dropdown(list(DROPDOWNS), label="choose style...")
  prompt = gradio.Textbox(label="image prompt...", elem_id="input-text")
  output = gradio.Image(elem_id="output-image")
  go_button = gradio.Button("draw it!", elem_id="go-button")
  go_button.click(fn=image_prompt, inputs=[prompt, dropdown], outputs=output)

demo.launch(share=True)
     
