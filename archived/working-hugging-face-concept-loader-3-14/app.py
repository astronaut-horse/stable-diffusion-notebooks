#@title Prepare the Concepts Library to be used


import requests
import os
import gradio as gr
import wget
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from huggingface_hub import HfApi
from transformers import CLIPTextModel, CLIPTokenizer
import html

community_icon_html = ""

loading_icon_html = ""
share_js = ""

api = HfApi()
models_list = api.list_models(author="sd-concepts-library", sort="likes", direction=-1)
models = []

my_token = os.environ['api_key']

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", revision="fp16", torch_dtype=torch.float16, use_auth_token=my_token).to("cuda")

def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
  loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
  
  _old_token = token
  # separate token and the embeds
  trained_token = list(loaded_learned_embeds.keys())[0]
  embeds = loaded_learned_embeds[trained_token]

  # cast to dtype of text_encoder
  dtype = text_encoder.get_input_embeddings().weight.dtype
  
  # add the token in tokenizer
  token = token if token is not None else trained_token
  num_added_tokens = tokenizer.add_tokens(token)
  i = 1
  while(num_added_tokens == 0):
    token = f"{token[:-1]}-{i}>"
    num_added_tokens = tokenizer.add_tokens(token)
    i+=1
  
  # resize the token embeddings
  text_encoder.resize_token_embeddings(len(tokenizer))
  
  # get the id for the token and assign the embeds
  token_id = tokenizer.convert_tokens_to_ids(token)
  text_encoder.get_input_embeddings().weight.data[token_id] = embeds
  return token


ahx_model_list = [model for model in models_list if "ahx" in model.modelId]


for model in ahx_model_list:
  model_content = {}
  model_id = model.modelId
  model_content["id"] = model_id
  embeds_url = f"https://huggingface.co/{model_id}/resolve/main/learned_embeds.bin"
  os.makedirs(model_id,exist_ok = True)
  if not os.path.exists(f"{model_id}/learned_embeds.bin"):
    try:
      wget.download(embeds_url, out=model_id)
    except:
      continue

  token_identifier = f"https://huggingface.co/{model_id}/raw/main/token_identifier.txt"
  response = requests.get(token_identifier)
  token_name = response.text
  
  concept_type = f"https://huggingface.co/{model_id}/raw/main/type_of_concept.txt"
  response = requests.get(concept_type)
  concept_name = response.text
  model_content["concept_type"] = concept_name
  images = []
  for i in range(4):
    url = f"https://huggingface.co/{model_id}/resolve/main/concept_images/{i}.jpeg"
    image_download = requests.get(url)
    url_code = image_download.status_code
    if(url_code == 200):
      file = open(f"{model_id}/{i}.jpeg", "wb") ## Creates the file for image
      file.write(image_download.content) ## Saves file content
      file.close()
      images.append(f"{model_id}/{i}.jpeg")
  model_content["images"] = images
  #if token cannot be loaded, skip it
  try:
    learned_token = load_learned_embed_in_clip(f"{model_id}/learned_embeds.bin", pipe.text_encoder, pipe.tokenizer, token_name)
  except: 
    continue
  model_content["token"] = learned_token
  models.append(model_content)
  models.append(model_content)


# -----------------------------------------------------------------------------------------------


#@title Dropdown Prompt Tab

model_tags = [model.modelId.split("/")[1] for model in ahx_model_list]
model_tags.sort()


import random 


#@title Gradio Concept Loader
DROPDOWNS = {}

for model in model_tags:
  if model != "ahx-model-1" and model != "ahx-model-2":
    DROPDOWNS[model] = f" in the style of <{model}>"

# def image_prompt(prompt, dropdown, guidance, steps, seed, height, width):
def image_prompt(prompt, guidance, steps, seed, height, width):
  # prompt = prompt + DROPDOWNS[dropdown]
  generator = torch.Generator(device="cuda").manual_seed(int(seed))
  return (
      pipe(prompt=prompt, guidance_scale=guidance, num_inference_steps=steps, generator=generator, height=int((height // 8) * 8), width=int((width // 8) * 8)).images[0], 
      f"prompt = '{prompt}'\nseed = {int(seed)}\nguidance_scale = {guidance}\ninference steps = {steps}\nheight = {int((height // 8) * 8)}\nwidth = {int((width // 8) * 8)}"
      )


def default_guidance():
  return 7.5

def default_steps():
  return 30

def default_pixel():
  return 768

def random_seed():
  return random.randint(0, 99999999999999) # <-- this is a random gradio limit, the seed range seems to actually be 0-18446744073709551615

with gr.Blocks(css=".gradio-container {max-width: 650px}") as dropdown_tab:
  gr.Markdown('''
      # 🧑‍🚀 Advanced Concept Loader

      This tool allows you to run your own text prompts into fine-tuned artist concepts with individual parameter controls. Text prompts need to manually include artist concept / model tokens, see the examples below. The seed controls the static starting.
      <br>
      <br>
      The images you generate here are not recorded unless you choose to share them. Please share any cool images / prompts on the community tab here or our discord server! 
      <br>
      <br>
      <a href="http://www.astronaut.horse">http://www.astronaut.horse</a>
  ''')

  with gr.Row():
    prompt = gr.Textbox(label="image prompt...", elem_id="input-text")
  with gr.Row():
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

  go_button = gr.Button("generate image", elem_id="go-button")
  output = gr.Image(elem_id="output-image")
  output_text = gr.Text(elem_id="output-text")
  # go_button.click(fn=image_prompt, inputs=[prompt, dropdown, guidance, steps, seed, height, width], outputs=[output, output_text])
  go_button.click(fn=image_prompt, inputs=[prompt, guidance, steps, seed, height, width], outputs=[output, output_text])
  gr.Markdown('''
    ## Prompt Examples Using Artist Tokens:
    * "an alien in the style of \<ahx-model-12>"
    * "a painting in the style of \<ahx-model-11>"
    * "a landscape in the style of \<ahx-model-10> and \<ahx-model-14> "

    ## Valid Artist Tokens:
    * \<ahx-model-3>
    * \<ahx-model-4>
    * \<ahx-model-6>
    * \<ahx-model-7>
    * \<ahx-model-9>
    * \<ahx-model-10>
    * \<ahx-model-11>
    * \<ahx-model-12>
    * \<ahx-model-13>
    * \<ahx-model-14>
  ''')

# -----------------------------------------------------------------------------------------------


#@title Dropdown Prompt Tab

model_tags = [model.modelId.split("/")[1] for model in ahx_model_list]
model_tags.sort()


import random 


#@title Gradio Concept Loader
DROPDOWNS = {}

for model in model_tags:
  if model != "ahx-model-1" and model != "ahx-model-2":
    DROPDOWNS[model] = f" in the style of <{model}>"

# def image_prompt(prompt, dropdown, guidance, steps, seed, height, width):



def default_guidance():
  return 7.5

def default_steps():
  return 30

def default_pixel():
  return 768

def random_seed():
  return random.randint(0, 99999999999999) # <-- this is a random gradio limit, the seed range seems to actually be 0-18446744073709551615


def simple_image_prompt(prompt, dropdown):
  seed = random_seed()
  guidance = 7.5
  height = 768
  width = 768
  steps = 30

  prompt = prompt + DROPDOWNS[dropdown]
  generator = torch.Generator(device="cuda").manual_seed(int(seed))
  return (
      pipe(prompt=prompt, guidance_scale=guidance, num_inference_steps=steps, generator=generator, height=int((height // 8) * 8), width=int((width // 8) * 8)).images[0], 
      f"prompt = '{prompt}'\nseed = {int(seed)}\nguidance_scale = {guidance}\ninference steps = {steps}\nheight = {int((height // 8) * 8)}\nwidth = {int((width // 8) * 8)}"
      )


with gr.Blocks(css=".gradio-container {max-width: 650px}") as new_welcome:
  gr.Markdown('''
      # 🧑‍🚀 Astronaut Horse Concept Loader

      This tool allows you to run your own text prompts into fine-tuned artist concepts from an ongoing series of Stable Diffusion collaborations with visual artists linked below. Select an artist's fine-tuned concept / model from the dropdown and enter any desired text prompt. You can check out example output images and project details on the project's webpage. Additionally if you can play around with more controls in the Advanced Prompting tab.
      <br>
      <br>
      The images you generate here are not recorded unless you choose to share them. Please share any cool images / prompts on the community tab here or our discord server! 
      <br>
      <br>
      <a href="http://www.astronaut.horse">http://www.astronaut.horse</a>
  ''')
  dropdown = gr.Dropdown(list(DROPDOWNS), label="choose style...")

  # with gr.Row():
  prompt = gr.Textbox(label="image prompt...", elem_id="input-text")

  go_button = gr.Button("generate image", elem_id="go-button")
  output = gr.Image(elem_id="output-image")
  output_text = gr.Text(elem_id="output-text")
  go_button.click(fn=simple_image_prompt, inputs=[prompt, dropdown], outputs=[output, output_text])

# -----------------------------------------------------------------------------------------------


def infer(text, dropdown):
  images_list = pipe(
              [f"{text} in the style of <{dropdown}>"],
              num_inference_steps=30,
              guidance_scale=7.5
  )
  return images_list.images, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)


css = ""
examples = []

with gr.Blocks(css=css) as demo:
  state = gr.Variable({
        'selected': -1
  })
  state = {}
  def update_state(i):
        global checkbox_states
        if(checkbox_states[i]):
          checkbox_states[i] = False
          state[i] = False
        else:
          state[i] = True
          checkbox_states[i] = True
  gr.Markdown('''
      # 🧑‍🚀 Astronaut Horse Concept Loader

      This tool allows you to run your own text prompts into fine-tuned artist concepts from an ongoing series of Stable Diffusion collaborations with visual artists linked below. Select an artist's fine-tuned concept / model from the dropdown and enter any desired text prompt. You can check out example output images and project details on the project's webpage. Additionally if you can play around with more controls in the Advanced Prompting tab. Enjoy!
      <a href="http://www.astronaut.horse">http://www.astronaut.horse</a>
  ''')
  with gr.Row():
        with gr.Column():
          dropdown = gr.Dropdown(list(DROPDOWNS), label="choose style...")
          text = gr.Textbox(
              label="Enter your prompt", placeholder="Enter your prompt", show_label=False, max_lines=1, elem_id="prompt_input"
          )
          btn = gr.Button("generate image",elem_id="run_btn")
          infer_outputs = gr.Gallery(show_label=False, elem_id="generated-gallery").style(grid=[1])
          with gr.Group(elem_id="share-btn-container"):
            community_icon = gr.HTML(community_icon_html, visible=False)
            loading_icon = gr.HTML(loading_icon_html, visible=False)
  checkbox_states = {}
  inputs = [text, dropdown]
  btn.click(
        infer,
        inputs=inputs,
        outputs=[infer_outputs, community_icon, loading_icon]
    )
  
  # -----------------------------------------------------------------------------------------------


tabbed_interface = gr.TabbedInterface([new_welcome, dropdown_tab], ["Welcome!", "Advanced Prompting"])
tabbed_interface.launch()