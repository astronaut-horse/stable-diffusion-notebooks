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

from share_btn import community_icon_html, loading_icon_html, share_js

api = HfApi()
models_list = api.list_models(author="sd-concepts-library", sort="likes", direction=-1)
models = []

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True, revision="fp16", torch_dtype=torch.float16).to("cuda")

def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
  loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
  
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
    print(f"The tokenizer already contains the token {token}.")
    token = f"{token[:-1]}-{i}>"
    print(f"Attempting to add the token {token}.")
    num_added_tokens = tokenizer.add_tokens(token)
    i+=1
  
  # resize the token embeddings
  text_encoder.resize_token_embeddings(len(tokenizer))
  
  # get the id for the token and assign the embeds
  token_id = tokenizer.convert_tokens_to_ids(token)
  text_encoder.get_input_embeddings().weight.data[token_id] = embeds
  return token

print("Setting up the public library")
for model in models_list:
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
  
#@title Run the app to navigate around [the Library](https://huggingface.co/sd-concepts-library)
#@markdown Click the `Running on public URL:` result to run the Gradio app

SELECT_LABEL = "Select concept"
def assembleHTML(model):
  html_gallery = ''
  html_gallery = html_gallery+'''
  <div class="flex gr-gap gr-form-gap row gap-4 w-full flex-wrap" id="main_row">
  '''
  cap = 0
  for model in models:
    html_gallery = html_gallery+f'''
    <div class="gr-block gr-box relative w-full overflow-hidden border-solid border border-gray-200 gr-panel">
      <div class="output-markdown gr-prose" style="max-width: 100%;">
        <h3>
          <a href="https://huggingface.co/{model["id"]}" target="_blank">
            <code>{html.escape(model["token"])}</code>
          </a>
        </h3>
      </div>
      <div id="gallery" class="gr-block gr-box relative w-full overflow-hidden border-solid border border-gray-200">
        <div class="wrap svelte-17ttdjv opacity-0"></div>
        <div class="absolute left-0 top-0 py-1 px-2 rounded-br-lg shadow-sm text-xs text-gray-500 flex items-center pointer-events-none bg-white z-20 border-b border-r border-gray-100 dark:bg-gray-900">
          <span class="mr-2 h-[12px] w-[12px] opacity-80">
            <svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="feather feather-image">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <circle cx="8.5" cy="8.5" r="1.5"></circle>
              <polyline points="21 15 16 10 5 21"></polyline>
            </svg>
          </span> {model["concept_type"]}
        </div>
        <div class="overflow-y-auto h-full p-2" style="position: relative;">
          <div class="grid gap-2 grid-cols-2 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-2 2xl:grid-cols-2 svelte-1g9btlg pt-6">
        '''
    for image in model["images"]:
                html_gallery = html_gallery + f'''    
                <button class="gallery-item svelte-1g9btlg">
                  <img alt="" loading="lazy" class="h-full w-full overflow-hidden object-contain" src="file/{image}">
                </button>
                '''
    html_gallery = html_gallery+'''
              </div>
              <iframe style="display: block; position: absolute; top: 0; left: 0; width: 100%; height: 100%; overflow: hidden; border: 0; opacity: 0; pointer-events: none; z-index: -1;" aria-hidden="true" tabindex="-1" src="about:blank"></iframe>
            </div>
          </div>
        </div>
        '''
    cap += 1
    if(cap == 99):
      break  
  html_gallery = html_gallery+'''
  </div>
  '''
  return html_gallery
  
def title_block(title, id):
  return gr.Markdown(f"### [`{title}`](https://huggingface.co/{id})")

def image_block(image_list, concept_type):
  return gr.Gallery(
          label=concept_type, value=image_list, elem_id="gallery"
          ).style(grid=[2], height="auto")

def checkbox_block():
  checkbox = gr.Checkbox(label=SELECT_LABEL).style(container=False)
  return checkbox

def infer(text):
  #with autocast("cuda"):
  images_list = pipe(
              [text]*2,
              num_inference_steps=50,
              guidance_scale=7.5
  )
  #output_images = []
  #for i, image in enumerate(images_list.images):
  #  output_images.append(image)
  return images_list.images, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

# idetnical to `infer` function without gradio state updates for share btn
def infer_examples(text):
  #with autocast("cuda"):
  images_list = pipe(
              [text]*2,
              num_inference_steps=50,
              guidance_scale=7.5
  )
  #output_images = []
  #for i, image in enumerate(images_list["sample"]):
  #  output_images.append(image)
  return images_list.images
  
css = '''
.gradio-container {font-family: 'IBM Plex Sans', sans-serif}
#top_title{margin-bottom: .5em}
#top_title h2{margin-bottom: 0; text-align: center}
/*#main_row{flex-wrap: wrap; gap: 1em; max-height: 550px; overflow-y: scroll; flex-direction: row}*/
#component-3{height: 760px; overflow: auto}
#component-9{position: sticky;top: 0;align-self: flex-start;}
@media (min-width: 768px){#main_row > div{flex: 1 1 32%; margin-left: 0 !important}}
.gr-prose code::before, .gr-prose code::after {content: "" !important}
::-webkit-scrollbar {width: 10px}
::-webkit-scrollbar-track {background: #f1f1f1}
::-webkit-scrollbar-thumb {background: #888}
::-webkit-scrollbar-thumb:hover {background: #555}
.gr-button {white-space: nowrap}
.gr-button:focus {
  border-color: rgb(147 197 253 / var(--tw-border-opacity));
  outline: none;
  box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
  --tw-border-opacity: 1;
  --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
  --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
  --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
  --tw-ring-opacity: .5;
}
#prompt_input{flex: 1 3 auto; width: auto !important;}
#prompt_area{margin-bottom: .75em}
#prompt_area > div:first-child{flex: 1 3 auto}
.animate-spin {
    animation: spin 1s linear infinite;
}
@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}
#share-btn-container {
    display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
}
#share-btn {
    all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
}
#share-btn * {
    all: unset;
}
'''
examples = ["a <cat-toy> in <madhubani-art> style", "a <line-art> style mecha robot", "a piano being played by <bonzi>", "Candid photo of <cheburashka>, high resolution photo, trending on artstation, interior design"]

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
  gr.HTML('''
  <div style="text-align: center; max-width: 720px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Stable Diffusion Conceptualizer
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                Navigate through community created concepts and styles via Stable Diffusion Textual Inversion and pick yours for inference.
                To train your own concepts and contribute to the library <a style="text-decoration: underline" href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb">check out this notebook</a>.
              </p>
            </div>
  ''')
  with gr.Row():
        with gr.Column():
          gr.Markdown(f"### Navigate the top 100 Textual-Inversion community trained concepts. Use 600+ from [The Library](https://huggingface.co/sd-concepts-library)")
          with gr.Row():
                  image_blocks = []
                  #for i, model in enumerate(models):
                  with gr.Box().style(border=None):
                    gr.HTML(assembleHTML(models))
                      #title_block(model["token"], model["id"])
                      #image_blocks.append(image_block(model["images"], model["concept_type"]))
        with gr.Column():
          with gr.Box():
                  with gr.Row(elem_id="prompt_area").style(mobile_collapse=False, equal_height=True):
                      text = gr.Textbox(
                          label="Enter your prompt", placeholder="Enter your prompt", show_label=False, max_lines=1, elem_id="prompt_input"
                      ).style(
                          border=(True, False, True, True),
                          rounded=(True, False, False, True),
                          container=False,
                      )
                      btn = gr.Button("Run",elem_id="run_btn").style(
                          margin=False,
                          rounded=(False, True, True, False),
                      )  
                  with gr.Row().style():
                      infer_outputs = gr.Gallery(show_label=False, elem_id="generated-gallery").style(grid=[2], height="512px")
                  with gr.Row():
                    gr.HTML("<p style=\"font-size: 95%;margin-top: .75em\">Prompting may not work as you are used to. <code>objects</code> may need the concept added at the end, <code>styles</code> may work better at the beginning. You can navigate on <a href='https://lexica.art'>lexica.art</a> to get inspired on prompts</p>")
                  with gr.Row():
                    gr.Examples(examples=examples, fn=infer_examples, inputs=[text], outputs=infer_outputs, cache_examples=True)
          with gr.Group(elem_id="share-btn-container"):
            community_icon = gr.HTML(community_icon_html, visible=False)
            loading_icon = gr.HTML(loading_icon_html, visible=False)
            share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)
  checkbox_states = {}
  inputs = [text]
  btn.click(
        infer,
        inputs=inputs,
        outputs=[infer_outputs, community_icon, loading_icon, share_button]
    )
  share_button.click(
      None,
      [],
      [],
      _js=share_js,
  )
demo.queue(max_size=20).launch()