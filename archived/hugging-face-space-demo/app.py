# !pip install -Uq diffusers transformers
# !pip install -Uq gradio
# !pip install -Uq accelerate

import gradio
from diffusers import StableDiffusionPipeline as pipeline
from accelerate import init_empty_weights
import torch
import os

api_key = os.environ['api_key']
my_token = api_key

with init_empty_weights():
  pipe = pipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=my_token).to("cuda")

DROPDOWNS = {
    "gustav": " by dan mumford and gustav klimt and john harris and jean delville and victo ngai and josan gonzalez",
    "hayao": " by studio ghibli",
    "vinny": " painting by Vincent van Gogh",
    "danny": " drawn by a child",
    "jeff": " by jeff koons",
}

def image_prompt(prompt, dropdown):
  prompt = prompt + DROPDOWNS[dropdown]
  return pipe(prompt=prompt, height=512, width=512).images[0]

with gradio.Blocks(css="""
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
""") as demo:
  dropdown = gradio.Dropdown(["danny", "gustav", "hayao", "vinny", "jeff"], label="choose style...")
  prompt = gradio.Textbox(label="image prompt...", elem_id="input-text")
  output = gradio.Image(elem_id="output-image")
  go_button = gradio.Button("draw it!", elem_id="go-button")
  go_button.click(fn=image_prompt, inputs=[prompt, dropdown], outputs=output)

demo.launch()