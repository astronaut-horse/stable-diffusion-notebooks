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

community_icon_html = """<svg id="share-btn-share-icon" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32">
    <path d="M20.6081 3C21.7684 3 22.8053 3.49196 23.5284 4.38415C23.9756 4.93678 24.4428 5.82749 24.4808 7.16133C24.9674 7.01707 25.4353 6.93643 25.8725 6.93643C26.9833 6.93643 27.9865 7.37587 28.696 8.17411C29.6075 9.19872 30.0124 10.4579 29.8361 11.7177C29.7523 12.3177 29.5581 12.8555 29.2678 13.3534C29.8798 13.8646 30.3306 14.5763 30.5485 15.4322C30.719 16.1032 30.8939 17.5006 29.9808 18.9403C30.0389 19.0342 30.0934 19.1319 30.1442 19.2318C30.6932 20.3074 30.7283 21.5229 30.2439 22.6548C29.5093 24.3704 27.6841 25.7219 24.1397 27.1727C21.9347 28.0753 19.9174 28.6523 19.8994 28.6575C16.9842 29.4379 14.3477 29.8345 12.0653 29.8345C7.87017 29.8345 4.8668 28.508 3.13831 25.8921C0.356375 21.6797 0.754104 17.8269 4.35369 14.1131C6.34591 12.058 7.67023 9.02782 7.94613 8.36275C8.50224 6.39343 9.97271 4.20438 12.4172 4.20438H12.4179C12.6236 4.20438 12.8314 4.2214 13.0364 4.25468C14.107 4.42854 15.0428 5.06476 15.7115 6.02205C16.4331 5.09583 17.134 4.359 17.7682 3.94323C18.7242 3.31737 19.6794 3 20.6081 3ZM20.6081 5.95917C20.2427 5.95917 19.7963 6.1197 19.3039 6.44225C17.7754 7.44319 14.8258 12.6772 13.7458 14.7131C13.3839 15.3952 12.7655 15.6837 12.2086 15.6837C11.1036 15.6837 10.2408 14.5497 12.1076 13.1085C14.9146 10.9402 13.9299 7.39584 12.5898 7.1776C12.5311 7.16799 12.4731 7.16355 12.4172 7.16355C11.1989 7.16355 10.6615 9.33114 10.6615 9.33114C10.6615 9.33114 9.0863 13.4148 6.38031 16.206C3.67434 18.998 3.5346 21.2388 5.50675 24.2246C6.85185 26.2606 9.42666 26.8753 12.0653 26.8753C14.8021 26.8753 17.6077 26.2139 19.1799 25.793C19.2574 25.7723 28.8193 22.984 27.6081 20.6107C27.4046 20.212 27.0693 20.0522 26.6471 20.0522C24.9416 20.0522 21.8393 22.6726 20.5057 22.6726C20.2076 22.6726 19.9976 22.5416 19.9116 22.222C19.3433 20.1173 28.552 19.2325 27.7758 16.1839C27.639 15.6445 27.2677 15.4256 26.746 15.4263C24.4923 15.4263 19.4358 19.5181 18.3759 19.5181C18.2949 19.5181 18.2368 19.4937 18.2053 19.4419C17.6743 18.557 17.9653 17.9394 21.7082 15.6009C25.4511 13.2617 28.0783 11.8545 26.5841 10.1752C26.4121 9.98141 26.1684 9.8956 25.8725 9.8956C23.6001 9.89634 18.2311 14.9403 18.2311 14.9403C18.2311 14.9403 16.7821 16.496 15.9057 16.496C15.7043 16.496 15.533 16.4139 15.4169 16.2112C14.7956 15.1296 21.1879 10.1286 21.5484 8.06535C21.7928 6.66715 21.3771 5.95917 20.6081 5.95917Z" fill="#FF9D00"></path>
    <path d="M5.50686 24.2246C3.53472 21.2387 3.67446 18.9979 6.38043 16.206C9.08641 13.4147 10.6615 9.33111 10.6615 9.33111C10.6615 9.33111 11.2499 6.95933 12.59 7.17757C13.93 7.39581 14.9139 10.9401 12.1069 13.1084C9.29997 15.276 12.6659 16.7489 13.7459 14.713C14.8258 12.6772 17.7747 7.44316 19.304 6.44221C20.8326 5.44128 21.9089 6.00204 21.5484 8.06532C21.188 10.1286 14.795 15.1295 15.4171 16.2118C16.0391 17.2934 18.2312 14.9402 18.2312 14.9402C18.2312 14.9402 25.0907 8.49588 26.5842 10.1752C28.0776 11.8545 25.4512 13.2616 21.7082 15.6008C17.9646 17.9393 17.6744 18.557 18.2054 19.4418C18.7372 20.3266 26.9998 13.1351 27.7759 16.1838C28.5513 19.2324 19.3434 20.1173 19.9117 22.2219C20.48 24.3274 26.3979 18.2382 27.6082 20.6107C28.8193 22.9839 19.2574 25.7722 19.18 25.7929C16.0914 26.62 8.24723 28.3726 5.50686 24.2246Z" fill="#FFD21E"></path>
</svg>"""

loading_icon_html = """<svg id="share-btn-loading-icon" style="display:none;" class="animate-spin"
   style="color: #ffffff; 
"
   xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" fill="none" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24"><circle style="opacity: 0.25;" cx="12" cy="12" r="10" stroke="white" stroke-width="4"></circle><path style="opacity: 0.75;" fill="white" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>"""

share_js = """async () => {
	async function uploadFile(file){
		const UPLOAD_URL = 'https://huggingface.co/uploads';
		const response = await fetch(UPLOAD_URL, {
			method: 'POST',
			headers: {
				'Content-Type': file.type,
				'X-Requested-With': 'XMLHttpRequest',
			},
			body: file, /// <- File inherits from Blob
		});
		const url = await response.text();
		return url;
	}
    const gradioEl = document.querySelector('body > gradio-app');
    const imgEls = gradioEl.querySelectorAll('#generated-gallery img');
    const promptTxt = gradioEl.querySelector('#prompt_input input').value;
    const shareBtnEl = gradioEl.querySelector('#share-btn');
    const shareIconEl = gradioEl.querySelector('#share-btn-share-icon');
    const loadingIconEl = gradioEl.querySelector('#share-btn-loading-icon');
    if(!imgEls.length){
        return;
    };
    shareBtnEl.style.pointerEvents = 'none';
    shareIconEl.style.display = 'none';
    loadingIconEl.style.removeProperty('display');
    const files = await Promise.all(
        [...imgEls].map(async (imgEl) => {
            const res = await fetch(imgEl.src);
            const blob = await res.blob();
            const imgId = Date.now() % 200;
            const fileName = `diffuse-the-rest-${{imgId}}.png`;
            return new File([blob], fileName, { type: 'image/png' });
        })
    );
    const REGEX_CONCEPT = /<(((?!<.+>).)+)>/gm;
    const matches = [...promptTxt.matchAll(REGEX_CONCEPT)];
    const concepts = matches.map(m => m[1]);
    const conceptLibraryMd = concepts.map(c => `<a href='https://huggingface.co/sd-concepts-library/${c}' target='_blank'>${c}</a>`).join(`, `);
    const urls = await Promise.all(files.map((f) => uploadFile(f)));
	const htmlImgs = urls.map(url => `<img src='${url}' width='400' height='400'>`);
    const htmlImgsMd = htmlImgs.join(`\n`);
	const descriptionMd = `#### Prompt:
${promptTxt.replace(/</g, '\\\<').replace(/>/g, '\\\>')}
#### Concepts Used:
${conceptLibraryMd}
#### Generations:
<div style='display: flex; flex-wrap: wrap; column-gap: 0.75rem;'>
${htmlImgsMd}
</div>`;
    const params = new URLSearchParams({
        title: promptTxt,
        description: descriptionMd,
    });
	const paramsStr = params.toString();
	window.open(`https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer/discussions/new?${paramsStr}`, '_blank');
    shareBtnEl.style.removeProperty('pointer-events');
    shareIconEl.style.removeProperty('display');
    loadingIconEl.style.display = 'none';
}"""

api = HfApi()
models_list = api.list_models(author="sd-concepts-library", sort="likes", direction=-1)
models = []

# NEW LOGIN ATTEMPT {{{
api_key = os.environ['api_key']
my_token = api_key

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", revision="fp16", torch_dtype=torch.float16, use_auth_token=my_token).to("cuda")
# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=my_token).to("cuda")
# }}}

# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True, revision="fp16", torch_dtype=torch.float16).to("cuda")

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
  print("start while loop **************")
  while(num_added_tokens == 0):
    token = f"{token[:-1]}-{i}>"
    num_added_tokens = tokenizer.add_tokens(token)
    print("i --> ", i)
    print("token --> ", token)
    print("num_added_tokens --> ", num_added_tokens)
    i+=1
  print("end while loop **************")
  
  # resize the token embeddings
  text_encoder.resize_token_embeddings(len(tokenizer))
  
  # get the id for the token and assign the embeds
  token_id = tokenizer.convert_tokens_to_ids(token)
  print("&&&&&&&&&&&&&&&&")
  print("learned_embeds_path --> ", learned_embeds_path)
  print("text_encoder --> ", text_encoder)
  print("tokenizer --> ", tokenizer)
  print("_old_token --> ", _old_token)
  print("token --> ", token)
  print("trained_token --> ", trained_token)
  print("dtype --> ", dtype)
  print("num_added_tokens --> ", num_added_tokens)
  print("text_encoder --> ", text_encoder)
  print("token_id --> ", token_id)
  print("embeds --> ", embeds)
  print("&&&&&&&&&&&&&&&&")
  text_encoder.get_input_embeddings().weight.data[token_id] = embeds # <------ POINT OF FAILURE
  return token


ahx_model_list = [model for model in models_list if "ahx" in model.modelId]



# UNDER CONSTRUCTION ---------------------------------------------------------------
from time import sleep

print("--------------------------------------------------")
print("--------------------------------------------------")
print("Setting up the public library........")
print("--------------------------------------------------")
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
      # print("FAILURE: <-------------------------------------------------------------------")
      # print("model -->", model)
      # print("model_id -->", model_id)
      # print("CONTINUING - MODEL NOT LOADING")
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
    print("success / model loaded:")
    print("model -->", model)
    print("model_id -->", model_id)
  except: 
    print("FAILURE: <- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    print("model -->", model)
    print("model_id -->", model_id)
    print("CONTINUING - MODEL NOT LOADING")
    continue
  model_content["token"] = learned_token
  models.append(model_content)
  models.append(model_content)
  print("--------------------------------------------------")
  sleep(5)


print("--------------------------------------------------")
print("--------------------------------------------------")
print("--------------------------------------------------")
sleep(60)
  
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
              [text],
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
              [text],
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
# examples = ["a <cat-toy> in <madhubani-art> style", "a <line-art> style mecha robot", "a piano being played by <bonzi>", "Candid photo of <cheburashka>, high resolution photo, trending on artstation, interior design"]
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
  gr.HTML('''
  <div style="text-align: left; max-width: 720px; margin: 0 auto;">
              <br>
              <h1 style="font-weight: 900; margin-bottom: 7px;">
                🧑‍🚀 Astronaut Horse Concept Loader
              </h1>
              <br>
              <p _style="margin-bottom: 10px; font-size: 94%">
                Run your own text prompts into fine-tuned artist concepts, see the example below. Currently only loading two artist concepts while testing. Soon will automatically be able to add all concepts from astronaut horse artist collaborations. There's some buggy stuff here that'll be cleared up next week but I wanted to at least get this usable for the weekend!
              </p>
              <br>
              <a style="text-decoration:underline;" href="http://www.astronaut.horse">http://www.astronaut.horse</a>
              <br>
              <br>
              <h2 style="font-weight:500;">Prompt Examples Using Artist Token:</h2>
              <ul>
                <li>"a photograph of pink crystals in the style of &lt;artist&gt;"</li>
                <li>"a painting of a horse in the style of &lt;ivan-stripes&gt;"</li>
              </ul>
              <br>
              <h2 style="font-weight:500;">Currently-Usable Concept Tokens:</h2>
              <ul>
                <li>&lt;artist&gt;</li>
                <li>&lt;ivan-stripes&gt;</li>
              </ul>
            </div>
  ''')
  with gr.Row():
        # with gr.Column():
        #   gr.Markdown(f"")
        #   with gr.Row():
        #           image_blocks = []
        #           #for i, model in enumerate(models):
        #           with gr.Box().style(border=None):
        #             gr.HTML(assembleHTML(models))
        #               #title_block(model["token"], model["id"])
        #               #image_blocks.append(image_block(model["images"], model["concept_type"]))
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
                      btn = gr.Button("generate image",elem_id="run_btn").style(
                          margin=False,
                          rounded=(False, True, True, False),
                      )  
                  with gr.Row().style():
                      infer_outputs = gr.Gallery(show_label=False, elem_id="generated-gallery").style(grid=[2], height="512px")
                  with gr.Row():
                    gr.HTML("<p></p>")
                  with gr.Row():
                    gr.Examples(examples=examples, fn=infer_examples, inputs=[text], outputs=infer_outputs, cache_examples=True)
          with gr.Group(elem_id="share-btn-container"):
            community_icon = gr.HTML(community_icon_html, visible=False)
            loading_icon = gr.HTML(loading_icon_html, visible=False)
  checkbox_states = {}
  inputs = [text]
  btn.click(
        infer,
        inputs=inputs,
        outputs=[infer_outputs, community_icon, loading_icon]
    )
# after loading_icon on line 392.5
#             share_button = gr.Button("", elem_id="share-btn", visible=False)
# and update outputs=[...] on line 398 to match this
        # outputs=[infer_outputs, community_icon, loading_icon, share_button]
# then this has to be added after line 399
#   share_button.click(
#       None,
#       [],
#       [],
#       _js=share_js,
#   )
demo.queue(max_size=20).launch()