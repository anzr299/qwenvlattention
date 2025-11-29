# ============================================================
# FULL OPTION-B IMPLEMENTATION
# Compare hidden_state_text[L] vs hidden_state_image[L]
# ============================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import cv2
import re

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

print(f"Loading model {MODEL_NAME} on {DEVICE}...")

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
dtype = torch.float16 if DEVICE in ("cuda","mps") else torch.float32

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    trust_remote_code=True,
)
model.to(DEVICE)
model.eval()


# ------------------------------------------------------------
# IMAGE
# ------------------------------------------------------------
IMAGE_PATH = "water.png"

def load_image(path):
    img = Image.open(path).convert("RGB")
    return img

raw_image = load_image(IMAGE_PATH)
raw_image = raw_image.resize((200,300))


# ------------------------------------------------------------
# BUILD CHAT INPUTS
# ------------------------------------------------------------
def build_chat_inputs(image, prompt):
    messages = [{
        "role":"user",
        "content":[
            {"type":"image", "image":image},
            {"type":"text", "text":prompt},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    inputs.pop("token_type_ids", None)
    return inputs.to(DEVICE)


# ------------------------------------------------------------
# CAPTION + HIDDEN STATES
# ------------------------------------------------------------
def generate_caption_and_all_hidden_states(image, prompt, max_new_tokens=20):

    # PASS 1 -- generate caption
    base_inputs = build_chat_inputs(image, prompt)
    prompt_len = base_inputs["input_ids"].shape[1]

    with torch.no_grad():
        sequences = model.generate(
            **base_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )

    full_ids = sequences[0]

    decoded = processor.tokenizer.decode(
        full_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    m = re.search(r"assistant\s*(.*)", decoded, re.DOTALL|re.IGNORECASE)
    caption_text = m.group(1).strip() if m else decoded.strip()

    # PASS 2 -- hidden states
    full_ids_batch = full_ids.unsqueeze(0).to(DEVICE)
    attention_mask = torch.ones_like(full_ids_batch)

    img_inputs = processor.image_processor(image, return_tensors="pt")
    pixel_values = img_inputs["pixel_values"].to(DEVICE)
    image_grid_thw = img_inputs["image_grid_thw"].to(DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=full_ids_batch,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            use_cache=False
        )

    all_hidden_states = outputs.hidden_states

    return caption_text, full_ids, all_hidden_states, pixel_values, image_grid_thw, prompt_len


caption, full_ids, all_hidden_states, pixel_values, image_grid_thw, prompt_len = \
    generate_caption_and_all_hidden_states(
        raw_image,
        prompt="Describe the sequence of events briefly at most 10 words.",
        max_new_tokens=40
    )

print("CAPTION:", caption)


# ------------------------------------------------------------
# PATCH EMBEDDINGS (for grid size only)
# ------------------------------------------------------------
def get_image_patch_embeddings(pixel_values, image_grid_thw):
    image_embs_list, deepstack = model.get_image_features(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw
    )
    image_embs = image_embs_list[0]

    # compute patch grid size
    T,H,W = image_grid_thw[0].tolist()
    spatial = model.visual.spatial_merge_size

    H_patches = H // spatial
    W_patches = W // spatial

    return image_embs, H_patches, W_patches


image_embs, H_patches, W_patches = get_image_patch_embeddings(
    pixel_values, image_grid_thw
)

num_image_tokens = image_embs.shape[0]   # this is 70 in your output


# ------------------------------------------------------------
# OPTION B SIMILARITY: h_text[L] vs h_image[L]
# ------------------------------------------------------------
def compute_similarity_optionB(
    hidden_states_layer,
    num_image_tokens,
    full_ids,
    prompt_len,
    processor
):
    """
    Compare text-token hidden states with image-token hidden states
    AT THE SAME TRANSFORMER LAYER.

    Correct handling of multimodal sequence offsets.
    """

    hidden = hidden_states_layer[0]  # [seq_len, hidden_dim]

    # Split
    image_states = hidden[:num_image_tokens]        # [N_img, d]
    text_states  = hidden[num_image_tokens:]        # [N_text, d]

    # Number of text-only prompt tokens
    num_prompt_text_tokens = prompt_len - num_image_tokens

    # Generated text tokens (IDs)
    gen_token_ids = full_ids[prompt_len:]           # correct ids

    # Decode tokens
    tokens = [
        processor.tokenizer.decode([tid], skip_special_tokens=True)
        for tid in gen_token_ids.tolist()
    ]
    tokens = [t for t in tokens if t.strip()]

    # Normalize image vectors
    img_norm = image_states / (image_states.norm(dim=-1, keepdim=True) + 1e-6)

    # Similarity vectors
    sim_vectors = []
    for i in range(len(tokens)):
        # Correct index inside text_states
        t_state = text_states[num_prompt_text_tokens + i]
        t_norm = t_state / (t_state.norm(dim=-1, keepdim=True) + 1e-6)

        sim = torch.matmul(img_norm, t_norm)
        sim_vectors.append(sim.detach().cpu().numpy())

    return sim_vectors, tokens


# ------------------------------------------------------------
# VISUALIZATION HELPERS
# ------------------------------------------------------------
def normalize_grid(x):
    x = x - x.min()
    m = x.max()
    return x / m if m>0 else x

def overlay_heatmap(image, grid, alpha=0.5, show_grid=True, H_patches=None, W_patches=None):
    img = np.array(image.convert("RGB"))
    H,W,_ = img.shape

    grid = normalize_grid(grid).astype(np.float32)
    heat = cv2.resize(grid, (W,H), interpolation=cv2.INTER_CUBIC)
    heat = normalize_grid(heat)

    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heat)[:,:,:3]

    blended = (1-alpha)*(img/255.0) + alpha*heat_rgb
    blended = (blended*255).astype(np.uint8)

    if show_grid:
        ph = H / H_patches
        pw = W / W_patches
        for i in range(1,H_patches):
            y = int(i*ph)
            cv2.line(blended,(0,y),(W,y),(255,255,255),1)
        for j in range(1,W_patches):
            x = int(j*pw)
            cv2.line(blended,(x,0),(x,H),(255,255,255),1)

    return Image.fromarray(blended)

def draw_pretty_caption_wrapped(fig, caption_tokens, current_idx, max_width_px=None):
    renderer = fig.canvas.get_renderer()
    fontsize = 18
    space_px = 12

    fig_width_px = fig.get_figwidth()*fig.dpi
    if max_width_px is None:
        max_width_px = fig_width_px*0.90

    measured=[]
    for i,tok in enumerate(caption_tokens):
        txt=fig.text(0,0,tok,fontsize=fontsize)
        bb = txt.get_window_extent(renderer=renderer)
        txt.remove()
        measured.append((tok,bb.width))

    lines=[]
    current_line=[]
    current_width=0
    for i,(tok,w) in enumerate(measured):
        extra = w if not current_line else (space_px+w)
        if current_width+extra > max_width_px:
            lines.append(current_line)
            current_line=[(tok,w,i)]
            current_width=w
        else:
            current_line.append((tok,w,i))
            current_width+=extra
    if current_line:
        lines.append(current_line)

    fig.patches.extend([
        plt.Rectangle(
            (0,0.88-0.05*len(lines)),
            1.0,
            0.12+0.05*len(lines),
            transform=fig.transFigure,
            color="white",
            zorder=-1,
        )
    ])

    y0=0.94
    ls=0.05
    for li,line in enumerate(lines):
        total_w = sum(w for _,w,_ in line) + space_px*(len(line)-1)
        total_fig_w = total_w/fig_width_px
        x=0.5-total_fig_w/2
        y=y0 - li*ls

        for tok,w,idx in line:
            col = "red" if idx==current_idx else "gray"
            wgt = "bold" if idx==current_idx else "regular"
            fig.text(
                x,y,tok,
                fontsize=fontsize,color=col,weight=wgt,
                ha="left",va="center"
            )
            x+=(w+space_px)/fig_width_px


def build_frames(image, tokens, sim_vectors, H_patches, W_patches, out_dir="frames", alpha=0.5):
    os.makedirs(out_dir, exist_ok=True)
    frames=[]
    for i in range(len(tokens)):
        vec = sim_vectors[i]
        grid = vec.reshape(H_patches, W_patches)

        frame_img = overlay_heatmap(image, grid, alpha=alpha, show_grid=True,
                                    H_patches=H_patches, W_patches=W_patches)

        fig,ax = plt.subplots(figsize=(6,7))
        fig.subplots_adjust(top=0.82)
        ax.imshow(frame_img)
        ax.axis("off")

        draw_pretty_caption_wrapped(fig, tokens, i)

        fname=f"{out_dir}/frame_{i:02d}.png"
        plt.savefig(fname,dpi=120,pad_inches=0.2)
        plt.close(fig)
        frames.append(fname)
    return frames

def make_gif(frames, out="attention.gif", duration=0.7):
    imgs=[imageio.imread(f) for f in frames]
    imageio.mimsave(out,imgs,duration=duration)
    print("GIF saved:", out)


# ------------------------------------------------------------
# MAIN: LAYER-BY-LAYER VISUALIZATION
# ------------------------------------------------------------
def generate_all_layer_gifs(
    image,
    all_hidden_states,
    full_ids,
    prompt_len,
    num_image_tokens,
    H_patches,
    W_patches,
    base_dir="layer_outputs",
    image_name="image"
):

    num_layers = len(all_hidden_states)
    print("Generating", num_layers, "layers...")

    os.makedirs(base_dir, exist_ok=True)

    for layer_idx in range(num_layers):
        print(f"Layer {layer_idx}/{num_layers-1}")

        hidden_states_layer = all_hidden_states[layer_idx]

        sim_vectors, tokens = compute_similarity_optionB(
            hidden_states_layer,
            num_image_tokens=num_image_tokens,
            full_ids=full_ids,
            prompt_len=prompt_len,
            processor=processor
        )

        layer_dir = os.path.join(base_dir,f"layer_{layer_idx:02d}")
        frames_dir = os.path.join(layer_dir,"frames")

        frames = build_frames(
            image=image,
            tokens=tokens,
            sim_vectors=sim_vectors,
            H_patches=H_patches,
            W_patches=W_patches,
            out_dir=frames_dir
        )

        gif_path = os.path.join(layer_dir, f"{image_name}_layer_{layer_idx:02d}_attention.gif")
        make_gif(frames, gif_path, duration=0.7)

    print("DONE.")
def generate_token_evolution_gifs(
    image,
    all_hidden_states,
    full_ids,
    prompt_len,
    num_image_tokens,
    H_patches,
    W_patches,
    out_dir="token_evolution",
    image_name="image",
    alpha=0.5,
):
    """
    Token evolution using the corrected Option-B multimodal indexing:
    Compare text-token hidden states with image-token hidden states
    at each layer, and track how the similarity changes ACROSS layers.
    """

    num_layers = len(all_hidden_states)
    print(f"\n{'='*70}")
    print(f"GENERATING TOKEN EVOLUTION ACROSS {num_layers} LAYERS")
    print(f"{'='*70}\n")

    # Compute number of prompt TEXT tokens
    num_prompt_text_tokens = prompt_len - num_image_tokens

    # Decode generated text tokens once
    gen_token_ids = full_ids[prompt_len:]
    tokens = [
        processor.tokenizer.decode([tid], skip_special_tokens=True)
        for tid in gen_token_ids.tolist()
    ]
    tokens = [t for t in tokens if t.strip()]
    num_tokens = len(tokens)

    print(f"Found {num_tokens} generated tokens.")

    # Each entry: token_layer_sims[token_idx][layer_idx] = vec of shape [N_img]
    token_layer_sims = [
        [] for _ in range(num_tokens)
    ]

    # -----------------------------------------------------------
    # COMPUTE SIMILARITY FOR ALL LAYERS (same logic as Option-B)
    # -----------------------------------------------------------
    print("Computing layer-wise similarities...")

    for layer_idx in range(num_layers):
        hidden = all_hidden_states[layer_idx][0]  # [seq_len, dim]

        image_states = hidden[:num_image_tokens]        # [N_img, d]
        text_states  = hidden[num_image_tokens:]        # [N_text, d]

        # Normalize image tokens once per layer
        img_norm = image_states / (image_states.norm(dim=-1, keepdim=True) + 1e-6)

        # For each generated token
        for i in range(num_tokens):
            # Corrected indexing inside text_states
            t_state = text_states[num_prompt_text_tokens + i]
            t_norm = t_state / (t_state.norm(dim=-1, keepdim=True) + 1e-6)

            sim = torch.matmul(img_norm, t_norm)  # [N_img]
            token_layer_sims[i].append(sim.detach().cpu().numpy())

    # -----------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # -----------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------------------------------------
    # VISUALIZE EACH TOKEN'S EVOLUTION ACROSS LAYERS
    # -----------------------------------------------------------
    print("Creating GIFs...")

    for token_idx in range(num_tokens):
        token_text = tokens[token_idx]
        layer_progression = token_layer_sims[token_idx]

        temp_frames = []

        for layer_idx in range(num_layers):
            sim_vec = layer_progression[layer_idx]
            grid = sim_vec.reshape(H_patches, W_patches)

            # frame_img = overlay_heatmap(image, grid, alpha=alpha)
            frame_img = overlay_heatmap(
                image,
                grid,
                alpha=alpha,
                show_grid=True,
                H_patches=H_patches,
                W_patches=W_patches
            )

            fig, ax = plt.subplots(figsize=(6, 7))
            fig.subplots_adjust(top=0.82)
            ax.imshow(frame_img)
            ax.axis("off")

            fig.text(
                0.5, 0.95,
                f"Token: '{token_text}'",
                fontsize=20,
                fontweight='bold',
                color='red',
                ha='center',
                va='center'
            )
            fig.text(
                0.5, 0.90,
                f"Layer {layer_idx}/{num_layers-1}",
                fontsize=16,
                color='gray',
                ha='center',
                va='center'
            )

            temp_path = f"/tmp/token{token_idx}_layer{layer_idx}.png"
            plt.savefig(temp_path, dpi=120, pad_inches=0.2, bbox_inches='tight')
            plt.close(fig)

            temp_frames.append(temp_path)

        # Save GIF
        safe_token = "".join(c for c in token_text.replace(" ", "_") if c.isalnum() or c in "_-")
        gif_path = os.path.join(out_dir, f"token_{token_idx:02d}_{safe_token}_layer_evolution.gif")

        imgs = [imageio.imread(f) for f in temp_frames]
        imageio.mimsave(gif_path, imgs, duration=0.3)

        for f in temp_frames:
            if os.path.exists(f):
                os.remove(f)

        print(f"Saved {gif_path}")

    print("\nTOKEN EVOLUTION COMPLETE.")


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
# 1. Layer-by-layer GIFs (Option B)
generate_all_layer_gifs(
    image=raw_image,
    all_hidden_states=all_hidden_states,
    full_ids=full_ids,
    prompt_len=prompt_len,
    num_image_tokens=num_image_tokens,
    H_patches=H_patches,
    W_patches=W_patches,
    base_dir="layer_outputs_optionB",
    image_name=IMAGE_PATH.replace(".png","").replace(".jpg","")
)

# 2. Token evolution GIFs (Option B)
generate_token_evolution_gifs(
    image=raw_image,
    all_hidden_states=all_hidden_states,
    full_ids=full_ids,
    prompt_len=prompt_len,
    num_image_tokens=num_image_tokens,
    H_patches=H_patches,
    W_patches=W_patches,
    out_dir="token_evolution_optionB",
    image_name=IMAGE_PATH.replace(".png","").replace(".jpg",""),
    alpha=0.5
)
print("Option B processing completed.")
