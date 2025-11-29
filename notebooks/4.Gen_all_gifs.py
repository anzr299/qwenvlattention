# %%
"""
=============================================================================
VISION-LANGUAGE MODEL ATTENTION VISUALIZATION
=============================================================================

This script analyzes how a Vision-Language Model (Qwen3-VL) attends to 
different image regions when generating text descriptions.

KEY CONCEPT:
- The model processes an image as a grid of patches (like dividing a photo 
  into a grid of smaller squares)
- When generating each word, the model's internal representation has some 
  "similarity" to each image patch
- Higher similarity = the model is "paying more attention" to that patch
- We visualize this as a heatmap overlay on the original image

WORKFLOW:
1. Load model and image
2. Generate caption + extract hidden states from ALL layers
3. For each layer, compute similarity between text tokens and image patches
4. Visualize as animated heatmaps showing attention evolution
"""

import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import cv2
import re

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# %%
# =============================================================================
# DEVICE SETUP
# =============================================================================

def get_device():
    """
    Selects the best available compute device.
    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
    
    Returns:
        str: Device identifier ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()

# %%
# =============================================================================
# MODEL LOADING
# =============================================================================

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"  # 2B parameter vision-language model

print(f"Loading model {MODEL_NAME} on {DEVICE}...")

# Processor: handles tokenization (text) and image preprocessing
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Choose precision based on device
# float16: faster, less memory (GPU/MPS) 
# float32: more precise, required for CPU
dtype = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32

# Load the actual model weights
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    trust_remote_code=True,
)
model.to(DEVICE)
model.eval()  # Set to evaluation mode (disables dropout, etc.)

# %%
# =============================================================================
# IMAGE LOADING AND PREPROCESSING
# =============================================================================

IMAGE_PATH = "water.png"  # <-- Change to your image path

def load_image(path: str) -> Image.Image:
    """
    Load image and convert to RGB.
    
    Args:
        path: File path to image
        
    Returns:
        PIL Image in RGB format
    """
    img = Image.open(path).convert("RGB")
    return img


raw_image = load_image(IMAGE_PATH)
print(f"Original image size: {raw_image.size}")  # (width, height)

# Resize for faster processing (optional)
raw_image = raw_image.resize((200, 300))  # (width, height)
print(f"Resized to: {raw_image.size}")

# %%
# =============================================================================
# INPUT PREPARATION
# =============================================================================

def build_chat_inputs(image, prompt):
    """
    Prepares inputs in the chat format expected by the model.
    
    Args:
        image: PIL Image
        prompt: Text prompt/question about the image
        
    Returns:
        Dict with tensors: {input_ids, attention_mask, pixel_values, image_grid_thw}
        
    STRUCTURE:
    - input_ids: Token IDs for text [batch=1, seq_len]
    - attention_mask: Which tokens to attend to [batch=1, seq_len]
    - pixel_values: **Image patch embeddings** (NOT raw pixels!) 
                    [batch=1, num_patches, embed_dim] e.g., [1, 280, 1536]
                    These are visual tokens after vision encoder processing
    - image_grid_thw: Grid dimensions [batch=1, 3] -> (temporal, height, width)
    
    NOTE: The name "pixel_values" is misleading - it's actually pre-computed
    image embeddings that serve as visual tokens for the language model.
    """
    # Chat format: [{'role': 'user', 'content': [image, text]}]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Apply chat template and tokenize
    # This adds special tokens like <|im_start|>, <|im_end|>, etc.
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # Adds "assistant:" prompt
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)  # Not needed for this model
    return inputs.to(DEVICE)


# %%
# =============================================================================
# CAPTION GENERATION + HIDDEN STATE EXTRACTION
# =============================================================================

def generate_caption_and_all_hidden_states(image, prompt, max_new_tokens=20):
    """
    Two-pass approach to get both caption and internal representations:
    
    PASS 1: Generate caption text
    - Model autoregressively generates tokens
    - Output: sequence of token IDs
    
    PASS 2: Extract hidden states
    - Run model again on FULL sequence (prompt + generated tokens)
    - This time, request hidden states from ALL layers
    - Output: internal representations at each layer
    
    WHY TWO PASSES?
    - Generation mode doesn't return hidden states efficiently
    - We need the complete sequence first, then analyze it
    
    Args:
        image: PIL Image
        prompt: Text prompt
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        caption_text: Generated description (str)
        full_ids: Complete token sequence [seq_len] 
        all_hidden_states: Tuple of tensors, one per layer
                          Each tensor: [batch=1, seq_len, hidden_dim=896]
        pixel_values: Image tensor [1, channels, H, W]
        image_grid_thw: Grid dimensions [1, 3]
        prompt_len: Length of prompt (to separate prompt from generation)
    """
    
    # -------------------------------------------------------------------------
    # PASS 1: GENERATE CAPTION
    # -------------------------------------------------------------------------
    base_inputs = build_chat_inputs(image, prompt)
    prompt_len = base_inputs["input_ids"].shape[1]  # Length before generation
    
    print(f"Prompt length: {prompt_len} tokens")

    # Generate new tokens autoregressively
    with torch.no_grad():
        sequences = model.generate(
            **base_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding (deterministic)
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    # sequences shape: [batch=1, total_seq_len]

    full_ids = sequences[0]  # Remove batch dimension -> [total_seq_len]
    print(f"Generated sequence length: {full_ids.shape[0]} tokens")

    # Decode to human-readable text
    decoded = processor.tokenizer.decode(
        full_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    
    # Extract only the assistant's response (after "assistant")
    m = re.search(r"assistant\s*(.*)", decoded, flags=re.DOTALL | re.IGNORECASE)
    caption_text = m.group(1).strip() if m else decoded.strip()
    
    print(f"Generated caption: '{caption_text}'")

    # -------------------------------------------------------------------------
    # PASS 2: EXTRACT HIDDEN STATES
    # -------------------------------------------------------------------------
    # Prepare full sequence for forward pass
    full_ids_batch = full_ids.unsqueeze(0).to(DEVICE)  # [1, seq_len]
    attention_mask = torch.ones_like(full_ids_batch)   # [1, seq_len] - attend to all
    
    # Preprocess image (same as in generation)
    img_inputs = processor.image_processor(image, return_tensors="pt")
    # IMPORTANT: Despite the name "pixel_values", this is NOT raw pixels!
    # It's actually image patch embeddings after vision encoder processing
    # Shape: [1, num_patches, embedding_dim] e.g., [1, 280, 1536]
    # Where: 280 = number of visual tokens, 1536 = embedding dimension
    pixel_values = img_inputs["pixel_values"].to(DEVICE)       # [1, num_patches, embed_dim]
    image_grid_thw = img_inputs["image_grid_thw"].to(DEVICE)   # [1, 3] -> (T, H, W)

    # Forward pass with hidden states output
    with torch.no_grad():
        outputs = model(
            input_ids=full_ids_batch,           # [1, seq_len]
            attention_mask=attention_mask,       # [1, seq_len]
            pixel_values=pixel_values,           # [1, C, H, W]
            image_grid_thw=image_grid_thw,       # [1, 3]
            output_hidden_states=True,           # KEY: request all layer outputs
            use_cache=False,                     # Don't need KV cache
        )

    # all_hidden_states: tuple of (num_layers + 1) tensors
    # - Index 0: embedding layer output
    # - Index 1 to num_layers: transformer layer outputs
    # Each tensor shape: [batch=1, seq_len, hidden_dim=896]
    all_hidden_states = outputs.hidden_states
    
    print(f"Extracted hidden states from {len(all_hidden_states)} layers")
    print(f"Each layer output shape: {all_hidden_states[0].shape}")
    
    return caption_text, full_ids, all_hidden_states, pixel_values, image_grid_thw, prompt_len


# Run the two-pass process
caption, full_ids, all_hidden_states, pixel_values, image_grid_thw, prompt_len = (
    generate_caption_and_all_hidden_states(
        raw_image,
        prompt="Describe the sequence of events briefly at most 10 words.",
        max_new_tokens=40,
    )
)

print(f"\n{'='*70}")
print(f"CAPTION: {caption}")
print(f"{'='*70}\n")

# %%
# =============================================================================
# IMAGE PATCH EMBEDDING EXTRACTION
# =============================================================================

def get_image_patch_embeddings(pixel_values, image_grid_thw):
    """
    Extract image embeddings from the vision encoder.
    
    IMPORTANT CLARIFICATION:
    Despite the parameter name "pixel_values", this is NOT raw pixel data!
    It's already processed image patch embeddings from the vision encoder.
    
    The vision encoder has ALREADY:
    1. Divided image into patches (e.g., 16x16 pixels per patch)
    2. Processed through vision transformer
    3. Applied spatial merging (combining neighboring patches)
    4. Produced embedding vectors
    
    What we get: A set of visual tokens ready for the language model
    Shape: [1, num_patches, embedding_dim] e.g., [1, 280, 1536]
    - 280 = number of visual tokens (after patch division & spatial merge)
    - 1536 = embedding dimension (Qwen3-VL's visual token space)
    
    This function extracts these embeddings to compute attention.
    
    Args:
        pixel_values: Image patch embeddings [1, num_patches, embed_dim]
                     (naming is historical/legacy from transformers library)
        image_grid_thw: Grid dimensions [1, 3] -> (T=1, H, W)
        
    Returns:
        image_embs: Patch embeddings [num_patches, hidden_dim]
                   Each row = one visual token
        H_patches: Number of patch rows in the grid
        W_patches: Number of patch columns in the grid
        
    EXAMPLE:
    - Input: [1, 280, 1536] visual tokens
    - Grid: 14 x 20 patches (before merge)
    - Spatial merge: 2x2 -> final grid 7 x 10 = 70 patches
    - Output: [70, 896] embeddings (projected to language model's hidden dim)
    """
    
    # Get image features from vision encoder
    # image_embs_list: list of embedding tensors (one per image in batch)
    image_embs_list, deepstack = model.get_image_features(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )

    # Extract single image (batch size = 1)
    image_embs = image_embs_list[0]  # [num_patches, hidden_dim=896]
    print(f"Image patch embeddings shape: {image_embs.shape}")

    # Decode grid dimensions
    T, H, W = image_grid_thw[0].tolist()  # Temporal (=1 for static image), Height, Width
    spatial = model.visual.spatial_merge_size  # How many patches merged together (typically 2)

    # Calculate patch grid dimensions after spatial merging
    H_patches = H // spatial  # e.g., 14 / 2 = 7 rows
    W_patches = W // spatial  # e.g., 20 / 2 = 10 columns

    expected = H_patches * W_patches
    print(f"Patch grid: {H_patches} x {W_patches} = {expected} patches")
    print(f"Spatial merge size: {spatial}x{spatial}")
    
    assert expected == image_embs.shape[0], f"Patch count mismatch: expected {expected}, got {image_embs.shape[0]}"

    return image_embs, H_patches, W_patches


image_embs, H_patches, W_patches = get_image_patch_embeddings(
    pixel_values, image_grid_thw
)

# %%
# =============================================================================
# SIMILARITY COMPUTATION
# =============================================================================

def compute_similarity(hidden_states_layer, full_ids, prompt_len, image_embs):
    """
    Compute cosine similarity between text token representations and image patches.
    
    INTUITION:
    - Each generated token has an internal representation (hidden state)
    - Each image patch has a representation (embedding)
    - Cosine similarity measures how "related" they are
    - High similarity = token is "attending to" that image region
    
    MATHEMATICAL FORMULATION:
    - Text representation: h_t ∈ R^d (hidden state at position t)
    - Image patch: p_i ∈ R^d (patch embedding i)
    - Similarity: sim(h_t, p_i) = (h_t · p_i) / (||h_t|| * ||p_i||)
    - Result: similarity matrix [num_tokens, num_patches]
    
    Args:
        hidden_states_layer: Hidden states from one layer [1, seq_len, hidden_dim]
        full_ids: Token IDs [seq_len]
        prompt_len: Where generation starts (int)
        image_embs: Image patch embeddings [num_patches, hidden_dim]
        
    Returns:
        sim_vectors: List of similarity vectors, one per generated token
                    Each vector: [num_patches] with values in [-1, 1]
        tokens: List of decoded token strings
        
    EXAMPLE:
    - Generated 5 tokens: ["A", "dog", "running", "in", "park"]
    - Image has 70 patches (7x10 grid)
    - Output: 5 vectors, each with 70 similarity scores
    """
    
    # Remove batch dimension
    hidden_states_layer = hidden_states_layer[0]  # [seq_len, hidden_dim]
    total_len = hidden_states_layer.shape[0]
    
    print(f"\nComputing similarity for layer...")
    print(f"Hidden states shape: {hidden_states_layer.shape}")
    print(f"Image embeddings shape: {image_embs.shape}")

    # Extract only the GENERATED tokens (exclude prompt)
    # Rationale: We want to see how generated words relate to image
    gen_positions = list(range(prompt_len, total_len))
    gen_token_ids = full_ids[prompt_len:]  # [num_generated]
    
    print(f"Analyzing {len(gen_positions)} generated tokens")

    # Decode token IDs to human-readable strings
    tokens = [
        processor.tokenizer.decode([tid], skip_special_tokens=True)
        for tid in gen_token_ids.tolist()
    ]
    tokens = [t for t in tokens if t.strip()]  # Remove empty tokens

    # -------------------------------------------------------------------------
    # NORMALIZE IMAGE EMBEDDINGS (for cosine similarity)
    # -------------------------------------------------------------------------
    # Normalization: v_norm = v / ||v||
    # After normalization, dot product = cosine similarity
    img_norm = image_embs / (image_embs.norm(dim=-1, keepdim=True) + 1e-6)
    # img_norm shape: [num_patches, hidden_dim]
    # Each row is a unit vector

    # -------------------------------------------------------------------------
    # COMPUTE SIMILARITY FOR EACH GENERATED TOKEN
    # -------------------------------------------------------------------------
    sim_vectors = []
    for idx, pos in enumerate(gen_positions):
        # Get hidden state for this token
        token_vec = hidden_states_layer[pos]  # [hidden_dim]
        
        # Normalize token vector
        token_norm = token_vec / (token_vec.norm(dim=-1, keepdim=True) + 1e-6)
        # token_norm shape: [hidden_dim]
        
        # Compute cosine similarity with ALL image patches
        # Matrix multiplication: [num_patches, hidden_dim] @ [hidden_dim] 
        #                      -> [num_patches]
        sim = torch.matmul(img_norm, token_norm)  # [num_patches]
        
        # Convert to numpy for visualization
        sim_vectors.append(sim.detach().cpu().numpy())
        
        if idx < 3:  # Print first few for debugging
            print(f"  Token '{tokens[idx]}': similarity range [{sim.min():.3f}, {sim.max():.3f}]")

    # Match token count (in case of truncation)
    if len(sim_vectors) > len(tokens):
        sim_vectors = sim_vectors[:len(tokens)]

    print(f"Generated {len(sim_vectors)} similarity vectors")
    return sim_vectors, tokens


# %%
# =============================================================================
# VISUALIZATION HELPER FUNCTIONS
# =============================================================================
# (Comments omitted for brevity as requested to focus on model operations)

def normalize_grid(x):
    """Normalize values to [0, 1] range."""
    x = x - x.min()
    m = x.max()
    return x / m if m > 0 else x


def overlay_heatmap(image, grid, alpha=0.5):
    """Overlay similarity heatmap on image."""
    img = np.array(image.convert("RGB"))
    H, W, _ = img.shape

    grid = normalize_grid(grid)
    grid = np.ascontiguousarray(grid.astype(np.float32))

    heat = cv2.resize(grid, (W, H), interpolation=cv2.INTER_CUBIC)
    heat = normalize_grid(heat)

    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heat)[:, :, :3]

    blended = (1-alpha) * (img/255.0) + alpha * heat_rgb
    blended = (blended*255).astype(np.uint8)
    return Image.fromarray(blended)


def draw_pretty_caption_wrapped(fig, caption_tokens, current_idx, max_width_px=None):
    """Draw wrapped caption with current token highlighted."""
    renderer = fig.canvas.get_renderer()
    fontsize = 18
    space_px = 12

    fig_width_px = fig.get_figwidth() * fig.dpi
    if max_width_px is None:
        max_width_px = fig_width_px * 0.90

    measured = []
    for i, tok in enumerate(caption_tokens):
        txt = fig.text(0, 0, tok, fontsize=fontsize)
        bb = txt.get_window_extent(renderer=renderer)
        txt.remove()
        measured.append((tok, bb.width))

    lines = []
    current_line = []
    current_width = 0

    for i, (tok, w) in enumerate(measured):
        extra = w if not current_line else (space_px + w)
        if current_width + extra > max_width_px:
            lines.append(current_line)
            current_line = [(tok, w, i)]
            current_width = w
        else:
            current_line.append((tok, w, i))
            current_width += extra

    if current_line:
        lines.append(current_line)

    fig.patches.extend([
        plt.Rectangle(
            (0, 0.88 - 0.05*len(lines)),
            1.0,
            0.12 + 0.05*len(lines),
            transform=fig.transFigure,
            color="white",
            zorder=-1,
        )
    ])

    y_start = 0.94
    line_spacing = 0.05

    for li, line in enumerate(lines):
        total_w = sum(w for _, w, _ in line) + space_px * (len(line) - 1)
        total_fig_w = total_w / fig_width_px

        x = 0.5 - total_fig_w / 2
        y = y_start - li * line_spacing

        for tok, w, token_index in line:
            color = "red" if token_index == current_idx else "gray"
            weight = "bold" if token_index == current_idx else "regular"

            fig.text(
                x,
                y,
                tok,
                fontsize=fontsize,
                color=color,
                weight=weight,
                ha="left",
                va="center",
            )
            x += (w + space_px) / fig_width_px


def build_frames(image, tokens, sim_vectors, H_patches, W_patches, out_dir="frames", alpha=0.5):
    """Generate frame images for animation."""
    os.makedirs(out_dir, exist_ok=True)
    frames = []

    for i in range(len(tokens)):
        vec = sim_vectors[i]
        grid = vec.reshape(H_patches, W_patches)

        frame_img = overlay_heatmap(image, grid, alpha=alpha)

        fig, ax = plt.subplots(figsize=(6, 7))
        fig.subplots_adjust(top=0.82)

        ax.imshow(frame_img)
        ax.axis("off")

        draw_pretty_caption_wrapped(fig, tokens, i)

        fname = f"{out_dir}/frame_{i:02d}.png"
        plt.savefig(fname, dpi=120, pad_inches=0.2)
        plt.close(fig)

        frames.append(fname)

    return frames


def make_gif(frames, out="attention.gif", duration=0.7):
    """Combine frames into animated GIF."""
    imgs = [imageio.imread(f) for f in frames]
    imageio.mimsave(out, imgs, duration=duration)
    print(f"  GIF saved: {out}")


# %%
# =============================================================================
# MAIN PROCESSING: GENERATE GIFS FOR ALL LAYERS
# =============================================================================

def generate_all_layer_gifs(
    image,
    all_hidden_states,
    full_ids,
    prompt_len,
    image_embs,
    H_patches,
    W_patches,
    base_dir="layer_outputs",
    image_name="image"
):
    """
    Generate attention visualization GIFs for every model layer.
    
    RATIONALE:
    - Different layers capture different levels of abstraction
    - Early layers: low-level features (edges, colors, textures)
    - Middle layers: object parts, patterns
    - Late layers: high-level semantics, relationships
    
    By visualizing all layers, we can observe how attention evolves through
    the network depth as the model processes information.
    
    Args:
        image: Original PIL image
        all_hidden_states: Tuple of hidden state tensors from all layers
                          Each: [1, seq_len, hidden_dim]
        full_ids: Complete token sequence [seq_len]
        prompt_len: Length of prompt portion
        image_embs: Image patch embeddings [num_patches, hidden_dim]
        H_patches: Number of patch rows
        W_patches: Number of patch columns
        base_dir: Output directory
        image_name: Prefix for output files
        
    OUTPUT STRUCTURE:
        layer_outputs/
        ├── layer_00/
        │   ├── frames/
        │   │   ├── frame_00.png
        │   │   └── ...
        │   └── image_layer_00_attention.gif
        ├── layer_01/
        │   └── ...
        └── ...
    """
    
    num_layers = len(all_hidden_states)
    print(f"\n{'='*70}")
    print(f"GENERATING VISUALIZATIONS FOR {num_layers} LAYERS")
    print(f"{'='*70}\n")
    
    os.makedirs(base_dir, exist_ok=True)
    
    for layer_idx in range(num_layers):
        print(f"\n[Layer {layer_idx}/{num_layers-1}]")
        
        # -------------------------------------------------------------------------
        # STEP 1: GET HIDDEN STATES FOR THIS LAYER
        # -------------------------------------------------------------------------
        hidden_states_layer = all_hidden_states[layer_idx]
        # Shape: [1, seq_len, hidden_dim=896]
        
        # -------------------------------------------------------------------------
        # STEP 2: COMPUTE SIMILARITY BETWEEN TOKENS AND IMAGE PATCHES
        # -------------------------------------------------------------------------
        # Returns:
        # - sim_vectors: List of [num_patches] arrays, one per token
        # - tokens: List of token strings
        sim_vectors, tokens = compute_similarity(
            hidden_states_layer=hidden_states_layer,
            full_ids=full_ids,
            prompt_len=prompt_len,
            image_embs=image_embs,
        )
        
        # -------------------------------------------------------------------------
        # STEP 3: CREATE OUTPUT DIRECTORIES
        # -------------------------------------------------------------------------
        layer_dir = os.path.join(base_dir, f"layer_{layer_idx:02d}")
        frames_dir = os.path.join(layer_dir, "frames")
        
        # -------------------------------------------------------------------------
        # STEP 4: GENERATE VISUALIZATION FRAMES
        # -------------------------------------------------------------------------
        # Each frame shows:
        # - Image with heatmap overlay (where model is "looking")
        # - Caption with current token highlighted
        frames = build_frames(
            image=image,
            tokens=tokens,
            sim_vectors=sim_vectors,
            H_patches=H_patches,
            W_patches=W_patches,
            out_dir=frames_dir,
            alpha=0.5  # Transparency of heatmap
        )
        
        # -------------------------------------------------------------------------
        # STEP 5: CREATE ANIMATED GIF
        # -------------------------------------------------------------------------
        gif_path = os.path.join(layer_dir, f"{image_name}_layer_{layer_idx:02d}_attention.gif")
        make_gif(frames, out=gif_path, duration=0.7)
        
        print(f"  ✓ Complete")
    
    print(f"\n{'='*70}")
    print(f"ALL LAYERS PROCESSED")
    print(f"Results saved in: {base_dir}/")
    print(f"{'='*70}\n")


# %%
# RUN THE COMPLETE PIPELINE
# =============================================================================

generate_all_layer_gifs(
    image=raw_image,
    all_hidden_states=all_hidden_states,
    full_ids=full_ids,
    prompt_len=prompt_len,
    image_embs=image_embs,
    H_patches=H_patches,
    W_patches=W_patches,
    base_dir="layer_outputs",
    image_name=IMAGE_PATH.replace(".png", "").replace(".jpg", "")
)