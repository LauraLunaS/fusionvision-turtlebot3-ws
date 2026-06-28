import os
import numpy as np
from PIL import Image
from transformers import pipeline

# ── Configs ──────────────────────────────────────────────────────────────
INPUT_DIR     = "test_data/dataset/test/images"
OUTPUT_DIR    = "test_data/depth"
COLORIZED_DIR = "test_data/depth_colorized"

REALSENSE_WIDTH  = 640
REALSENSE_HEIGHT = 480
MAX_DEPTH_MM     = 3000
DEPTH_SCALE      = 1000.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COLORIZED_DIR, exist_ok=True)


print("Carregando Depth Anything V2 Small (CPU)...")
depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
)
print("Modelo carregado.\n")


images = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if not images:
    print(f"Nenhuma imagem encontrada em '{INPUT_DIR}'.")
    exit(1)

print(f"Processando {len(images)} imagem(ns)...\n")

for i, fname in enumerate(images, 1):
    stem     = os.path.splitext(fname)[0]
    out_path = os.path.join(OUTPUT_DIR, f"{stem}_depth.png")

    if os.path.exists(out_path):
        print(f"  [{i}/{len(images)}] {fname} já processada, pulando.")
        continue

    print(f"  [{i}/{len(images)}] {fname}...", end=" ", flush=True)

    rgb_img = Image.open(os.path.join(INPUT_DIR, fname)).convert("RGB")
    rgb_img = rgb_img.resize((REALSENSE_WIDTH, REALSENSE_HEIGHT), Image.LANCZOS)

    result    = depth_pipe(rgb_img)
    depth_raw = np.array(result["depth"])

    depth_norm = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min() + 1e-8)
    depth_mm   = ((1.0 - depth_norm) * MAX_DEPTH_MM).astype(np.uint16)

    Image.fromarray(depth_mm, mode="I;16").save(out_path)

    depth_8bit = (depth_norm * 255).astype(np.uint8)
    Image.fromarray(depth_8bit).save(os.path.join(COLORIZED_DIR, f"{stem}_depth_color.png"))

    print(f"OK")

print(f"\nConcluído! {len(images)} depth maps em '{OUTPUT_DIR}'")
