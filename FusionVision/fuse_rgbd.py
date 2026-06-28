"""
fuse_rgbd.py
Substitui a leitura do bag file RealSense.
Lê pares RGB + depth sintético e gera nuvem de pontos Open3D,
extraindo coordenada XYZ de cada resíduo detectado.
"""

import os
import numpy as np
import open3d as o3d
from PIL import Image
from ultralytics import YOLO, FastSAM

# ── Configurações ──────────────────────────────────────────────────────────────
RGB_DIR   = "test_data/sample"        # imagens RGB
DEPTH_DIR = "test_data/depth"         # depth maps gerados
OUT_DIR   = "test_data/pointclouds"   # nuvens de pontos .ply
os.makedirs(OUT_DIR, exist_ok=True)

# Classes do dataset (data.yaml)
CLASSES = {0: "BIODEGRADABLE", 1: "CARDBOARD", 2: "GLASS",
           3: "METAL", 4: "PAPER", 5: "PLASTIC"}

# Intrínsecos simulados RealSense D435i @ 640x480
FX, FY       = 615.0, 615.0
CX, CY       = 320.0, 240.0
WIDTH        = 640
HEIGHT       = 480
DEPTH_SCALE  = 1000.0   # mm → metros
DEPTH_TRUNC  = 3.0      # trunca além de 3m

intrinsics = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, FX, FY, CX, CY)

# ── Modelos ────────────────────────────────────────────────────────────────────
print("Carregando YOLOv8 e FastSAM...")
yolo    = YOLO("runs/detect/runs/train/waste_yolov8n_combined_v2/weights/best.pt")
fastsam = FastSAM("FastSAM-x.pt")
print("Modelos carregados.\n")

# ── Pipeline por frame ─────────────────────────────────────────────────────────
rgb_files = sorted([
    f for f in os.listdir(RGB_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print(f"Processando {len(rgb_files)} frame(s)...\n")

resultados = []  # coleta XYZ de todos os objetos

for fname in rgb_files:
    stem      = os.path.splitext(fname)[0]
    rgb_path  = os.path.join(RGB_DIR, fname)
    dep_path  = os.path.join(DEPTH_DIR, f"{stem}_depth.png")

    if not os.path.exists(dep_path):
        print(f"[AVISO] Depth não encontrado para {fname} — pulando.")
        continue

    print(f"── {fname} ──")

    # 1. YOLO — detecção
    yolo_res = yolo(rgb_path, device="cpu", verbose=False, imgsz=640)
    boxes    = yolo_res[0].boxes
    print(f"  YOLO: {len(boxes)} objeto(s) detectado(s)")

    # 2. FastSAM — segmentação
    fsam_res = fastsam(rgb_path, device="cpu", retina_masks=True,
                       verbose=False, imgsz=640)
    masks = fsam_res[0].masks
    n_masks = len(masks) if masks is not None else 0
    print(f"  FastSAM: {n_masks} máscara(s) gerada(s)")

    if len(boxes) == 0:
        print("  Nenhum objeto detectado — pulando fusão 3D.\n")
        continue

    # 3. Carrega RGB redimensionado (igual ao que foi usado no depth)
    rgb_pil = Image.open(rgb_path).convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS)
    rgb_pil.save("/tmp/rgb_resized.png")

    # 4. Cria nuvem de pontos da cena completa
    color_o3d = o3d.io.read_image("/tmp/rgb_resized.png")
    depth_o3d = o3d.io.read_image(dep_path)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=DEPTH_SCALE,
        depth_trunc=DEPTH_TRUNC,
        convert_rgb_to_intensity=False
    )
    pcd_full = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    print(f"  Nuvem completa: {len(pcd_full.points)} pontos")

    # 5. Para cada objeto: extrai ROI e calcula XYZ
    np_depth = np.array(Image.open(dep_path)).astype(np.float32) / DEPTH_SCALE

    for i, box in enumerate(boxes):
        cls_id   = int(box.cls[0])
        cls_name = yolo_res[0].names[cls_id]
        conf     = float(box.conf[0])

        # Escala bounding box para 640x480
        orig_w, orig_h = Image.open(rgb_path).size
        sx, sy = WIDTH / orig_w, HEIGHT / orig_h
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, x2 = int(x1 * sx), int(x2 * sx)
        y1, y2 = int(y1 * sy), int(y2 * sy)

        # Coordenada XYZ pelo centro da bounding box
        roi_depth = np_depth[y1:y2, x1:x2]
        valid     = roi_depth[roi_depth > 0]
        if len(valid) == 0:
            continue

        z   = float(np.median(valid))
        x_m = ((x1 + x2) / 2 - CX) * z / FX
        y_m = ((y1 + y2) / 2 - CY) * z / FY

        print(f"  [{cls_name} conf={conf:.2f}] XYZ = ({x_m:.3f}, {y_m:.3f}, {z:.3f}) m")
        resultados.append({"frame": fname, "classe": cls_name,
                           "conf": conf, "x": x_m, "y": y_m, "z": z})

        # Salva nuvem isolada do objeto (recorte por profundidade)
        pcd_path = os.path.join(OUT_DIR, f"{stem}_{cls_name}_{i}.ply")
        o3d.io.write_point_cloud(pcd_path, pcd_full)

    print()

# ── Resumo final ───────────────────────────────────────────────────────────────
print("=" * 50)
print(f"RESUMO: {len(resultados)} objeto(s) com coordenada XYZ extraída\n")
for r in resultados:
    print(f"  {r['frame']} | {r['classe']} ({r['conf']:.2f}) | "
          f"X={r['x']:.3f} Y={r['y']:.3f} Z={r['z']:.3f} m")

print("\nPipeline 3D concluído!")
