import os
import json
import shutil
from glob import glob

import matplotlib.pyplot as plt
from PIL import Image

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def yolo_line_to_bbox_and_kps(parts):
    cls_id = int(parts[0])

    cx = float(parts[1]) * IMAGE_WIDTH
    cy = float(parts[2]) * IMAGE_HEIGHT
    bw = float(parts[3]) * IMAGE_WIDTH
    bh = float(parts[4]) * IMAGE_HEIGHT

    x1 = round(cx - bw / 2.0)
    y1 = round(cy - bh / 2.0)
    x2 = round(cx + bw / 2.0)
    y2 = round(cy + bh / 2.0)

    k1x = round(float(parts[5]) * IMAGE_WIDTH)
    k1y = round(float(parts[6]) * IMAGE_HEIGHT)
    vis1 = int(parts[7])

    k2x = round(float(parts[8]) * IMAGE_WIDTH)
    k2y = round(float(parts[9]) * IMAGE_HEIGHT)
    vis2 = int(parts[10])

    bbox = [x1, y1, x2, y2]
    keypoints = [[k1x, k1y, vis1], [k2x, k2y, vis2]]

    return cls_id, bbox, keypoints


def plot_comparison(image_path, txt_path, json_path, save_path, max_objs=3):
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)

    with open(txt_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    with open(json_path, "r") as f:
        data = json.load(f)

    bboxes_json = data["bboxes"]
    kps_json = data["keypoints"]

    n = min(len(lines), len(bboxes_json), max_objs)
    colors = ["r", "g", "b"]

    for i in range(n):
        parts = lines[i].split()
        _, bbox_txt, kps_txt = yolo_line_to_bbox_and_kps(parts)
        bbox_j = bboxes_json[i]
        kps_j = kps_json[i]
        color = colors[i % len(colors)]

        # bbox do txt (linha contínua)
        x1, y1, x2, y2 = bbox_txt
        w = x2 - x1
        h = y2 - y1
        rect = plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)

        # bbox do json (linha tracejada do mesmo objeto) - PRETO e com transparência
        x1j, y1j, x2j, y2j = bbox_j
        wj = x2j - x1j
        hj = y2j - y1j
        rect_j = plt.Rectangle((x1j, y1j), wj, hj, fill=False, edgecolor="black", linewidth=2, linestyle="--", alpha=0.6)
        ax.add_patch(rect_j)

        # keypoints txt (círculos cheios)
        for (kx, ky, vis) in kps_txt:
            ax.scatter(kx, ky, c=color, s=30)

        # keypoints json (x marcando) - PRETO e com transparência
        for (kx, ky, vis) in kps_j:
            ax.scatter(kx, ky, c="black", s=50, marker="x", alpha=0.6)

    ax.set_title(os.path.basename(image_path))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # descobre o dataset_rcnn_* mais recente
    candidates = sorted(
        [d for d in os.listdir(ROOT_DIR) if d.startswith("dataset_rcnn_")]
    )
    if not candidates:
        print("Nenhum dataset_rcnn_* encontrado na raiz.")
        return

    output_dir = os.path.join(ROOT_DIR, "tmp-comparison")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    latest = candidates[-1]
    dataset_dir = os.path.join(ROOT_DIR, latest)
    print("Usando dataset:", dataset_dir)

    shown_videos = set()

    for fold_name in os.listdir(DATA_DIR):
        if len(shown_videos) >= 5:
            break

        fold_path = os.path.join(DATA_DIR, fold_name)
        if not os.path.isdir(fold_path):
            continue

        txt_labels_dir = os.path.join(fold_path, "treino", "labels")
        images_dir = os.path.join(fold_path, "treino", "images")
        if not os.path.isdir(txt_labels_dir) or not os.path.isdir(images_dir):
            continue

        out_labels_dir = os.path.join(dataset_dir, fold_name, "treino", "labels")
        out_images_dir = os.path.join(dataset_dir, fold_name, "treino", "images")
        if not os.path.isdir(out_labels_dir) or not os.path.isdir(out_images_dir):
            continue

        txt_files = sorted(glob(os.path.join(txt_labels_dir, "*.txt")))

        for txt_path in txt_files:
            if len(shown_videos) >= 5:
                break

            base = os.path.splitext(os.path.basename(txt_path))[0]
            # Extrai o ID do vídeo (ex: v025_f242 -> v025)
            video_id = base.split("_")[0]

            if video_id in shown_videos:
                continue

            json_path = os.path.join(out_labels_dir, base + ".json")
            img_path = os.path.join(out_images_dir, base + ".jpg")
            if not os.path.exists(json_path) or not os.path.exists(img_path):
                continue

            print(f"Visualizando {txt_path} ↔ {json_path}")
            save_path = os.path.join(output_dir, f"plot_{video_id}.png")
            plot_comparison(img_path, txt_path, json_path, save_path)
            print(f"Salvo em {save_path}")
            shown_videos.add(video_id)


if __name__ == "__main__":
    main()
