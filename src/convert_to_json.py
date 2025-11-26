import os
import json
from glob import glob
from datetime import datetime


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def yolo_line_to_bbox_and_kps(parts, img_w, img_h):
	"""Converte uma linha no formato YOLO para bbox e keypoints.

	Recebe o formato:
		class_id cx cy w h k1x k1y vis1 k2x k2y vis2

	Retorna:
		(class_id, [x1, y1, x2, y2], [[k1x, k1y, vis1], [k2x, k2y, vis2]])
	"""

	class_id = int(parts[0])

	cx = float(parts[1]) * img_w
	cy = float(parts[2]) * img_h
	bw = float(parts[3]) * img_w
	bh = float(parts[4]) * img_h

	x1 = cx - bw / 2.0
	y1 = cy - bh / 2.0
	x2 = cx + bw / 2.0
	y2 = cy + bh / 2.0

	k1x = float(parts[5]) * img_w
	k1y = float(parts[6]) * img_h
	vis1 = int(parts[7])

	k2x = float(parts[8]) * img_w
	k2y = float(parts[9]) * img_h
	vis2 = int(parts[10])

	bbox = [round(x1), round(y1), round(x2), round(y2)]
	keypoints = [
		[round(k1x), round(k1y), vis1],  # head
		[round(k2x), round(k2y), vis2],  # tail
	]

	return class_id, bbox, keypoints


def process_labels_folder(labels_dir, out_labels_dir, img_w, img_h):
	"""Lê todos os .txt de um diretório de labels e gera .json.

	Saída por arquivo:
		{"bboxes": [[x1, y1, x2, y2], ...], "keypoints": [[[hx, hy, vis], [tx, ty, vis]], ...]}
	"""

	os.makedirs(out_labels_dir, exist_ok=True)
	label_files = sorted(glob(os.path.join(labels_dir, "*.txt")))

	for label_path in label_files:
		with open(label_path, "r") as f:
			lines = [ln.strip() for ln in f.readlines() if ln.strip()]

		bboxes = []
		keypoints_all = []

		for line in lines:
			parts = line.split()

			if len(parts) != 11:
				print(f"Ignorando linha com formato inesperado em {label_path}: {line}")
				continue

			_, bbox, kps = yolo_line_to_bbox_and_kps(parts, img_w, img_h)
			bboxes.append(bbox)
			keypoints_all.append(kps)

		data = {"bboxes": bboxes, "keypoints": keypoints_all}

		base_name = os.path.splitext(os.path.basename(label_path))[0]
		json_path = os.path.join(out_labels_dir, base_name + ".json")
		with open(json_path, "w") as jf:
			json.dump(data, jf)

		print(f"Salvo: {json_path}")


def main():
	from shutil import copy2

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	dataset_name = f"dataset_rcnn_{timestamp}"
	out_dataset_dir = os.path.join(ROOT_DIR, dataset_name)
	os.makedirs(out_dataset_dir, exist_ok=True)

	for fold_name in os.listdir(DATA_DIR):
		fold_path = os.path.join(DATA_DIR, fold_name)
		if not os.path.isdir(fold_path):
			continue

		# destino deste fold no novo dataset
		out_fold_dir = os.path.join(out_dataset_dir, fold_name)
		os.makedirs(out_fold_dir, exist_ok=True)

		# copia os CSVs de vídeos (se existirem)
		for csv_name in ("train_videos.csv", "test_videos.csv"):
			csv_src = os.path.join(fold_path, csv_name)
			if os.path.isfile(csv_src):
				csv_dst = os.path.join(out_fold_dir, csv_name)
				copy2(csv_src, csv_dst)
				print(f"Copiado CSV: {csv_src} -> {csv_dst}")

		# processa subpastas de treino e teste
		for split in ("treino", "teste"):
			split_dir = os.path.join(fold_path, split)
			labels_dir = os.path.join(split_dir, "labels")
			images_dir = os.path.join(split_dir, "images")

			if not os.path.isdir(labels_dir):
				continue

			print(f"Processando {split} de {fold_name}: {labels_dir}")

			out_split_dir = os.path.join(out_fold_dir, split)
			out_labels_dir = os.path.join(out_split_dir, "labels")
			os.makedirs(out_split_dir, exist_ok=True)

			# copia imagens (se houver) para manter dataset completo
			if os.path.isdir(images_dir):
				out_images_dir = os.path.join(out_split_dir, "images")
				os.makedirs(out_images_dir, exist_ok=True)
				for img_name in os.listdir(images_dir):
					src_img = os.path.join(images_dir, img_name)
					if os.path.isfile(src_img):
						copy2(src_img, os.path.join(out_images_dir, img_name))

			# gera os arquivos json a partir dos labels originais
			process_labels_folder(labels_dir, out_labels_dir, IMAGE_WIDTH, IMAGE_HEIGHT)

if __name__ == "__main__":
	main()
