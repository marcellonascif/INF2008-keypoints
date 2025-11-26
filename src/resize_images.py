import os
from glob import glob
from PIL import Image
from tqdm import tqdm

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

def resize_all_images():
    # Encontrar todas as imagens jpg recursivamente dentro de data/
    # O padrão **/*.jpg com recursive=True funciona no glob do python 3.10+
    # Mas para garantir compatibilidade e pegar tudo, vamos iterar pelos folds

    print(f"Procurando imagens em {DATA_DIR}...")
    images = glob(os.path.join(DATA_DIR, "**", "*.jpg"), recursive=True)

    print(f"Encontradas {len(images)} imagens. Redimensionando para {IMAGE_WIDTH}x{IMAGE_HEIGHT}...")

    for img_path in tqdm(images):
        try:
            with Image.open(img_path) as img:
                # Verifica se já está no tamanho certo para evitar reprocessamento desnecessário
                if img.size == (IMAGE_WIDTH, IMAGE_HEIGHT):
                    continue

                # Redimensiona (pode distorcer se o aspect ratio for diferente, mas é o solicitado)
                img_resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)

                # Salva sobrescrevendo a original
                img_resized.save(img_path)
        except Exception as e:
            print(f"Erro ao processar {img_path}: {e}")

    print("Redimensionamento concluído.")

if __name__ == "__main__":
    resize_all_images()
