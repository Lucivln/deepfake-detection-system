import os
import cv2
from mtcnn import MTCNN
from tqdm import tqdm

detector = MTCNN()

INPUT_REAL = "data/processed/real_frames"
INPUT_FAKE = "data/processed/fake_frames"

OUTPUT_REAL = "data/processed/real_crops"
OUTPUT_FAKE = "data/processed/fake_crops"

os.makedirs(OUTPUT_REAL, exist_ok=True)
os.makedirs(OUTPUT_FAKE, exist_ok=True)


def extract_face(img_path, output_dir, prefix):
    img = cv2.imread(img_path)
    if img is None:
        return

    results = detector.detect_faces(img)

    if len(results) > 0:
        best_face = max(results, key=lambda x: x['box'][2] * x['box'][3])
        x, y, w, h = best_face['box']

        h_img, w_img, _ = img.shape

        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)

        face = img[y:y+h, x:x+w]

        if face.size == 0:
            return

        face = cv2.resize(face, (224, 224))

        filename = f"{prefix}_{os.path.basename(img_path)}"
        cv2.imwrite(os.path.join(output_dir, filename), face)

# -------- REAL --------
for img_name in tqdm(os.listdir(INPUT_REAL), desc="Real faces"):
    img_path = os.path.join(INPUT_REAL, img_name)
    extract_face(img_path, OUTPUT_REAL, "real")

# -------- FAKE --------
for img_name in tqdm(os.listdir(INPUT_FAKE), desc="Fake faces"):
    img_path = os.path.join(INPUT_FAKE, img_name)
    extract_face(img_path, OUTPUT_FAKE, "fake")

print("Face extraction done!")