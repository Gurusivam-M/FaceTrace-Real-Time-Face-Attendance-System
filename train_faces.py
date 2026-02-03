# train_faces.py

import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
path = 'faces'

def get_images_and_labels(path):
    face_samples = []
    ids = []
    label_map = {}
    current_id = 0

    for person_name in os.listdir(path):
        person_path = os.path.join(path, person_name)
        if not os.path.isdir(person_path): continue

        label_map[current_id] = person_name

        for image in os.listdir(person_path):
            img_path = os.path.join(person_path, image)
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            try:
                pil_img = Image.open(img_path).convert('L')  # Convert to grayscale
                img_np = np.array(pil_img, 'uint8')
                faces = face_cascade.detectMultiScale(img_np)

                for (x, y, w, h) in faces:
                    face_samples.append(img_np[y:y+h, x:x+w])
                    ids.append(current_id)
            except Exception as e:
                print(f"Could not process {img_path}: {e}")

        current_id += 1

    return face_samples, ids, label_map

faces, ids, label_map = get_images_and_labels(path)

if len(faces) == 0:
    print("[ERROR] No faces found. Training aborted.")
else:
    recognizer.train(faces, np.array(ids))
    recognizer.write('trained_model.yml')

    # Save label map
    with open("label_map.txt", "w") as f:
        for k, v in label_map.items():
            f.write(f"{k}:{v}\n")

    print("[INFO] Training completed and model saved.")
