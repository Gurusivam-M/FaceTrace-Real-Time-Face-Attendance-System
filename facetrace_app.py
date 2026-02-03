import os
import cv2
import csv
import datetime
import tkinter.messagebox as messagebox
import customtkinter as ctk
from PIL import Image
import numpy as np

# Ensure necessary folders exist
if not os.path.exists("faces"):
    os.makedirs("faces")

# Main Application Class
class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FaceTrace - Face Recognition App")
        self.geometry("500x400")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        ctk.CTkLabel(self, text="FaceTrace", font=("Arial", 24)).pack(pady=20)
        ctk.CTkButton(self, text="Register New Face", command=self.register_face).pack(pady=10)
        ctk.CTkButton(self, text="Detect Face (Real-time)", command=self.detect_face).pack(pady=10)
        ctk.CTkButton(self, text="Recognize & Mark Attendance", command=self.recognize_faces).pack(pady=10)
        ctk.CTkButton(self, text="Exit", command=self.quit).pack(pady=10)

    def register_face(self):
        name = ctk.CTkInputDialog(text="Enter Name for Face Registration:", title="Register Face").get_input()
        if not name:
            return messagebox.showwarning("Input Error", "Name cannot be empty.")

        person_path = os.path.join("faces", name)
        os.makedirs(person_path, exist_ok=True)

        cap = cv2.VideoCapture(0)
        count = 0
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while count < 50:
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                count += 1
                face = gray[y:y+h, x:x+w]
                cv2.imwrite(f"{person_path}/{name}_{count}.jpg", face)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Capturing {count}/50", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Registering Face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Success", f"{count} images saved for {name}")

    def detect_face(self):
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Real-Time Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def recognize_faces(self):
        if not os.path.exists("trained_model.yml"):
            return messagebox.showwarning("Train Model", "Train the model first using train_faces.py")

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("trained_model.yml")

        # Load label map
        label_map = {}
        with open("label_map.txt", "r") as f:
            for line in f:
                key, value = line.strip().split(":")
                label_map[int(key)] = value

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)
        attendance = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                face_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                if confidence < 60:
                    name = label_map.get(face_id, "Unknown")
                    attendance.add(name)
                    label = f"{name} ({round(100 - confidence)}%)"
                else:
                    label = "Unknown"

                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("FaceTrace - Recognition (Press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Save attendance
        if attendance:
            with open("attendance.csv", "a", newline='') as f:
                writer = csv.writer(f)
                for name in attendance:
                    writer.writerow([name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

            messagebox.showinfo("Attendance", f"Marked attendance for: {', '.join(attendance)}")

# Start the app
app = MainApp()
app.mainloop()

