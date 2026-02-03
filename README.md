🎯 FaceTrace – Real-Time Face Attendance System

FaceTrace is a real-time face recognition–based attendance management system that automates the traditional attendance process using a live camera feed. It leverages computer vision techniques to detect, recognize, and record attendance accurately and efficiently.

🚀 Features

📸 Real-time face detection using webcam

🧑 Face registration with multiple image samples

🤖 Face recognition using LBPH algorithm

🕒 Automatic attendance marking with date & time

📁 Attendance stored securely in CSV format

🖥️ User-friendly GUI built with CustomTkinter

🛠️ Technologies Used

Programming Language: Python

Libraries & Frameworks:

OpenCV

NumPy

Pillow

CustomTkinter

Algorithms:

Haar Cascade (Face Detection)

LBPH – Local Binary Patterns Histograms (Face Recognition)

📂 Project Structure
FaceTrace/
│
├── faces/                 # Stored face images
├── attendance.csv         # Attendance records
├── trained_model.yml      # Trained face recognition model
├── label_map.txt          # Mapping of labels to names
├── main.py                # Main application file
├── train_faces.py         # Face training script
└── README.md              # Project documentation

⚙️ Installation

Clone the repository:

git clone https://github.com/your-username/FaceTrace.git


Install required dependencies:

pip install opencv-python customtkinter pillow numpy


Run the application:

python main.py

🧪 How It Works

Register a new face by capturing multiple samples

Train the model using stored face images

Detect and recognize faces in real time

Automatically mark attendance with timestamp

🎓 Use Cases

Educational institutions

Offices and organizations

Secure attendance management systems

🔮 Future Enhancements

Database integration (MySQL / SQLite)

Cloud-based attendance storage

Face mask detection

Mobile or web-based interface

👨‍💻 Author

Gurusivam
BE Computer Science Engineering
📌 Face Recognition | Computer Vision | Python
