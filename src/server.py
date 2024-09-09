import os
import socket
import cv2
import pickle
import struct
import threading
import time
from deepface import DeepFace
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject

HOST = socket.gethostbyname(socket.gethostname())
PORT = 9999

class SignalEmitter(QObject):
    update_ui = pyqtSignal(object, object)

class ServerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepFace Server")
        self.setGeometry(100, 100, 1200, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        main_layout.addWidget(self.video_label)

        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        self.age_label = QLabel("Age: ")
        self.gender_label = QLabel("Gender: ")
        self.emotion_label = QLabel("Emotion: ")
        self.face_confidence_label = QLabel("Face Confidence: ")
        
        analysis_layout.addWidget(self.age_label)
        analysis_layout.addWidget(self.gender_label)
        analysis_layout.addWidget(self.emotion_label)
        analysis_layout.addWidget(self.face_confidence_label)
        analysis_layout.addStretch()

        main_layout.addWidget(analysis_widget)

        self.signal_emitter = SignalEmitter()
        self.signal_emitter.update_ui.connect(self.update_ui)

    def update_ui(self, frame, analysis_result):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)

        if isinstance(analysis_result, list) and len(analysis_result) > 0:
            result = analysis_result[0]
        else:
            result = analysis_result

        self.age_label.setText(f"Age: {result.get('age', 'N/A')}")
        self.gender_label.setText(f"Gender: {result.get('dominant_gender', 'N/A')} "
                                  f"({result.get('gender', {}).get(result.get('dominant_gender', ''), 'N/A'):.2f}%)")
        emotions = result.get('emotion', {})
        dominant_emotion = max(emotions, key=emotions.get) if emotions else 'N/A'
        self.emotion_label.setText(f"Emotion: {dominant_emotion} ({emotions.get(dominant_emotion, 'N/A'):.2f}%)")
        self.face_confidence_label.setText(f"Face Confidence: {result.get('face_confidence', 'N/A'):.2f}")

def analyze_frame(frame):
    try:
        result = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)
        return result
    except Exception as e:
        print('Error analyzing image:', e)
        return []

# def analyze_frame(frame):
#     try:
#         result = DeepFace.analyze(frame,
#                                   actions=['age', 'gender', 'emotion'],
#                                   enforce_detection=False,
#                                   detector_backend='retinaface',
#                                   age_model='age_real',
#                                   gender_model='gender_vgg',
#                                   emotion_model='emotion_ferplus')
#         return result
#     except Exception as e:
#         print('Error analyzing image:', e)
#         return []

def server_thread(signal_emitter):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"Listening at: {HOST}:{PORT}")

    client_socket, addr = server_socket.accept()
    print(f"Connected to: {addr}")

    vid = cv2.VideoCapture(0)

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        img_serialize = pickle.dumps(frame)
        message = struct.pack("Q", len(img_serialize)) + img_serialize
        client_socket.sendall(message)

        analysis_result = analyze_frame(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        signal_emitter.update_ui.emit(rgb_frame, analysis_result)

        if cv2.waitKey(10) == 13:
            break

    vid.release()
    client_socket.close()

if __name__ == "__main__":
    app = QApplication([])
    server_ui = ServerUI()
    server_ui.show()

    server_thread = threading.Thread(target=server_thread, args=(server_ui.signal_emitter,))
    server_thread.start()

    app.exec_()
    server_thread.join()