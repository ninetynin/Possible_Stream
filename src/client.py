import os
import socket
import cv2
import pickle
import struct
import threading
import gc
from deepface import DeepFace
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSignal, QObject
from logger import get_logger, get_analysis_logger
import sqlite3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

HOST = '10.90.0.177'
PORT = 9999

logger = get_logger(__name__)
analysis_logger = get_analysis_logger()

class SignalEmitter(QObject):
    update_ui = pyqtSignal(object, object)

class ClientUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepFace Client")
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

def decompress_frame(data):
    frame = pickle.loads(data)
    return frame

def analyze_frame(frame):
    try:
        result = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)
        analysis_logger.debug(f"Analysis result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return []

def client_thread(signal_emitter):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    logger.info(f"Connected to server: {HOST}:{PORT}")

    data_buffer = b""
    payload_size = struct.calcsize("Q")

    try:
        while True:
            while len(data_buffer) < payload_size:
                packet = client_socket.recv(4 * 1024)
                if not packet:
                    break
                data_buffer += packet

            packed_msg_size = data_buffer[:payload_size]
            data_buffer = data_buffer[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            while len(data_buffer) < msg_size:
                data_buffer += client_socket.recv(4 * 1024)

            frame_data = data_buffer[:msg_size]
            data_buffer = data_buffer[msg_size:]

            frame = decompress_frame(frame_data)

            analysis_result = analyze_frame(frame)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            signal_emitter.update_ui.emit(rgb_frame, analysis_result)

            gc.collect()

    except ConnectionResetError as e:
        logger.error(f"Connection reset: {e}")
    except Exception as e:
        logger.error(f"Error in client thread: {e}")
    finally:
        client_socket.close()
        logger.info("Client disconnected.")

if __name__ == "__main__":
    app = QApplication([])
    client_ui = ClientUI()
    client_ui.show()

    client_thread_instance = threading.Thread(target=client_thread, args=(client_ui.signal_emitter,))
    client_thread_instance.start()

    app.exec_()
    client_thread_instance.join()