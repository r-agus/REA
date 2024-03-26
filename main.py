# main.py
import sys
sys.path.insert(0, './Face_Recognition')
sys.path.insert(0, './EmotionDetector')
sys.path.insert(0, './DeepFloyd')
sys.path.insert(0, './Whisper')

from face_reco_class import FaceRecognizer
from REA_EmotionDetector import REA_EmotionDetector

fr = FaceRecognizer()
print(fr.recognize_face("unknown.jpg"))
