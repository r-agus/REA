# main.py
import sys
sys.path.insert(0, './Face_Recognition')

from face_reco_class import FaceRecognizer

fr = FaceRecognizer()
print(fr.recognize_face("unknown.jpg"))
