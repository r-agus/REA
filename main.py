# main.py
import sys
sys.path.insert(0, './Face_Recognition')
sys.path.insert(0, './EmotionDetector')
sys.path.insert(0, './DeepFloyd')
sys.path.insert(0, './Whisper')

from face_reco_class import FaceRecognizer
from REA_EmotionDetector import REA_EmotionDetector

fr = FaceRecognizer()
ed = REA_EmotionDetector()

image = "unknown.jpg"

#fr.save_person(image, "Teo")    # Uncomment to save teo

if fr.is_teo(image):
    # Teo has been recognized. Detect emotion 
    print(ed.analizar_emocion_en_imagen(image))
else:

    print(fr.recognize_face(image))