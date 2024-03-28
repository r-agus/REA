# main.py
import sys
sys.path.insert(0, './Face_Recognition')
sys.path.insert(0, './EmotionDetector')
sys.path.insert(0, './DeepFloyd')
sys.path.insert(0, './Whisper')

from Face_Recognition.face_reco_class import FaceRecognizer
from EmotionDetector.REA_EmotionDetector import REA_EmotionDetector
from Whisper.speech_recognition import SpeechRecognition


sr = SpeechRecognition()
fr = FaceRecognizer()
ed = REA_EmotionDetector()

image = "unknown.jpg"

#fr.save_person(image, "Teo")    # Uncomment to save teo

# Se puede lanzar la interfaz
#sr.crear_interfaz()

if fr.is_teo(image):
    # Teo has been recognized. Detect emotion 
    print(ed.analizar_emocion_en_imagen(image))
else:
    print(fr.recognize_face(image))
    
#TODO EMOTION IMAGE GENERATOR TO ADULT


#TRANSCRIPT AUDIO FROM ADULT
sr.transcribir("audio.mp3", "transcripcion.txt")

    
#TODO LAST STEP:TEXT TO IMAGE GENERATOR.
#run from colab or kaggle
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
	"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

prompt = "pictogram happy emoji"
print(f"prompt: {prompt}")
image = pipeline(
	prompt
).images[0]

image
