# Ejemplo de la implementación de la clase SpeechRecognition
from speech_recognition import SpeechRecognition

sr = SpeechRecognition()

# Se puede lanzar la interfaz
#sr.crear_interfaz()

# o se puede hacer sin interfaz, con el uso de las funciones
sr.grabar_audio("audio.mp3", 5)
sr.transcribir("audio.mp3", "transcripcion.txt")

