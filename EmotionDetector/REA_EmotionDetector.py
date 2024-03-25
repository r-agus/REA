import cv2
from deepface import DeepFace

class REA_EmotionDetector:
    """
    Clase para detectar emociones en tiempo real utilizando DeepFace y OpenCV, la clase esta preparada para que se imprima por pantalla la emoción detectada.

    Parámetros:
        capture_interval (int): El intervalo de tiempo en milisegundos entre cada captura de fotograma.

    Métodos:
        detectar_emociones(): Inicia la detección de emociones en tiempo real desde la cámara.
    """
    def __init__(self, capture_interval=10):
        """
        Inicializa la clase EmotionDetector.

        Parámetros:
            capture_interval (int): El intervalo de tiempo en milisegundos entre cada captura de fotograma.
        """
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0)
        self.capture_interval = capture_interval

    def detectar_emociones(self):
        """
        Inicia la detección de emociones en tiempo real desde la cámara.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                results = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                
                for result in results:
                    emotion = result['dominant_emotion']
                    print("Emotion:", emotion)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(self.capture_interval) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        
    def analizar_emocion_en_imagen(self, image_path):
        img = cv2.imread(image_path)
        results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        return results[0]['dominant_emotion'] if results else None
        
if __name__ == "__main__":
    # Uso de la clase
    detector = REA_EmotionDetector(capture_interval=10)
    detector.detectar_emociones()
    
    # emocion = detector.analizar_emocion_en_imagen("RUTA") # Especificar la ruta donde esta la imagen
    # if emocion:
    #     print("Emoción detectada en la imagen:", emocion)
    # else:
    #     print("No se pudo detectar ninguna emoción en la imagen.")
