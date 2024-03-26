import cv2
import dlib
import imutils
cap = cv2.VideoCapture("camara.mp4") #aqui se introduce el nombre del video que se quiera reconocer
#Si quisieras capturar de un streaming, seria: VideoCapture(0, cv2.CAP_DSHOW)

# Detector facial
face_detector = dlib.get_frontal_face_detector()
# Predictor de 68 puntos de referencia
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
     ret, frame = cap.read()
     if ret == False:
          break
     frame = imutils.resize(frame, width=1080)
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     coordinates_bboxes = face_detector(gray, 1)
     #print("coordinates_bboxes:", coordinates_bboxes)
     for c in coordinates_bboxes:
          x_ini, y_ini, x_fin, y_fin = c.left(), c.top(), c.right(), c.bottom()
          cv2.rectangle(frame, (x_ini, y_ini), (x_fin, y_fin), (0, 255, 0), 1)
          shape = predictor(gray, c)
          for i in range(0, 68):
               x, y = shape.part(i).x, shape.part(i).y
               cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
               cv2.putText(frame, str(i + 1), (x, y -5), 1, 0.8, (0, 255, 255), 1)
     cv2.imshow("Frame", frame)
     if cv2.waitKey(1) == 27 & 0xFF:
          break
cap.release()
cv2.destroyAllWindows()