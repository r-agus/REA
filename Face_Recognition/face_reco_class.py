import os
import face_recognition
import numpy as np
import pickle

class FaceRecognizer:
    def __init__(self, known_persons_dir="known_persons", data_file="known_encodings.pkl"):
        self.known_persons_dir = known_persons_dir
        self.data_file = data_file
        self.known_encodings = []
        self.known_ids = []
        self.load_known_faces()

    def load_known_faces(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, "rb") as f:
                self.known_encodings, self.known_ids = pickle.load(f)
        else:
            for filename in os.listdir(self.known_persons_dir):
                image = face_recognition.load_image_file(os.path.join(self.known_persons_dir, filename))
                encoding = face_recognition.face_encodings(image)[0]
                id = os.path.splitext(filename)[0]
                self.known_encodings.append(encoding)
                self.known_ids.append(id)

    def recognize_face(self, unknown_image_file):
        unknown_image = face_recognition.load_image_file(unknown_image_file)
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        results = face_recognition.compare_faces(self.known_encodings, unknown_encoding)
        if True in results:
            id = self.known_ids[results.index(True)]
            return f"The unknown person has been recognized as {id}."
        else:
            new_id = str(len(self.known_ids) + 1)
            self.known_encodings.append(unknown_encoding)
            self.known_ids.append(new_id)
            with open(self.data_file, "wb") as f:
                pickle.dump((self.known_encodings, self.known_ids), f)
            return f"The unknown person has not been recognized. Assigned new id: {new_id}."

    # Returns true if Teo is recognized.
    def is_teo(self, unknown_image_file):
        id = self.recognize_face(unknown_image_file)
        return id == "Teo"
    
    def is_person(self, unknown_image_file, person_name):
        id = self.recognize_face(unknown_image_file)
        return id == person_name