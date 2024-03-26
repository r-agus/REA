import os
import face_recognition
import numpy as np
import pickle

# Directory where known persons images are stored
known_persons_dir = "known_persons"

# File to store the known encodings and ids
data_file = "known_encodings.pkl"

# Load known encodings and ids if file exists
if os.path.exists(data_file):
    with open(data_file, "rb") as f:
        known_encodings, known_ids = pickle.load(f)
else:
    known_encodings = []
    known_ids = []

# Process each image in the known persons directory
for filename in os.listdir(known_persons_dir):
    # Load the image
    image = face_recognition.load_image_file(os.path.join(known_persons_dir, filename))
    # Encode the image
    encoding = face_recognition.face_encodings(image)[0]
    # Extract the id from the filename
    id = os.path.splitext(filename)[0]
    # Add the encoding and id to the known lists
    known_encodings.append(encoding)
    known_ids.append(id)

# Load the unknown image
unknown_image = face_recognition.load_image_file("unknown.jpg")

# Encode the unknown image
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare faces
results = face_recognition.compare_faces(known_encodings, unknown_encoding)

# Check if the unknown person has been recognized
if True in results:
    # Get the id of the recognized person
    id = known_ids[results.index(True)]
    print(f"The unknown person has been recognized as {id}.")
else:
    # Assign a new id to the unknown person
    new_id = str(len(known_ids) + 1)
    print(f"The unknown person has not been recognized. Assigned new id: {new_id}.")
    # Save the unknown person for future recognition
    known_encodings.append(unknown_encoding)
    known_ids.append(new_id)
    with open(data_file, "wb") as f:
        pickle.dump((known_encodings, known_ids), f)