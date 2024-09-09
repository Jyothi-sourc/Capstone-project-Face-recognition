import os
import face_recognition
import numpy as np
from tqdm import tqdm
import random
import pickle
from pytictoc import TicToc

def load_data_from_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
    
def train_face_recognition(images, labels):
    known_face_encodings = []
    known_face_names = []

    timer = TicToc()  # Initialize TicToc
    timer.tic()  # Start timer

    # Iterate with tqdm for progress display
    for image_path, label in tqdm(zip(images, labels), total=len(images), desc="Training face recognition"):
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(label)

    timer.toc()  # Stop timer
    print(f"Encoding completed in {timer.tocvalue()} seconds")

    return known_face_encodings, known_face_names


def recognize_faces(test_images, test_labels, known_face_encodings, known_face_names):
    recognized_names = []
    timer = TicToc()  # Initialize TicToc for timing
    timer.tic()  # Start timer

    # Iterate with tqdm for progress display
    for test_image_path, test_label in tqdm(zip(test_images, test_labels), total=len(test_images), desc="Recognizing faces"):
        test_image = face_recognition.load_image_file(test_image_path)
        face_encodings = face_recognition.face_encodings(test_image)

        if face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            recognized_names.append(name)

    timer.toc()  # Stop timer
    print(f"Face recognition completed in {timer.tocvalue()} seconds")
    return recognized_names

# Load and encode faces
images, labels = load_data_from_pickle('images_labels.pkl')
known_face_encodings, known_face_names = train_face_recognition(images, labels)

# Select a random sample of 10% images for recognition
sample_size = len(images)//10
sample_indices = random.sample(range(len(images)), sample_size)
sample_images = [images[i] for i in sample_indices]
sample_labels = [labels[i] for i in sample_indices]

# Recognize faces in the sample
recognized_names = recognize_faces(sample_images, sample_labels, known_face_encodings, known_face_names)

# Compare results 
correct_matches = sum(recognized_name == label for recognized_name, label in zip(recognized_names, sample_labels))
print(f"Accuracy: {correct_matches / sample_size * 100}%")
