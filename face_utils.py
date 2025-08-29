# face_utils.py

import face_recognition
import pickle
import os
from database import Student, Faculty

# Define the path for storing the encodings file
ENCODINGS_PATH = "known_faces.pkl"

def _load_encodings():
    """Loads face encodings from the pickle file."""
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}

def _save_encodings(data):
    """Saves face encodings to the pickle file."""
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)

def add_user_encoding(user):
    """Adds a single user's face encoding to the existing pickle file."""
    if not hasattr(user, 'image_path') or not user.image_path:
        return

    image_path = os.path.join("static", user.image_path)
    if not os.path.exists(image_path):
        return

    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            known_face_data = _load_encodings()
            known_face_data["encodings"].append(encodings[0])
            known_face_data["names"].append(user.username)
            _save_encodings(known_face_data)
            print(f"Encoding for {user.full_name} added successfully.")
    except Exception as e:
        print(f"Error adding encoding for {user.full_name}: {e}")

def remove_user_encoding(username):
    """Removes a user's face encoding from the pickle file by their username."""
    known_face_data = _load_encodings()
    if username in known_face_data["names"]:
        indices_to_remove = []
        for i, name in enumerate(known_face_data["names"]):
            if name == username:
                indices_to_remove.append(i)
        
        for index in sorted(indices_to_remove, reverse=True):
            del known_face_data["encodings"][index]
            del known_face_data["names"][index]
        
        _save_encodings(known_face_data)
        print(f"Encoding for {username} removed successfully.")

def generate_and_save_encodings():
    """
    Scans the database for approved users (students and faculty),
    generates face encodings from their images, and saves them to a pickle file.
    This function must be called from within a Flask app context.
    """
    print("Generating and saving face encodings for approved users...")
    
    approved_students = Student.query.filter_by(is_approved=True).filter(Student.image_path != None).all()
    all_faculty = Faculty.query.filter(Faculty.image_path != None).all()

    all_users = approved_students + all_faculty

    known_encodings = []
    known_names = []

    for user in all_users:
        image_path = os.path.join("static", user.image_path)
        
        if os.path.exists(image_path):
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(user.username)
            except Exception as e:
                print(f"Error processing image for {user.full_name}: {e}")

    _save_encodings({"encodings": known_encodings, "names": known_names})
    print("Encodings saved successfully.")