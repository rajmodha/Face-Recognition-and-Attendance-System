# face_utils.py

import face_recognition
import pickle
import os
from database import Student, Faculty

# Path to the file where face encodings are stored.
# A pickle file is used to save Python objects to disk.
ENCODINGS_PATH = os.path.join("data", "known_faces.pkl")

def _load_encodings():
    """Loads known face encodings from the pickle file."""
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            return pickle.load(f)
    # If no file exists, return an empty structure.
    return {"encodings": [], "names": []}

def _save_encodings(data):
    """Saves the given face encodings to the pickle file."""
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)

def add_user_encoding(user):
    """
    Generates a face encoding from a user's image and adds it to the known faces.
    A face encoding is a unique numerical representation of a face.
    """
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
            # Use the first face found in the image.
            known_face_data["encodings"].append(encodings[0])
            known_face_data["names"].append(user.username)
            _save_encodings(known_face_data)
            print(f"Encoding for {user.full_name} added successfully.")
        else:
            print(f"No face found in the image for {user.full_name}.")
    except Exception as e:
        print(f"Error adding encoding for {user.full_name}: {e}")

def remove_user_encoding(username):
    """Removes a user's face encoding from the known faces by their username."""
    known_face_data = _load_encodings()

    names = known_face_data.get("names", [])
    encodings = known_face_data.get("encodings", [])

    # Create a new list of (name, encoding) pairs, excluding the user to be removed.
    # This is a safe way to rebuild the lists without the specified user.
    filtered_pairs = [
        (name, enc) for name, enc in zip(names, encodings) if name != username
    ]

    # If the new list of pairs is shorter, it means the user was found and removed.
    if len(filtered_pairs) < len(names):
        # If the list is not empty after removal, "unzip" it back into two lists.
        if filtered_pairs:
            new_names, new_encodings = zip(*filtered_pairs)
            known_face_data["names"] = list(new_names)
            known_face_data["encodings"] = list(new_encodings)
        else:
            # If the list is now empty, clear the original data.
            known_face_data["names"] = []
            known_face_data["encodings"] = []

        _save_encodings(known_face_data)
        print(f"Encoding for {username} removed successfully.")
    else:
        print(f"Could not find encoding for username '{username}' to remove.")

def generate_and_save_encodings():
    """
    Re-creates all face encodings from scratch using images of approved users.
    This function overwrites the existing encodings file and requires a Flask
    app context to access the database.
    """
    print("Regenerating all face encodings from the database...")

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
                # If one image fails, print an error and continue with the rest.
                print(f"Error processing image for {user.full_name}: {e}")

    _save_encodings({"encodings": known_encodings, "names": known_names})
    print(f"Encodings regenerated and saved for {len(known_names)} users.")