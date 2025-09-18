# Import libraries
import os
import cv2
import numpy as np
import pickle
import face_recognition
import dlib
from scipy.spatial import distance as dist
from datetime import datetime
import calendar
import holidays
from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify, send_file
import pandas as pd
import io
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message
import secrets
import re
from datetime import datetime, timedelta

# Imports from modules
from database import db, Admin, Faculty, Student
from face_utils import add_user_encoding, remove_user_encoding, generate_and_save_encodings, ENCODINGS_PATH
import dotenv

# App instance
dotenv.load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# Configure Flask-Mail for sending password reset emails.
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').lower() in ('true', '1', 't')
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')
mail = Mail(app)

# Configure database and file paths.
project_dir = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(project_dir, 'data')
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(project_dir, 'instance/face_attendance.db')}"
app.config['UPLOAD_FOLDER'] = os.path.join(project_dir, 'static/uploads')
ENCODINGS_PATH = os.path.join(data, "known_faces.pkl")
db.init_app(app)

# Configure Flask-Login for user session management.
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# In-memory state for attendance sessions to manage liveness checks.
ATTENDANCE_STATE = {}

# Helper functions
def load_known_faces():
    """Loads known face encodings from the pickle file into memory."""
    global known_face_data
    if os.path.exists(ENCODINGS_PATH):
        try:
            with open(ENCODINGS_PATH, 'rb') as f:
                known_face_data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Error loading encoding file: {e}")
            known_face_data = {"encodings": [], "names": []}
        except Exception as e:
            print(f"An unexpected error occurred while loading encodings: {e}")
            known_face_data = {"encodings": [], "names": []}

def reset_attendance_state(user_id):
    """Resets the attendance state for a given user."""
    if user_id in ATTENDANCE_STATE:
        state = ATTENDANCE_STATE[user_id]
        state['challenge_passed'] = False
        state['recognition_done'] = False
        state['blink_counter'] = 0
        state['eye_closed_for_frames'] = 0
        state['last_frame_encoded'] = None
    else:
        ATTENDANCE_STATE[user_id] = {
            'challenge_passed': False,
            'recognition_done': False,
            'blink_counter': 0,
            'eye_closed_for_frames': 0,
            'last_frame_encoded': None
        }

def get_attendance_state(user_id):
    """Gets the attendance state for a user, initializing if not present."""
    if user_id not in ATTENDANCE_STATE:
        reset_attendance_state(user_id)
    return ATTENDANCE_STATE[user_id]

def send_reset_email(user):
    """Generates a password reset token and sends a reset email to the user."""
    token = secrets.token_urlsafe(32)
    user.reset_token = token
    user.reset_token_expiration = datetime.utcnow() + timedelta(minutes=15)
    db.session.commit()

    msg = Message('Password Reset Request',
                  sender=app.config['MAIL_DEFAULT_SENDER'],
                  recipients=[user.email])

    reset_url = url_for('reset_token', token=token, _external=True)
    msg.body = f'''To reset your password, visit the following link:
    {reset_url}
    If you did not make this request then simply ignore this email and no changes will be made.
    '''
    mail.send(msg)

def get_user_by_email(email):
    """Finds a user by their email across all user types (Admin, Faculty, Student)."""
    user = Admin.query.filter_by(email=email).first()
    if not user:
        user = Faculty.query.filter_by(email=email).first()
    if not user:
        user = Student.query.filter_by(email=email).first()
    return user

def _validate_user_credentials(username, password, email, existing_username=None, existing_email=None):
    """Validates username and password constraints for new and existing users."""
    if not username or not username.islower() or ' ' in username:
        flash('Username must be in lowercase and without spaces.', 'danger')
        return False

    if email and not re.match(r'[^@]+@[^@]+\.[^@]+', email):
        flash('Invalid email address.', 'danger')
        return False

    if email and (not existing_email or email != existing_email):
        if Student.query.filter_by(email=email).first() or \
           Faculty.query.filter_by(email=email).first() or \
           Admin.query.filter_by(email=email).first():
            flash('Email address already in use. Please choose another.', 'danger')
            return False

    reserved_keywords = ['admin', 'faculty', 'student']
    if any(keyword in username for keyword in reserved_keywords):
        flash(f'Username cannot contain reserved words: {", ".join(reserved_keywords)}.', 'danger')
        return False

    if not existing_username and not password:
        flash('Password is required.', 'danger')
        return False
        
    if password and not _validate_password_length(password):
        return False
    if not existing_username or username != existing_username:
        if Student.query.filter_by(username=username).first() or \
           Faculty.query.filter_by(username=username).first() or \
           Admin.query.filter_by(username=username).first():
            flash('Username already exists. Please choose another.', 'danger')
            return False
            
    return True

def _validate_password_length(password):
    """Validates that the password meets the length requirements."""
    if password and (len(password) < 8 or len(password) > 14):
        flash('Password must be at least 8 characters long and no more than 14 characters long.', 'danger')
        return False
    return True

def get_available_cameras():
    """Detects and returns a list of available camera indices."""
    index = 0
    arr = []
    while index < 5:
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if cap.isOpened():
            arr.append(index)
            cap.release()
        index += 1
    return arr

def _create_student(form_data, file_storage, is_approved=False, email=None):
    """function to create a new student, handling validation and file uploads."""
    username = form_data['username']
    password = form_data['password']

    if not _validate_user_credentials(username, password, email):
        return None

    if 'profile_pic' not in file_storage or not file_storage['profile_pic'].filename:
        flash('Profile picture is required.', 'danger')
        return None
    
    file = file_storage['profile_pic']
    filename = secure_filename(file.filename)
    stream = form_data.get('stream', 'unknown')
    sem = form_data.get('sem', 'unknown')
    student_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'students', stream, sem)
    os.makedirs(student_upload_dir, exist_ok=True)
    save_path = os.path.join(student_upload_dir, filename)
    file.save(save_path)
    relative_path = os.path.join('uploads', 'students', stream, sem, filename)
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    new_student = Student(username=username, password=hashed_password, full_name=form_data['full_name'], email=email, stream=form_data.get('stream'), sem=form_data.get('sem'), image_path=relative_path, is_approved=is_approved)
    db.session.add(new_student)
    db.session.commit()
    return new_student

def _create_faculty(form_data, file_storage, is_approved=True, email=None):
    """function to create a new faculty member."""
    username = form_data['username']
    password = form_data['password']

    if not _validate_user_credentials(username, password, email):
        return None

    if 'profile_pic' not in file_storage or not file_storage['profile_pic'].filename:
        flash('Profile picture is required.', 'danger')
        return None
    file = file_storage['profile_pic']
    filename = secure_filename(file.filename)
    faculty_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'faculty')
    os.makedirs(faculty_upload_dir, exist_ok=True)
    save_path = os.path.join(faculty_upload_dir, filename)
    file.save(save_path)
    relative_path = os.path.join('uploads', 'faculty', filename)
    hashed_password = generate_password_hash(form_data['password'], method='pbkdf2:sha256')
    new_faculty = Faculty(username=username, password=hashed_password, full_name=form_data['full_name'], email=email, subject=form_data.get('subject'), image_path=relative_path)
    db.session.add(new_faculty)
    db.session.commit()
    return new_faculty

def _draw_on_frame(frame, face_locations, face_names, marked_this_session):
    """Draws rectangles and names on the video frame for recognized faces."""
    frame_h, frame_w, _ = frame.shape
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        padding = (right - left) // 8
        
        top = max(0, top - padding)
        left = max(0, left - padding)
        right = min(frame_w, right + padding)
        bottom = min(frame_h, bottom + padding + 10)
        
        box_color = (0, 0, 255)
        if name != "Unknown":
            box_color = (0, 165, 255) if name in marked_this_session else (0, 255, 0)

        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

def mark_attendance(name, faculty_name, subject):
    """Records student attendance in a CSV file for the current day."""
    reports_dir = os.path.join(project_dir, 'attendance_reports')
    os.makedirs(reports_dir, exist_ok=True)
    filename = os.path.join(reports_dir, f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv")
    file_exists = os.path.isfile(filename)
    
    if file_exists:
        with open(filename, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                if len(parts) == 4 and parts[0] == name and parts[3] == subject:
                    return False

    with open(filename, 'a+', newline='') as f:
        if not file_exists or f.tell() == 0:
            f.write('Name,Timestamp,Taken By,Subject\n')
        f.write(f'{name},{datetime.now().strftime("%I:%M:%S %p")},{faculty_name},{subject}\n')
    return True

def generate_frames(faculty_name, subject, student_names, camera_index=0, start_session=False, user_id=None):
    """Generates video frames for the attendance-taking process, including liveness detection and face recognition."""
    if user_id is None:
        return

    if start_session:
        reset_attendance_state(user_id)
    state = get_attendance_state(user_id)
    if not os.path.exists(predictor_path):
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Error: Predictor file not found", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        while True:
            ret, buffer = cv2.imencode('.jpg', error_img)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
    with app.app_context():
        all_approved_students = Student.query.filter_by(is_approved=True).all()
        username_to_fullname = {student.username: student.full_name for student in all_approved_students}

    video_capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    blinks_required = 2
    EYE_AR_THRESH = 0.22
    EYE_AR_CONSEC_FRAMES = 3
    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)

    marked_students_for_subject = set()
    reports_dir = os.path.join(project_dir, 'attendance_reports')
    today_file = os.path.join(reports_dir, f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv")
    if os.path.exists(today_file):
        try:
            with open(today_file, 'r') as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split(',')
                    if len(parts) == 4 and parts[3] == subject:
                        marked_students_for_subject.add(parts[0])
        except IOError as e:
            print(f"Error reading attendance file: {e}")


    if not video_capture.isOpened():
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Error: Camera not found", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        while True:
            ret, buffer = cv2.imencode('.jpg', error_img)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    while True:
        try:
            if state['recognition_done'] and state['last_frame_encoded']:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + state['last_frame_encoded'] + b'\r\n')
                continue

            success, frame = video_capture.read()
            if not success:
                break

            frame = cv2.flip(frame, 2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            face_locations = []
            face_names = []
            
            if not state['challenge_passed']:
                instruction_text = f"Blink {blinks_required} times ({state['blink_counter']}/{blinks_required})"
                cv2.putText(frame, instruction_text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
                if rects:
                    shape = predictor(gray, rects[0])
                    shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
                    if ear < EYE_AR_THRESH:
                        state['eye_closed_for_frames'] += 1
                    else:
                        if state['eye_closed_for_frames'] >= EYE_AR_CONSEC_FRAMES:
                            state['blink_counter'] += 1
                        state['eye_closed_for_frames'] = 0
                if state['blink_counter'] >= blinks_required:
                    state['challenge_passed'] = True
            else:
                cv2.putText(frame, "Liveness Check Passed!", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                marked_a_student_this_cycle = False
                
                for face_encoding in face_encodings:
                    username = "Unknown"
                    if known_face_data["encodings"]:
                        matches = face_recognition.compare_faces(known_face_data["encodings"], face_encoding)
                        face_distances = face_recognition.face_distance(known_face_data["encodings"], face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            username = known_face_data["names"][best_match_index]
                            
                            full_name = username_to_fullname.get(username)

                            if full_name and full_name in student_names and full_name not in marked_students_for_subject:
                                try:
                                    if mark_attendance(full_name, faculty_name, subject):
                                        marked_students_for_subject.add(full_name)
                                        marked_a_student_this_cycle = True
                                except IOError as e:
                                    print(f"Error marking attendance: {e}")
                    
                    name_to_display = "Unknown"
                    if 'full_name' in locals() and full_name:
                        name_to_display = full_name
                        if full_name not in student_names:
                            cv2.putText(frame, "Not in selected stream/sem", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
                            state['recognition_done'] = True
                    face_names.append(name_to_display)

                _draw_on_frame(frame, face_locations, face_names, marked_students_for_subject)
                
                if marked_a_student_this_cycle:
                    cv2.putText(frame, "Marked! Click 'Next Student'", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 2)
                    state['recognition_done'] = True
                else:
                    is_known_face_present = any(name != "Unknown" for name in face_names)
                    if face_locations and is_known_face_present and full_name in student_names:
                         cv2.putText(frame, "Already Marked. Click 'Next Student'.", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 165, 255), 2)
                         state['recognition_done'] = True
                    elif face_locations and not is_known_face_present:
                         cv2.putText(frame, "Face Not Recognized.", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
                         state['recognition_done'] = True

            ret, buffer = cv2.imencode('.jpg', frame)
            if state['recognition_done']:
                state['last_frame_encoded'] = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except cv2.error as e:
            print(f"OpenCV error in generate_frames: {e}")
            break
        except KeyError:
            break

def eye_aspect_ratio(eye):
    """Calculates the eye aspect ratio to detect blinks."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize dlib's face detector and facial landmark predictor.
predictor_path = os.path.join(data, "shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

with app.app_context():
    os.makedirs(os.path.join(project_dir, 'instance'), exist_ok=True)
    db.create_all()
    if not Admin.query.filter_by(username='admin').first():
        hashed_password = generate_password_hash('admin', method='pbkdf2:sha256')
        admin = Admin(username='admin', password=hashed_password, full_name='Admin User')
        db.session.add(admin)
        db.session.commit()

known_face_data = {"encodings": [], "names": []}

@login_manager.user_loader
def load_user(user_id):
    """Loads a user from the database based on the user ID stored in the session."""
    try:
        role, user_id = user_id.split('-')
        user_id = int(user_id)
    except ValueError:
        return None

    if role == 'admin':
        return Admin.query.get(user_id)
    elif role == 'faculty':
        return Faculty.query.get(user_id)
    elif role == 'student':
        return Student.query.get(user_id)
    return None

with app.app_context():
    load_known_faces()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles student registration."""
    if request.method == 'POST':
        if _create_student(request.form, request.files, email=request.form.get('email')):
            flash('Registration successful! Please wait for admin approval.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login for all roles."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = Admin.query.filter_by(username=username).first()
        if not user:
            user = Faculty.query.filter_by(username=username).first()
        if not user:
            user = Student.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            if user.role == 'student' and not user.is_approved:
                flash('Your account is pending approval from the admin.', 'warning')
                return redirect(url_for('login'))
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        flash('Login failed. Check your username and password.', 'danger')
    return render_template('login.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    """Handles the request to reset a password."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        email = request.form.get('email')
        user = get_user_by_email(email)
        if user:
            send_reset_email(user)
        flash('If an account with that email exists, a password reset link has been sent.', 'info')
        return redirect(url_for('login'))
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_token(token):
    """Handles password reset using a token."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    user = Admin.query.filter_by(reset_token=token).first()
    if not user:
        user = Faculty.query.filter_by(reset_token=token).first()
    if not user:
        user = Student.query.filter_by(reset_token=token).first()

    if not user or user.reset_token_expiration < datetime.utcnow():
        flash('That is an invalid or expired token.', 'warning')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not password or not confirm_password:
            flash('Password and confirm password are required.', 'danger')
            return render_template('reset_password.html', token=token)

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('reset_password.html', token=token)

        if not _validate_password_length(password):
            return render_template('reset_password.html', token=token)

        user.password = generate_password_hash(password, method='pbkdf2:sha256')
        user.reset_token = None
        user.reset_token_expiration = None
        db.session.commit()
        flash('Your password has been updated! You are now able to log in.', 'success')
        return redirect(url_for('login'))

    return render_template('reset_password.html', token=token)

@app.route('/logout')
@login_required
def logout():
    """Logs the current user out."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    """Displays the admin dashboard with pending students and all users."""
    if current_user.role != 'admin': 
        return redirect(url_for('index'))
    pending_students = Student.query.filter_by(is_approved=False).all()
    all_faculty = Faculty.query.all()
    all_students = Student.query.filter_by(is_approved=True).order_by(Student.full_name).all()
    return render_template('admin_dashboard.html', students=pending_students, faculties=all_faculty, all_students=all_students)

@app.route('/faculty/dashboard')
@login_required
def faculty_dashboard():
    """Displays the faculty dashboard."""
    if current_user.role != 'faculty': 
        return redirect(url_for('index'))
    students = Student.query.filter_by(is_approved=True).all()
    return render_template('faculty_dashboard.html', students=students)

@app.route('/student/dashboard')
@login_required
def student_dashboard():
    """Displays the student dashboard."""
    if current_user.role != 'student': 
        return redirect(url_for('index'))
    return render_template('student_dashboard.html')

@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Allows logged-in users to change their password."""
    user_model = None
    if current_user.role == 'faculty':
        user_model = Faculty
    elif current_user.role == 'student':
        user_model = Student
    elif current_user.role == 'admin':
        user_model = Admin
    else:
        flash('You are not authorized to change password.', 'danger')
        return redirect(url_for('index'))

    user = user_model.query.get(current_user.id)

    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        if not check_password_hash(user.password, current_password):
            flash('Incorrect current password.', 'danger')
            return render_template('change_password.html')
        
        if not new_password or not confirm_password:
            flash('New password and confirm password are required.', 'danger')
            return render_template('change_password.html')

        if new_password != confirm_password:
            flash('New password and confirm password do not match.', 'danger')
            return render_template('change_password.html')

        if not _validate_password_length(new_password):
            return render_template('change_password.html')

        user.password = generate_password_hash(new_password, method='pbkdf2:sha256')
        db.session.commit()
        flash('Password updated successfully!', 'success')
        
        if current_user.role == 'faculty':
            return redirect(url_for('faculty_dashboard'))
        elif current_user.role == 'student':
            return redirect(url_for('student_dashboard'))
        elif current_user.role == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('index'))

    return render_template('change_password.html')

@app.route('/admin/approve/<int:student_id>')
@login_required
def approve_student(student_id):
    """Approves a pending student registration."""
    if current_user.role != 'admin': 
        return redirect(url_for('index'))
    student = db.get_or_404(Student, student_id)
    student.is_approved = True
    db.session.commit()
    flash(f'{student.full_name} has been approved.', 'success')
    add_user_encoding(student)
    load_known_faces()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/decline_student/<int:student_id>', methods=['POST'])
@login_required
def decline_student(student_id):
    """Declines and deletes a pending student registration."""
    if current_user.role != 'admin':
        flash('You are not authorized to perform this action.', 'danger')
        return redirect(url_for('index'))
    
    student = db.get_or_404(Student, student_id)
    
    if hasattr(student, 'image_path') and student.image_path:
        try:
            image_path = os.path.join(project_dir, 'static', student.image_path)
            if os.path.exists(image_path):
                os.remove(image_path)
        except OSError as e:
            print(f"Error deleting image for {student.username}: {e}")

    remove_user_encoding(student.username)
    load_known_faces()

    db.session.delete(student)
    db.session.commit()
    flash(f'Student {student.full_name} has been declined and deleted.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/add_faculty', methods=['POST'])
@login_required
def add_faculty():
    """Adds a new faculty member."""
    if current_user.role != 'admin': 
        return redirect(url_for('index'))
    new_faculty = _create_faculty(request.form, request.files, email=request.form.get('email'))
    if new_faculty:
        flash('Faculty added successfully.', 'success')
        add_user_encoding(new_faculty)
        load_known_faces()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/add_student', methods=['POST'])
@login_required
def add_student():
    """Adds a new, pre-approved student."""
    if current_user.role != 'admin': 
        return redirect(url_for('index'))
    new_student = _create_student(request.form, request.files, is_approved=True, email=request.form.get('email'))
    if new_student:
        flash('Student added successfully and approved.', 'success')
        add_user_encoding(new_student)
        load_known_faces()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/manage_users')
@login_required
def manage_users():
    """Renders the user management page for admins."""
    if current_user.role != 'admin': 
        return redirect(url_for('index'))
    return render_template('admin/manage_users.html')

@app.route('/admin/add_admin', methods=['GET', 'POST'])
@login_required
def add_admin():
    """Allows an admin to add another admin."""
    if current_user.role != 'admin':
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form.get('email')

        if not _validate_user_credentials(username, password, email):
            return render_template('admin/add_admin.html')
        
        hashed_password = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
        new_admin = Admin(
            username=username,
            password=hashed_password,
            full_name=request.form['full_name'],
            email=request.form.get('email')
        )
        db.session.add(new_admin)
        db.session.commit()
        flash('Admin added successfully.', 'success')
        return redirect(url_for('manage_users'))
    return render_template('admin/add_admin.html')

@app.route('/admin/edit_user/<role>/<int:user_id>', methods=['GET', 'POST'])
@login_required
def edit_user(role, user_id):
    """Allows admins to edit user details."""
    if current_user.role != 'admin':
        return redirect(url_for('index'))

    user_model = {'admin': Admin, 'faculty': Faculty, 'student': Student}.get(role)
    if not user_model:
        flash('Invalid user role.', 'danger')
        return redirect(url_for('manage_users'))

    user_to_edit = db.get_or_404(user_model, user_id)
    original_username = user_to_edit.username

    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        new_username = request.form.get('username', '').strip().lower()
        new_password = request.form.get('password')
        email = request.form.get('email')

        if not _validate_user_credentials(new_username, new_password, email, existing_username=original_username, existing_email=user_to_edit.email):
            return render_template('admin/edit_user.html', user=user_to_edit)

        if not full_name:
            flash('Full Name cannot be empty.', 'danger')
            return render_template('admin/edit_user.html', user=user_to_edit)

        if role == 'student':
            stream = request.form.get('stream')
            sem = request.form.get('sem')
            if not stream:
                flash('Stream cannot be empty for students.', 'danger')
                return render_template('admin/edit_user.html', user=user_to_edit)
            if not sem:
                flash('Semester cannot be empty for students.', 'danger')
                return render_template('admin/edit_user.html', user=user_to_edit)
            user_to_edit.stream = stream
            user_to_edit.sem = sem
        elif role == 'faculty':
            subject = request.form.get('subject', '').strip()
            if not subject:
                flash('Subject cannot be empty for faculty.', 'danger')
                return render_template('admin/edit_user.html', user=user_to_edit)
            user_to_edit.subject = subject

        user_to_edit.full_name = full_name
        user_to_edit.username = new_username
        user_to_edit.email = email

        if new_password:
            user_to_edit.password = generate_password_hash(new_password, method='pbkdf2:sha256')

        db.session.commit()

        if original_username != new_username:
            remove_user_encoding(original_username)
            add_user_encoding(user_to_edit)
            load_known_faces()

        flash(f'User {user_to_edit.username} updated successfully.', 'success')
        return redirect(url_for('manage_users'))

    return render_template('admin/edit_user.html', user=user_to_edit)


@app.route('/admin/delete_user/<role>/<int:user_id>', methods=['POST'])
@login_required
def delete_user(role, user_id):
    """Allows admins to delete users."""
    if current_user.role != 'admin': 
        return redirect(url_for('index'))
    user_model = None
    if role == 'admin': user_model = Admin
    elif role == 'faculty': user_model = Faculty
    elif role == 'student': user_model = Student
    user_to_delete = db.get_or_404(user_model, user_id)
    user_username = user_to_delete.username
    if hasattr(user_to_delete, 'image_path') and user_to_delete.image_path:
        try:
            image_path = os.path.join(project_dir, 'static', user_to_delete.image_path)
            if os.path.exists(image_path):
                os.remove(image_path)
        except OSError as e:
            print(f"Error deleting image: {e}")
    db.session.delete(user_to_delete)
    db.session.commit()
    remove_user_encoding(user_username)
    load_known_faces()
    flash(f'User {user_to_delete.username} has been deleted.', 'success')
    return redirect(url_for(request.form.get('redirect_to', 'manage_users')))

@app.route('/admin/profile', methods=['GET', 'POST'])
@login_required
def admin_profile():
    """Allows the admin to update their own profile."""
    if current_user.role != 'admin': 
        return redirect(url_for('index'))
    if request.method == 'POST':
        new_username = request.form['username']
        full_name = request.form['full_name']

        if not new_username or not new_username.islower() or ' ' in new_username:
            flash('Username must be in lowercase and without spaces.', 'danger')
            return render_template('admin/admin_profile.html')

        reserved_keywords = ['admin', 'faculty', 'student']
        if any(keyword in new_username for keyword in reserved_keywords):
            flash(f'Username cannot contain reserved words: {", ".join(reserved_keywords)}.', 'danger')
            return render_template('admin/admin_profile.html')

        admin_user = db.get_or_404(Admin, current_user.id) 
        
        if new_username != admin_user.username:
            if Admin.query.filter_by(username=new_username).first():
                flash('Username already exists. Please choose another.', 'danger')
                return render_template('admin/admin_profile.html')

        admin_user.full_name = full_name
        admin_user.username = new_username
        
        db.session.commit()
        flash('Your profile has been updated successfully.', 'success')
        return redirect(url_for('admin_dashboard'))
    return render_template('admin/admin_profile.html')

@app.route('/admin/regenerate_encodings', methods=['POST', 'GET'])
@login_required
def regenerate_encodings():
    """Triggers a regeneration of all face encodings."""
    if current_user.role != 'admin': 
        return redirect(url_for('index'))
    generate_and_save_encodings()
    load_known_faces()
    flash('Face encodings regenerated successfully!', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/next_student', methods=['POST'])
@login_required
def next_student():
    """Resets the attendance state for the next student."""
    if current_user.role not in ['faculty', 'admin']:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
    reset_attendance_state(current_user.get_id())
    return jsonify({'status': 'success', 'message': 'State reset for next student.'})


@app.route('/take_attendance')
@login_required
def take_attendance():
    """Renders the page for taking attendance."""
    if current_user.role not in ['faculty', 'admin']:
        flash('You are not authorized to take attendance.', 'danger')
        return redirect(url_for('index'))
    
    subjects = set()
    if current_user.role == 'faculty':
        if current_user.subject:
            subjects.update([s.strip() for s in current_user.subject.split(',')])
    elif current_user.role == 'admin':
        all_faculty = Faculty.query.all()
        for f in all_faculty:
            if f.subject:
                subjects.update([s.strip() for s in f.subject.split(',')])

    all_subjects = sorted(list(subjects))
    all_streams = sorted([str(item[0]) for item in db.session.query(Student.stream).distinct()])
    all_sems = sorted([str(item[0]) for item in db.session.query(Student.sem).distinct()])
    return render_template('take_attendance.html', subjects=sorted(all_subjects), streams=all_streams, sems=all_sems)

@app.route('/get_cameras')
@login_required
def get_cameras():
    """Returns a list of available cameras."""
    if current_user.role not in ['faculty', 'admin']:
        return jsonify({'error': 'Unauthorized'}), 401
    available_cameras = get_available_cameras()
    return jsonify({'cameras': available_cameras})

@app.route('/video_feed')
@login_required
def video_feed():
    """Provides the video feed for face recognition."""
    subject = request.args.get('subject')
    stream = request.args.get('stream')
    sem = request.args.get('sem')
    camera_index = int(request.args.get('camera', 0))
    start_session = request.args.get('start') == 'true'
    user_id = current_user.get_id()
    
    query = Student.query
    if stream:
        query = query.filter_by(stream=stream)
    if sem:
        query = query.filter_by(sem=sem)
    
    student_names = {student.full_name for student in query.all()}
    return Response(generate_frames(current_user.full_name, subject, student_names, camera_index, start_session=start_session, user_id=user_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/view_attendance', methods=['GET', 'POST'])
@login_required
def view_attendance():
    """Displays attendance records for students, faculty, and admins."""
    if current_user.role == 'student':
        current_dt = datetime.now()
        year = int(request.form.get('year', current_dt.year))
        month = int(request.form.get('month', current_dt.month))
        days_in_month = calendar.monthrange(year, month)[1]
        holiday_info = {}
        for day in range(1, days_in_month + 1):
            if calendar.weekday(year, month, day) == calendar.SUNDAY:
                holiday_info[day] = "Sunday"
        in_holidays = holidays.India(years=year)
        for date, name in in_holidays.items():
            if date.year == year and date.month == month:
                holiday_info[date.day] = name
        attendance_data = {}
        reports_dir = os.path.join(project_dir, 'attendance_reports')
        for day in range(1, days_in_month + 1):
            try:
                date_to_check = datetime(year, month, day)
                filename = os.path.join(reports_dir, f"attendance_{date_to_check.strftime('%Y-%m-%d')}.csv")
                if os.path.exists(filename):
                    with open(filename, 'r') as f:
                        next(f, None)
                        for line in f:
                            parts = line.strip().split(',')
                            if len(parts) < 4: continue
                            record_name, _, _, record_subject = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                            if record_name == current_user.full_name:
                                if day not in attendance_data:
                                    attendance_data[day] = []
                                attendance_data[day].append(record_subject)
            except (ValueError, IOError) as e:
                print(f"Error processing attendance file for {date_to_check.strftime('%Y-%m-%d')}: {e}")
                break
        return render_template('view_attendance.html', year=year, month=month, days_in_month=days_in_month, holidays=holiday_info, attendance_data=attendance_data)
    else:
        selected_date = request.form.get('date', datetime.now().strftime('%Y-%m-%d'))
        selected_subject = request.form.get('subject', 'all')
        subjects_for_dropdown = set()
        if current_user.role == 'faculty':
            subjects_for_dropdown.update([s.strip() for s in current_user.subject.split(',')])
        else:
            all_faculties = Faculty.query.all()
            for f in all_faculties:
                subjects_for_dropdown.update([s.strip() for s in f.subject.split(',')])
        attendance_data = []
        reports_dir = os.path.join(project_dir, 'attendance_reports')
        filename = os.path.join(reports_dir, f"attendance_{selected_date}.csv")
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    next(f, None)
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) < 4: continue
                        record_name, record_timestamp, record_taken_by, record_subject = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                        if selected_subject != 'all' and record_subject != selected_subject:
                            continue
                        if current_user.role == 'faculty':
                            faculty_subjects = [s.strip() for s in current_user.subject.split(',')]
                            if record_subject not in faculty_subjects:
                                continue
                        attendance_data.append({'name': record_name, 'timestamp': record_timestamp, 'taken_by': record_taken_by, 'subject': record_subject})
            except IOError as e:
                print(f"Error reading attendance file {filename}: {e}")
        return render_template('view_attendance.html', attendance_data=attendance_data, selected_date=selected_date, subjects=sorted(list(subjects_for_dropdown)), selected_subject=selected_subject)

@app.route('/export_attendance', methods=['GET', 'POST'])
@login_required
def export_attendance():
    """Exports attendance data to a CSV file."""
    if current_user.role not in ['faculty', 'admin']:
        flash('You are not authorized to export attendance reports.', 'danger')
        return redirect(url_for('index'))

    subjects = set()
    if current_user.role == 'faculty':
        if current_user.subject:
            subjects.update([s.strip() for s in current_user.subject.split(',')])
    elif current_user.role == 'admin':
        all_faculty = Faculty.query.all()
        for f in all_faculty:
            if f.subject:
                subjects.update([s.strip() for s in f.subject.split(',')])

    if request.method == 'POST':
        start_date_str = request.form.get('start_date')
        end_date_str = request.form.get('end_date')
        subject = request.form.get('subject')

        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError:
            flash('Invalid date format. Please use YYYY-MM-DD.', 'danger')
            return render_template('export_attendance.html', subjects=sorted(list(subjects)))

        all_dfs = []
        current_date = start_date
        while current_date <= end_date:
            filename = os.path.join(project_dir, 'attendance_reports', f"attendance_{current_date.strftime('%Y-%m-%d')}.csv")
            if os.path.exists(filename):
                try:
                    df = pd.read_csv(filename)
                    df['Date'] = current_date.strftime('%Y-%m-%d')
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
            current_date += timedelta(days=1)

        if not all_dfs:
            flash('No attendance data found for the selected criteria.', 'info')
            return render_template('export_attendance.html', subjects=sorted(list(subjects)))

        combined_df = pd.concat(all_dfs, ignore_index=True)

        if subject != 'all':
            combined_df = combined_df[combined_df['Subject'] == subject]

        if combined_df.empty:
            flash('No attendance data found for the selected subject.', 'info')
            return render_template('export_attendance.html', subjects=sorted(list(subjects)))

        cols = ['Date'] + [col for col in combined_df.columns if col != 'Date']
        combined_df = combined_df[cols]

        output = io.StringIO()
        combined_df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'attendance_report_{start_date_str}_to_{end_date_str}.csv'
        )

    return render_template('export_attendance.html', subjects=sorted(list(subjects)))

@app.route('/admin/search_users')
@login_required
def search_users():
    """API endpoint for searching users."""
    if current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401

    user_type = request.args.get('type', 'student')

    query = None

    if user_type == 'admin':
        query = Admin.query.filter(Admin.id != current_user.id)
    elif user_type == 'faculty':
        query = Faculty.query
    else:
        query = Student.query
    
    users = query.all()

    return jsonify([user.to_dict() for user in users])

@app.route('/api/pending_students')
@login_required
def get_pending_students():
    """API endpoint to get a list of pending students."""
    if current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401
    pending_students = Student.query.filter_by(is_approved=False).all()
    return jsonify([student.to_dict() for student in pending_students])

@app.route('/api/approved_students')
@login_required
def get_approved_students():
    """API endpoint to get a list of approved students."""
    if current_user.role not in ['faculty', 'admin']:
        return jsonify({'error': 'Unauthorized'}), 401
    approved_students = Student.query.filter_by(is_approved=True).all()
    return jsonify([student.to_dict() for student in approved_students])


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=8080)
