# app.py

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
from flask import Flask, render_template, request, redirect, url_for, flash, Response
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from database import db, Admin, Faculty, Student
from face_utils import add_user_encoding, remove_user_encoding, generate_and_save_encodings, ENCODINGS_PATH

# --- App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()

# --- Database and File Path Configuration ---
project_dir = os.path.dirname(os.path.abspath(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(project_dir, 'instance/face_attendance.db')}"
app.config['UPLOAD_FOLDER'] = os.path.join(project_dir, 'static/uploads')
ENCODINGS_PATH = os.path.join(project_dir, "known_faces.pkl")

db.init_app(app)

# --- Flask-Login Configuration ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- Helper Functions ---
# (Your _create_student and _create_faculty functions remain the same)
def _create_student(form_data, file_storage, is_approved=False):
    username = form_data['username']
    if Student.query.filter_by(username=username).first() or Faculty.query.filter_by(username=username).first() or Admin.query.filter_by(username=username).first():
        flash('Username already exists. Please choose another.', 'danger')
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
    hashed_password = generate_password_hash(form_data['password'], method='pbkdf2:sha256')
    new_student = Student(username=username, password=hashed_password, full_name=form_data['full_name'], stream=form_data.get('stream'), sem=form_data.get('sem'), image_path=relative_path, is_approved=is_approved)
    db.session.add(new_student)
    db.session.commit()
    return new_student

def _create_faculty(form_data, file_storage, is_approved=True):
    username = form_data['username']
    if Student.query.filter_by(username=username).first() or Faculty.query.filter_by(username=username).first() or Admin.query.filter_by(username=username).first():
        flash('Username already exists. Please choose another.', 'danger')
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
    new_faculty = Faculty(username=username, password=hashed_password, full_name=form_data['full_name'], subject=form_data.get('subject'), image_path=relative_path)
    db.session.add(new_faculty)
    db.session.commit()
    return new_faculty

# --- UPDATED DRAWING FUNCTION ---
def _draw_on_frame(frame, face_locations, face_names, marked_this_session):
    """Helper function to draw larger rectangles and names on the video frame."""
    # Get the height and width of the video frame
    frame_h, frame_w, _ = frame.shape
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale the coordinates back up to the original frame size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # --- CHANGE: Add padding to make the box bigger ---
        # Calculate padding as 25% of the face width
        padding = (right - left) // 8
        
        # Apply padding, ensuring the box stays within the frame boundaries
        top = max(0, top - padding)
        left = max(0, left - padding)
        right = min(frame_w, right + padding)
        bottom = min(frame_h, bottom + padding + 10) # Add a little extra at the bottom for the name tag
        
        # Set box color based on recognition status
        box_color = (0, 0, 255)  # Red for "Unknown"
        if name != "Unknown":
            box_color = (0, 165, 255) if name in marked_this_session else (0, 255, 0) # Orange for marked, Green for new

        # Draw the rectangle for the face
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        
        # Draw the label with a background
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# --- Dlib Initializations and Helper Function ---
# (This section remains the same)
predictor_path = os.path.join(project_dir, "shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# (All other code down to generate_frames remains the same)
# ...
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

def load_known_faces():
    """Load face encodings from the pickle file."""
    global known_face_data
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, 'rb') as f:
            known_face_data = pickle.load(f)

with app.app_context():
    load_known_faces()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if _create_student(request.form, request.files):
            flash('Registration successful! Please wait for admin approval.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
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
            if user.role == 'admin': 
                return redirect(url_for('admin_dashboard'))
            elif user.role == 'faculty': 
                return redirect(url_for('faculty_dashboard'))
            else: 
                return redirect(url_for('student_dashboard'))
        flash('Login failed. Check your username and password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin': 
        return redirect(url_for('index'))
    pending_students = Student.query.filter_by(is_approved=False).all()
    all_faculty = Faculty.query.all()
    all_students = Student.query.filter_by(is_approved=True).order_by(Student.full_name).all()
    return render_template('admin_dashboard.html', students=pending_students, faculties=all_faculty, all_students=all_students)

@app.route('/faculty/dashboard')
@login_required
def faculty_dashboard():
    if current_user.role != 'faculty': 
        return redirect(url_for('index'))
    students = Student.query.filter_by(is_approved=True).all()
    return render_template('faculty_dashboard.html', students=students)

@app.route('/student/dashboard')
@login_required
def student_dashboard():
    if current_user.role != 'student': 
        return redirect(url_for('index'))
    return render_template('student_dashboard.html')

@app.route('/admin/approve/<int:student_id>')
@login_required
def approve_student(student_id):
    if current_user.role != 'admin': 
        return redirect(url_for('index'))
    student = db.get_or_404(Student, student_id)
    student.is_approved = True
    db.session.commit()
    flash(f'{student.full_name} has been approved.', 'success')
    add_user_encoding(student)
    load_known_faces()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/add_faculty', methods=['POST'])
@login_required
def add_faculty():
    if current_user.role != 'admin': 
        return redirect(url_for('index'))
    new_faculty = _create_faculty(request.form, request.files)
    if new_faculty:
        flash('Faculty added successfully.', 'success')
        add_user_encoding(new_faculty)
        load_known_faces()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/add_student', methods=['POST'])
@login_required
def add_student():
    if current_user.role != 'admin': 
        return redirect(url_for('index'))
    new_student = _create_student(request.form, request.files, is_approved=True)
    if new_student:
        flash('Student added successfully and approved.', 'success')
        add_user_encoding(new_student)
        load_known_faces()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/manage_users')
@login_required
def manage_users():
    if current_user.role != 'admin': 
        return redirect(url_for('index'))
    faculties = Faculty.query.all()
    students = Student.query.all()
    return render_template('admin/manage_users.html', faculties=faculties, students=students)

@app.route('/admin/edit_user/<role>/<int:user_id>', methods=['GET', 'POST'])
@login_required
def edit_user(role, user_id):
    if current_user.role != 'admin': 
        return redirect(url_for('index'))
    user_model = None
    if role == 'admin': user_model = Admin
    elif role == 'faculty': user_model = Faculty
    elif role == 'student': user_model = Student
    user_to_edit = db.get_or_404(user_model, user_id)
    original_full_name = user_to_edit.full_name
    if request.method == 'POST':
        new_full_name = request.form['full_name']
        if original_full_name != new_full_name:
            remove_user_encoding(original_full_name)
            user_to_edit.full_name = new_full_name
            add_user_encoding(user_to_edit)
        user_to_edit.username = request.form['username']
        if role == 'student':
            user_to_edit.stream = request.form.get('stream')
            user_to_edit.sem = request.form.get('sem')
        elif role == 'faculty':
            user_to_edit.subject = request.form.get('subject')
        if request.form.get('password'):
            user_to_edit.password = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
        db.session.commit()
        flash(f'User {user_to_edit.username} updated successfully.', 'success')
        load_known_faces()
        return redirect(url_for('manage_users'))
    return render_template('admin/edit_user.html', user=user_to_edit)

@app.route('/admin/delete_user/<role>/<int:user_id>', methods=['POST'])
@login_required
def delete_user(role, user_id):
    if current_user.role != 'admin': 
        return redirect(url_for('index'))
    user_model = None
    if role == 'admin': user_model = Admin
    elif role == 'faculty': user_model = Faculty
    elif role == 'student': user_model = Student
    user_to_delete = db.get_or_404(user_model, user_id)
    user_full_name = user_to_delete.full_name
    if hasattr(user_to_delete, 'image_path') and user_to_delete.image_path:
        try:
            image_path = os.path.join(project_dir, 'static', user_to_delete.image_path)
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception as e:
            print(f"Error deleting image: {e}")
    db.session.delete(user_to_delete)
    db.session.commit()
    remove_user_encoding(user_full_name)
    load_known_faces()
    flash(f'User {user_to_delete.username} has been deleted.', 'success')
    return redirect(url_for(request.form.get('redirect_to', 'manage_users')))

@app.route('/admin/profile', methods=['GET', 'POST'])
@login_required
def admin_profile():
    if current_user.role != 'admin': 
        return redirect(url_for('index'))
    if request.method == 'POST':
        admin_user = db.get_or_404(Admin, current_user.id)
        if not check_password_hash(admin_user.password, request.form.get('current_password')):
            flash('Incorrect current password.', 'danger')
            return redirect(url_for('admin_profile'))
        admin_user.full_name = request.form['full_name']
        admin_user.username = request.form['username']
        if request.form.get('new_password'):
            admin_user.password = generate_password_hash(request.form.get('new_password'), method='pbkdf2:sha256')
        db.session.commit()
        flash('Your profile has been updated successfully.', 'success')
        return redirect(url_for('admin_dashboard'))
    return render_template('admin/admin_profile.html')

@app.route('/admin/regenerate_encodings', methods=['POST', 'GET'])
@login_required
def regenerate_encodings():
    if current_user.role != 'admin': 
        return redirect(url_for('index'))
    generate_and_save_encodings()
    load_known_faces()
    flash('Face encodings regenerated successfully!', 'success')
    return redirect(url_for('admin_dashboard'))

# --- Attendance Tracking ---
def mark_attendance(name, faculty_name, subject):
    """
    Records attendance if the student hasn't been marked for the subject today.
    Returns True if a new record was added, False otherwise.
    """
    reports_dir = os.path.join(project_dir, 'attendance_reports')
    os.makedirs(reports_dir, exist_ok=True)
    filename = os.path.join(reports_dir, f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv")
    file_exists = os.path.isfile(filename)
    
    # Check if already marked to avoid re-opening and seeking
    if file_exists:
        with open(filename, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                if len(parts) == 4 and parts[0] == name and parts[3] == subject:
                    return False # Already marked

    # If not marked, append the new record
    with open(filename, 'a+', newline='') as f:
        if not file_exists or f.tell() == 0:
            f.write('Name,Timestamp,Taken By,Subject\n')
        f.write(f'{name},{datetime.now().strftime("%I:%M:%S %p")},{faculty_name},{subject}\n')
    return True

# --- REVISED generate_frames FUNCTION ---
def generate_frames(faculty_name, subject, student_names):
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    challenge_passed = False
    recognition_done = False
    last_frame_encoded = None
    
    blinks_required = 2
    blink_counter = 0
    eye_closed_for_frames = 0
    EYE_AR_THRESH = 0.22
    EYE_AR_CONSEC_FRAMES = 3
    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)

    # --- CHANGE: Load already marked students for this subject today ---
    marked_students_for_subject = set()
    reports_dir = os.path.join(project_dir, 'attendance_reports')
    today_file = os.path.join(reports_dir, f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv")
    if os.path.exists(today_file):
        with open(today_file, 'r') as f:
            for line in f.readlines()[1:]: # Skip header
                parts = line.strip().split(',')
                if len(parts) == 4 and parts[3] == subject:
                    marked_students_for_subject.add(parts[0])

    if not video_capture.isOpened():
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Error: Camera not found", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        while True:
            ret, buffer = cv2.imencode('.jpg', error_img)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    while True:
        if recognition_done and last_frame_encoded:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + last_frame_encoded + b'\r\n')
            continue

        success, frame = video_capture.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        face_locations = []
        face_names = []
        
        if not challenge_passed:
            instruction_text = f"Blink {blinks_required} times ({blink_counter}/{blinks_required})"
            cv2.putText(frame, instruction_text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
            if rects:
                shape = predictor(gray, rects[0])
                shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
                if ear < EYE_AR_THRESH:
                    eye_closed_for_frames += 1
                else:
                    if eye_closed_for_frames >= EYE_AR_CONSEC_FRAMES:
                        blink_counter += 1
                    eye_closed_for_frames = 0
            if blink_counter >= blinks_required:
                challenge_passed = True
        else:
            cv2.putText(frame, "Liveness Check Passed! Identifying...", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            marked_a_student_this_cycle = False
            
            for face_encoding in face_encodings:
                name = "Unknown"
                if known_face_data["encodings"]:
                    matches = face_recognition.compare_faces(known_face_data["encodings"], face_encoding)
                    face_distances = face_recognition.face_distance(known_face_data["encodings"], face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_data["names"][best_match_index]
                        if name in student_names and name not in marked_students_for_subject:
                            if mark_attendance(name, faculty_name, subject):
                                marked_students_for_subject.add(name)
                                marked_a_student_this_cycle = True
                face_names.append(name)

            _draw_on_frame(frame, face_locations, face_names, marked_students_for_subject)
            
            if marked_a_student_this_cycle:
                cv2.putText(frame, "Marked! Click 'Next Student'", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 2)
                recognition_done = True
            else:
                is_known_face_present = any(name != "Unknown" for name in face_names)
                if face_locations and is_known_face_present:
                     cv2.putText(frame, "Already Marked. Click 'Next Student'.", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 165, 255), 2)
                     recognition_done = True
                elif face_locations and not is_known_face_present:
                     cv2.putText(frame, "Face Not Recognized.", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if recognition_done:
            last_frame_encoded = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- UPDATED take_attendance ROUTE ---
@app.route('/take_attendance')
@login_required
def take_attendance():
    if current_user.role != 'faculty':
        flash('You are not authorized to take attendance.', 'danger')
        return redirect(url_for('index'))
    faculty_subjects = []
    if current_user.subject:
        faculty_subjects = [subject.strip() for subject in current_user.subject.split(',')]
    all_streams = sorted([str(item[0]) for item in db.session.query(Student.stream).distinct()])
    all_sems = sorted([str(item[0]) for item in db.session.query(Student.sem).distinct()])
    return render_template('take_attendance.html', subjects=sorted(faculty_subjects), streams=all_streams, sems=all_sems)


# (The rest of your routes: video_feed, view_attendance, etc., remain the same)
# ...
@app.route('/video_feed')
@login_required
def video_feed():
    subject = request.args.get('subject')
    stream = request.args.get('stream')
    sem = request.args.get('sem')
    
    query = Student.query
    if stream:
        query = query.filter_by(stream=stream)
    if sem:
        query = query.filter_by(sem=sem)
    
    student_names = {student.full_name for student in query.all()}
    return Response(generate_frames(current_user.full_name, subject, student_names), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/view_attendance', methods=['GET', 'POST'])
@login_required
def view_attendance():
    if current_user.role == 'student':
        current_dt = datetime.now()
        year = int(request.form.get('year', current_dt.year))
        month = int(request.form.get('month', current_dt.month))
        days_in_month = calendar.monthrange(year, month)[1]
        sundays = [day for day in range(1, days_in_month + 1) if calendar.weekday(year, month, day) == calendar.SUNDAY]
        in_holidays = holidays.India(years=year)
        indian_holidays_this_month = [date.day for date, name in in_holidays.items() if date.month == month]
        all_holidays = sorted(list(set(sundays + indian_holidays_this_month)))
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
            except ValueError:
                break
        return render_template('view_attendance.html', year=year, month=month, days_in_month=days_in_month, holidays=all_holidays, attendance_data=attendance_data)
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
        return render_template('view_attendance.html', attendance_data=attendance_data, selected_date=selected_date, subjects=sorted(list(subjects_for_dropdown)), selected_subject=selected_subject)

# --- Main Execution ---
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=8080)