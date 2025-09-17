# Importing required libraries
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

# Create a SQLAlchemy database instance.
db = SQLAlchemy()

# Defines the Admin user model for the database.
# Inherits from UserMixin for Flask-Login integration (e.g., is_authenticated)
# and db.Model to be a SQLAlchemy model.
class Admin(UserMixin, db.Model):
    # 'id' is the primary key for the table, an integer that uniquely identifies each admin.
    id = db.Column(db.Integer, primary_key=True)
    # 'username' must be unique for each admin and cannot be empty.
    username = db.Column(db.String(150), unique=True, nullable=False)
    # 'password' stores the hashed password for the admin.
    password = db.Column(db.String(150), nullable=False)
    # 'full_name' stores the full name of the admin.
    full_name = db.Column(db.String(150), nullable=False)
    # 'email' stores the email of the user, must be unique.
    email = db.Column(db.String(150), unique=True, nullable=True)
    # 'role' defines the user's role, defaulting to 'admin'.
    role = db.Column(db.String(50), nullable=False, default='admin')
    # 'reset_token' stores the token for resetting the password.
    reset_token = db.Column(db.String(100), nullable=True)
    # 'reset_token_expiration' stores the expiration date and time of the reset token.
    reset_token_expiration = db.Column(db.DateTime, nullable=True)

    def get_id(self):
        """
        Returns a unique ID for the user, prefixed with their role.
        This is required by Flask-Login to manage user sessions.
        """
        return f'admin-{self.id}'

    def to_dict(self):
        """
        Converts the Admin object into a dictionary.
        Useful for serializing the object, for example, to send as JSON in an API response.
        """
        return {
            'id': self.id,
            'username': self.username,
            'full_name': self.full_name,
            'email': self.email,
            'role': self.role
        }

    def __repr__(self):
        """
        Provides a developer-friendly string representation of the Admin object.
        Useful for debugging.
        """
        return f'<Admin {self.username}>'

# Defines the Faculty user model for the database.
class Faculty(UserMixin, db.Model):
    # 'id' is the primary key for the table.
    id = db.Column(db.Integer, primary_key=True)
    # 'username' must be unique and is used for logging in.
    username = db.Column(db.String(150), unique=True, nullable=False)
    # 'password' stores the hashed password for the faculty member.
    password = db.Column(db.String(150), nullable=False)
    # 'full_name' stores the full name of the faculty member.
    full_name = db.Column(db.String(150), nullable=False)
    # 'email' stores the email of the user, must be unique.
    email = db.Column(db.String(150), unique=True, nullable=True)
    # 'subject' stores the subject(s) taught by the faculty member.
    subject = db.Column(db.String(200), nullable=False)
    # 'image_path' stores the path to the faculty's profile picture for face recognition.
    image_path = db.Column(db.String(200), nullable=True)
    # 'role' defines the user's role, defaulting to 'faculty'.
    role = db.Column(db.String(50), nullable=False, default='faculty')
    # 'reset_token' stores the token for resetting the password.
    reset_token = db.Column(db.String(100), nullable=True)
    # 'reset_token_expiration' stores the expiration date and time of the reset token.
    reset_token_expiration = db.Column(db.DateTime, nullable=True)

    def get_id(self):
        """Returns a unique ID for the user, prefixed with their role."""
        return f'faculty-{self.id}'

    def to_dict(self):
        """Converts the Faculty object into a dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'full_name': self.full_name,
            'email': self.email,
            'subject': self.subject,
            'role': self.role
        }

    def __repr__(self):
        """Provides a developer-friendly string representation of the Faculty object."""
        return f'<Faculty {self.username}>' 

# Defines the Student user model for the database.
class Student(UserMixin, db.Model):
    # 'id' is the primary key for the table.
    id = db.Column(db.Integer, primary_key=True)
    # 'username' must be unique and is used for logging in.
    username = db.Column(db.String(150), unique=True, nullable=False)
    # 'password' stores the hashed password for the student.
    password = db.Column(db.String(150), nullable=False)
    # 'full_name' stores the full name of the student.
    full_name = db.Column(db.String(150), nullable=False)
    # 'email' stores the email of the user, must be unique.
    email = db.Column(db.String(150), unique=True, nullable=True)
    # 'stream' stores the academic stream of the student (e.g., 'Computer Science').
    stream = db.Column(db.String(100), nullable=False)
    # 'sem' stores the current semester of the student.
    sem = db.Column(db.String(50), nullable=False)
    # 'image_path' stores the path to the student's profile picture for face recognition.
    image_path = db.Column(db.String(200), nullable=False)
    # 'is_approved' is a boolean flag to indicate if the student's registration is approved by an admin.
    is_approved = db.Column(db.Boolean, default=False)
    # 'role' defines the user's role, defaulting to 'student'.
    role = db.Column(db.String(50), nullable=False, default='student')
    # 'reset_token' stores the token for resetting the password.
    reset_token = db.Column(db.String(100), nullable=True)
    # 'reset_token_expiration' stores the expiration date and time of the reset token.
    reset_token_expiration = db.Column(db.DateTime, nullable=True)

    def get_id(self):
        """Returns a unique ID for the user, prefixed with their role."""
        return f'student-{self.id}'

    def to_dict(self):
        """Converts the Student object into a dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'full_name': self.full_name,
            'email': self.email,
            'stream': self.stream,
            'sem': self.sem,
            'is_approved': self.is_approved,
            'role': self.role
        }

    def __repr__(self):
        """Provides a developer-friendly string representation of the Student object."""
        return f'<Student {self.username}>'