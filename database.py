# database.py

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class Admin(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    full_name = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), nullable=False, default='admin')

    def get_id(self):
        return f'admin-{self.id}'

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'full_name': self.full_name,
            'role': self.role
        }

    def __repr__(self): #Makes the object printable
        return f'<Admin {self.username}>'

class Faculty(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    full_name = db.Column(db.String(150), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    image_path = db.Column(db.String(200), nullable=True)
    role = db.Column(db.String(50), nullable=False, default='faculty')

    def get_id(self):
        return f'faculty-{self.id}'

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'full_name': self.full_name,
            'subject': self.subject,
            'role': self.role
        }

    def __repr__(self): 
        return f'<Faculty {self.username}>' 

class Student(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    full_name = db.Column(db.String(150), nullable=False)
    stream = db.Column(db.String(100), nullable=False)
    sem = db.Column(db.String(50), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    is_approved = db.Column(db.Boolean, default=False)
    role = db.Column(db.String(50), nullable=False, default='student')

    def get_id(self):
        return f'student-{self.id}'

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'full_name': self.full_name,
            'stream': self.stream,
            'sem': self.sem,
            'is_approved': self.is_approved,
            'role': self.role
        }

    def __repr__(self):
        return f'<Student {self.username}>'