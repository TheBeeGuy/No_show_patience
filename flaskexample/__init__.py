from flask import Flask
import os


ALLOWED_EXTENSIONS = set(['csv'])
UPLOAD_FOLDER = './flaskexample/uploads'
cwd = os.getcwd()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from flaskexample import views