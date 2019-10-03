from flask import Flask
import os


ALLOWED_EXTENSIONS = set(['csv'])
UPLOAD_FOLDER = os.path.join('static', 'uploads')
cwd = os.getcwd()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from No_Show_Patience_ElasticBeanstalk import views