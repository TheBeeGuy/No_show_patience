from flask import render_template, request
from flaskexample import app
from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import os
from werkzeug.utils import secure_filename



cwd = os.getcwd()


user = 'aishagharsalli' #add your username here (same as previous postgreSQL)
host = 'localhost'
dbname = 'dental_predictions'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = psycopg2.connect(database = dbname, user = user)

@app.route('/',  methods=['GET', 'POST'])
def index():
    return render_template("index.html")


@app.route('/uploader', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'))
        return render_template('upload2.html', shape=df.shape)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

