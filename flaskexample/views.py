from flask import render_template, request
from flaskexample import app
from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
import psycopg2
import os
import featuretools as ft
import featuretools.variable_types as vtypes
import pandas as pd
import pickle
from dotenv import load_dotenv
import numpy as np

cwd = os.getcwd()

##read in secret keys stored in .env file
load_dotenv('.env')

SECRET = os.environ['dark_sky_secret']
LAT = os.environ['LAT']
LONG = os.environ['LONG']


user = os.environ['username'] #add your username here (same as previous postgreSQL)
host = 'localhost'
dbname = 'dental_predictions'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = psycopg2.connect(database = dbname, user = user)




@app.route('/',  methods=['GET', 'POST'])
def index():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'dental_cartoon.jpg')
    return render_template("index.html", user_image = full_filename)

@app.route('/cover',  methods=['GET', 'POST'])
def cover():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'dental_cartoon.jpg')
    return render_template("cover.html", user_image = full_filename)

@app.route('/tradeoff', methods=['GET', 'POST'])
def tradeoff():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'dental_cartoon.jpg')
    return render_template("tradeoff.html", user_image = full_filename)


@app.route('/uploader', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        full_dataset = pd.read_csv(request.files.get('file'))
        full_dataset.AppointmentDate = (pd.to_datetime(full_dataset.AppointmentDate) +
                                        (pd.Timedelta(days=1) - pd.Timedelta(seconds=1)) * (full_dataset.noshow))

        ##load in weather data
        weather = pickle.load(open('data/weather2018.pkl', 'rb'))
        weather = pd.DataFrame(weather)

        # ##fill NAs
        # # Explicitly impute values for missing fields for RF and Log
        # full_dataset = full_dataset.fillna(full_dataset.mean())
        # full_dataset = full_dataset.fillna(0)

        def merge_appt_weather(appointments_df, weather_df, appt_date):  # takes appointments, weather, and column of appointment date
            x = appointments_df[appt_date].dt.round('60min')
            appointments_df['datetime'] = pd.to_datetime(x)
            weather_df['datetime'] = pd.to_datetime(weather_df['time'], unit='s').dt.round('60min')
            appointments_df = appointments_df.merge(weather_df, on='datetime', how="left")
            return appointments_df
        appt_date = 'AppointmentDate'
        full_dataset = merge_appt_weather(full_dataset, weather, appt_date)

        #prepare for feature engineering using feature tools
        variable_types = {'PatientId': vtypes.Numeric, 'newbie': vtypes.Boolean, 'insuranceDummy': vtypes.Boolean,
                          'guarantorIsPatient': vtypes.Boolean, 'loyalty': vtypes.Numeric, 'noshow': vtypes.Boolean,
                          'PatientEmail': vtypes.EmailAddress, 'GuarantorEmail': vtypes.EmailAddress,
                          'PatientPhone1': vtypes.PhoneNumber, 'patientEmailDomain': vtypes.Categorical,
                          'guarantorEmailDomain': vtypes.Categorical, 'patientPhone1AreaCode': vtypes.Categorical,
                          'insurance': vtypes.Categorical, 'provider': vtypes.Categorical,
                          'source': vtypes.Categorical, 'procedure': vtypes.Categorical,
                          'weekday': vtypes.Categorical, 'apptType': vtypes.Categorical,
                          "newpatientfile": vtypes.Ordinal, "Age_npf": vtypes.Numeric, "Patient": vtypes.Text,
                          'summary': vtypes.Text, 'icon': vtypes.Text, 'precipIntensity': vtypes.Numeric,
                          'temperature': vtypes.Numeric, 'apparentTemperature': vtypes.Numeric,
                          'dewPoint': vtypes.Numeric,
                          'humidity': vtypes.Numeric, 'pressure': vtypes.Numeric, 'windSpeed': vtypes.Numeric,
                          'windGust': vtypes.Numeric, 'windBearing': vtypes.Numeric, 'cloudCover': vtypes.Numeric,
                          }
        variable_list = (['PatientId', 'AppointmentDate', 'newbie',
                          'insuranceDummy', 'guarantorIsPatient', 'loyalty', 'noshow',
                          'PatientEmail', 'GuarantorEmail', 'PatientPhone1',
                          'patientEmailDomain', 'guarantorEmailDomain', 'patientPhone1AreaCode',
                          'insurance', 'provider', 'source', 'procedure', 'weekday', 'apptType',
                          "newpatientfile", "Age_npf", "Patient", 'AppointmentId', 'summary', 'icon',
                          'precipIntensity', 'temperature',
                          'apparentTemperature', 'dewPoint', 'humidity', 'pressure', 'windSpeed',
                          'windGust', 'windBearing', 'cloudCover'])

        # Make an entity named 'appointments' which stores dataset metadata with the dataframe
        es = ft.EntitySet('Appointments')
        es = es.entity_from_dataframe(entity_id="appointments",
                                      dataframe=full_dataset[variable_list],
                                      index='AppointmentId',
                                      time_index='AppointmentDate',
                                      variable_types=variable_types)
        # Make patients, weekdays, emails and insurances entities

        es.normalize_entity(base_entity_id='appointments', new_entity_id='patients', index='PatientId',
                            additional_variables=['Patient', 'PatientEmail', 'GuarantorEmail', 'PatientPhone1',
                                                  'newpatientfile', 'Age_npf'], make_time_index=True)
        # 'PatientEmail', 'GuarantorEmail', 'PatientPhone1'
        es.normalize_entity('appointments', 'patientemails', 'patientEmailDomain', make_time_index=True)
        es.normalize_entity('appointments', 'guarantoremails', 'guarantorEmailDomain', make_time_index=True)
        es.normalize_entity('appointments', 'patientPhones', 'patientPhone1AreaCode', make_time_index=True)
        es.normalize_entity('appointments', 'insurances', 'insurance', make_time_index=True)
        es.normalize_entity('appointments', 'providers', 'provider', make_time_index=True)
        es.normalize_entity('appointments', 'sources', 'source', make_time_index=True)
        es.normalize_entity('appointments', 'procedures', 'procedure', make_time_index=True)
        es.normalize_entity('appointments', 'weekdays', 'weekday', make_time_index=True)
        es.normalize_entity('appointments', 'apptTypes', 'apptType', make_time_index=True)

        cutoff_times = es['appointments'].df[['AppointmentId', 'AppointmentDate']]
        cat_cols = ['insurance', 'provider', 'procedure', 'patientEmailDomain', 'guarantorEmailDomain',
                    'patientPhone1AreaCode', 'weekday', 'apptType', 'source']

        X_ft, features = ft.dfs(entityset=es,
                                target_entity='appointments',
                                agg_primitives=['count', 'percent_true', 'num_true', 'trend', 'skew', 'all', 'std',
                                                'max', 'mean', 'min', 'median', 'num_unique'],
                                max_depth=1,
                                trans_primitives=['day', 'month', 'year', 'num_words', 'num_characters'],
                                cutoff_time=cutoff_times,
                                verbose=True,
                                approximate='3h',
                                # approximate = "1 day",
                                chunk_size=50,
                                n_jobs=1)

        final_data = pd.concat([X_ft], axis=1).drop(cat_cols, axis=1).drop(['noshow'],axis=1)

        # insightMVP_model_simple.pkl has 1 max depth feature engineering
        pkl_file = open('insightMVP_model_simple.pkl', 'rb')
        XGBmodel = pickle.load(pkl_file)

        id = final_data['PatientId']
        y_pred_model_xgb = XGBmodel.predict_proba(final_data)[:,1]
        percentage= pd.DataFrame({'percent_noshow': y_pred_model_xgb})*1000

        results=pd.concat([id.reset_index(drop=True), percentage.reset_index(drop=True)],axis=1)

        def highlight_greaterthan(s, threshold, column):
            is_max = pd.Series(data=False, index=s.index)
            is_max[column] = s.loc[column] >= threshold
            return ['background-color: red' if is_max.any() else '' for v in is_max]

        style1 = results.style.apply(highlight_greaterthan, threshold=75, column=['percent_noshow'], axis=1)
        df_html = style1.render()

        return render_template('uploaded_csv_block.html', test= df_html)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

