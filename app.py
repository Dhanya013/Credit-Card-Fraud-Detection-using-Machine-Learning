from flask import Flask, render_template, session, redirect, url_for
import pandas as pd
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from flask_mail import Message, Mail
import numpy as np  
import joblib

def return_prediction(model, scaler, sample_json):
    ft_a = sample_json['distance_from_home']
    ft_b = sample_json['distance_from_last_transaction']
    ft_c = sample_json['ratio_to_median_purchase_price']
    ft_d = sample_json['repeat_retailer']
    ft_e = sample_json['used_chip']
    ft_f = sample_json['used_pin_number']
    ft_g = sample_json['online_order']

    columns = ['distance_from_home',
               'distance_from_last_transaction',
               'ratio_to_median_purchase_price',
               'repeat_retailer',
               'used_chip',
               'used_pin_number',
               'online_order']

    transaction = [[ft_a, ft_b, ft_c, ft_d, ft_e, ft_f, ft_g]]
    transaction = pd.DataFrame(transaction, columns=columns)
    transaction = scaler.transform(transaction)
    transaction = pd.DataFrame(transaction, columns=columns)

    classes = np.array(['not fraudulent', 'fraudulent'])
    class_ind = model.predict(transaction)
    class_ind = class_ind[0]

    return classes[int(class_ind)]

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

app.config["MAIL_SERVER"] = 'smtp.gmail.com'
app.config["MAIL_PORT"] = 587
app.config["MAIL_USERNAME"] = 'dhanyanaik013@gmail.com'
app.config["MAIL_PASSWORD"] = 'ekjf rwiz ngby hkrf'
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USE_SSL"] = False
app.config["MAIL_DEFAULT_SENDER"] = 'dhanyanaik013@gmail.com'

mail = Mail(app)

# LOAD THE MODEL AND THE SCALER!
model = joblib.load("agada_credit_card_fraud_prediction.sav")
scaler = joblib.load("agada_credit_card_fraud_prediction_scaler.pkl")

def send_email_notification(transaction_details):
    msg = Message('Fraudulent Transaction Detected', recipients=['recipient-email@example.com'])
    msg.body = f"A fraudulent transaction has been detected with the following details:\n{transaction_details}"
    mail.send(msg)

class TransactionForm(FlaskForm):
    ft_1 = StringField('Distance from home', validators=[DataRequired()])
    ft_2 = StringField('Distance from location of last transaction', validators=[DataRequired()])
    ft_3 = StringField('Ratio to median purchase price', validators=[DataRequired()])
    ft_4 = StringField('Repeat retailer? Enter 1 if yes, 0 otherwise', validators=[DataRequired()])
    ft_5 = StringField('Used chip? Enter 1 if yes, 0 otherwise', validators=[DataRequired()])
    ft_6 = StringField('Used pin number? Enter 1 if yes, 0 otherwise', validators=[DataRequired()])
    ft_7 = StringField('Online transaction? Enter 1 if yes, 0 otherwise', validators=[DataRequired()])

    submit = SubmitField('Predict')

@app.route('/about', methods=['GET', 'POST'])
def index():
    form = TransactionForm()
    if form.validate_on_submit():
        session['ft_1'] = form.ft_1.data
        session['ft_2'] = form.ft_2.data
        session['ft_3'] = form.ft_3.data
        session['ft_4'] = form.ft_4.data
        session['ft_5'] = form.ft_5.data
        session['ft_6'] = form.ft_6.data
        session['ft_7'] = form.ft_7.data
        return redirect(url_for('prediction'))
    return render_template('about.html', form=form)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/post')
def post():
    return render_template('post.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'ft_1' in session:
        content = {
            'distance_from_home': float(session['ft_1']),
            'distance_from_last_transaction': float(session['ft_2']),
            'ratio_to_median_purchase_price': float(session['ft_3']),
            'repeat_retailer': float(session['ft_4']),
            'used_chip': float(session['ft_5']),
            'used_pin_number': float(session['ft_6']),
            'online_order': float(session['ft_7'])
        }

        results = return_prediction(model=model, scaler=scaler, sample_json=content)
        if results == 'fraudulent':
            send_email_notification(content)

        return render_template('prediction.html', results=results)
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
