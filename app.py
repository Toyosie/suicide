#importing relevant libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

#load the encryped model
model = pickle.load(open('booster.pkl', 'rb'))

app = Flask(__name__)

#home page
@app.route('/')
def home():
    return render_template ('home.html')

#reading fill.html file
@app.route('/form')
def main():
    return render_template ('fill.html')

#getting the values from the form and predicting using those values
@app.route('/predict', methods=['POST'])
def form():
    data9 = request.form['radio1'] #gender
    data4 = request.form['E'] #sexuality￼
    data1 = request.form['B'] #age
    data8 = request.form['I'] #income
    data2 = request.form['C'] #race
    data3 = request.form['D'] #bodyweight
    data10 = request.form['radio2'] #virgin
    data5 = request.form['F'] #friends
    data11 = request.form['radio3']#social fear
    data12 = request.form['radio4'] #depressed
    data7 = request.form['H'] #job
    data6 = request.form['G'] #edu

    arr = np.array([[data9, data4, data1, data8, data2, data3, data10, data5, data11, data12, data7, data6]])
    pred = model.predict(arr)
    return render_template('predict.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)
