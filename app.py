# Import libraries
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('dt_model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    init_features = [float(request.form['car_wt']),
                     request.form.get('cut'),
                     request.form.get('color'),
                     request.form.get('clarity'),
                     request.form.get('polish'),
                     request.form.get('symmetry')]
    print(init_features)
    final_features = [np.array(init_features)]
    predictions = model.predict(final_features)

    output = int(predictions[0])

    return render_template('index.html', prediction_text = "The price is {}".format(output))


if __name__=='__main__':
    app.run(debug = True)