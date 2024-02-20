from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the .h5 model
model = load_model('brnn_model.h5')
def preprocess_input(frame_len, tcp_srcport, tcp_dstport, tcp_window_size):
    # Preprocess the input data as necessary
    # For example, you might normalize the input data to be between 0 and 1
    input_array = np.array([
        frame_len / 1000.0,  # Frame length in seconds
        tcp_srcport / 65535.0,  # TCP source port normalized
        tcp_dstport / 65535.0,  # TCP destination port normalized
        tcp_window_size / 65535.0  # TCP window size normalized
    ]).reshape(1, -1)

    # Add extra dimensions to match the input shape of the model
    input_array = np.repeat(input_array, 128, axis=1)
    input_array = input_array.reshape(1, -1)

    return input_array

def postprocess_output(prediction):
    # Postprocess the prediction data as necessary
    # For example, you might convert the prediction into a human-readable format
    if prediction > 0.5:
        output_text = 'The packet is likely to be a SYN packet.'
    else:
        output_text = 'The packet is likely not to be a SYN packet.'
    return output_text

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/execute', methods=['POST'])
def execute():
    # Get the user's input from the form
    frame_len = int(request.form['frame_len'])
    tcp_srcport = int(request.form['tcp_srcport'])
    tcp_dstport = int(request.form['tcp_dstport'])
    tcp_window_size = int(request.form['tcp_window_size'])

    # Preprocess the input (if necessary)
    input_array = preprocess_input(frame_len, tcp_srcport, tcp_dstport, tcp_window_size)

    # Use the model to make a prediction
    prediction = model.predict(input_array)

    # Postprocess the prediction (if necessary)
    output_text = postprocess_output(prediction)

    return render_template('result.html', input=input_array, output=output_text)

if __name__ == '__main__':
    app.run(debug=True)