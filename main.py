from flask import Flask, send_from_directory
from predict_colour import PREDICT_COLOUR
import os

app = Flask(__name__)

predict_colour = PREDICT_COLOUR(onnx_file="models/best.onnx")
predict_colour('images/vibrant-rooms-8-1548883440.jpg', 'public/output.jpg')

@app.route('/public/<path:path>')
def send_report(path):
    return send_from_directory('public', path)

@app.route('/')
def index():
    return '<center><img style="width:60vw" src="public/output.jpg" /></center>'

if __name__ == '__main__':
    app.run(host='0.0.0.0')
