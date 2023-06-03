from flask import Flask, send_from_directory
from predict_colour import PREDICT_COLOUR
import os

pred = PREDICT_COLOUR(onnx_file="models/best.onnx")
app = Flask(__name__)

@app.route('/public/<path:path>')
def send_report(path):
    return send_from_directory('public', path)

@app.route('/')
def index():
    result = []
    for i,j in enumerate(os.listdir('images')):
       img = pred(f'images/{j}', f'public/output{i+1}.jpg')
       result.append(f"<center><img style='width:60vw' src={img} /></center>")
    return "".join(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=1200)
