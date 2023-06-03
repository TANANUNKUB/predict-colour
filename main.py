from flask import Flask, send_from_directory
from predict_colour import PREDICT_COLOUR
import os

pred = PREDICT_COLOUR(onnx_file="models/best.onnx")
app = Flask(__name__)

global everything_is_red_panda
everything_is_red_panda = True

@app.route('/public/<path:path>')
def send_report(path):
    return send_from_directory('public', path)

@app.route('/images/<path:path>')
def send_report_images(path):
    return send_from_directory('images', path)

@app.route('/')
def index():
    result = []
    for i,j in enumerate(os.listdir('images')):
        try:
            error = False
            img = pred(f'images/{j}', f'public/output{i+1}.jpg', everything_is_red_panda)
        except Exception as e:
            print(e)
            error = True    
            img = e
        if not error:
            result.append(f"<center style='margin-top:20px;'><img style='width:40vw;' src=images/{j} /><img style='width:40vw; margin-left:10px' src={img} /></center>")
        else:
            result.append(f"<center style='margin-top:20px; display:flex;'><img style='width:40vw;' src=images/{j} /><div style='width:40vw; margin-left:10px'>{img}</div></center>")
    return "".join(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=1200)
