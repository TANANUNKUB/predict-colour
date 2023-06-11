from flask import *
from predict_colour import PREDICT_COLOUR

app = Flask(__name__, template_folder='template')
pred = PREDICT_COLOUR(onnx_file="models/best.onnx")

@app.route('/public/<path:path>')
def send_report(path):
    return send_from_directory('public', path)

@app.route('/')
def main():
	return render_template("index.html")

@app.route('/success', methods = ['POST'])
def success():
	if request.method == 'POST':
		f = request.files['file']
		f.save("public/image.jpg")
		_ = pred('public/image.jpg', 'public/image1.jpg')
		return render_template("success.html")

if __name__ == '__main__':
	app.run(host='0.0.0.0',port=80)


