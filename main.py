from flask import *
from predict_colour import PREDICT_COLOUR
from datetime import datetime
app = Flask(__name__, template_folder='template')
pred = PREDICT_COLOUR(onnx_file="models/best.onnx")

@app.route('/public/<path:path>')
def send_report(path):
    return send_from_directory('public', path)

@app.route('/')
def main():
	return render_template("index.html")

@app.route('/success', methods = ['POST', 'GET'])
def successPOST():
	if request.method == 'POST':
		f = request.files['file']
		date = str(datetime.now()).replace(" ","")
		filename = f"public/{date}.jpg"
		output = f"public/{date}_output.jpg"
		f.save(filename)
		_ = pred(filename, output)
		return render_template("success.html", image=output)
	else:
		return redirect("/",code=302)

	

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0',port=80)


