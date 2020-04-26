import cv2
#from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, url_for

app = Flask(__name__)

#model = load_model("model/model.h5")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=["GET"])
def result():
    img = request.files["myimage"]
    img.save(secure_filename(img.filename))
    
    Img = cv2.imread(img.filename)
    Img = cv2.resize(Img, (128,128))

    res = model.predict(Img)
    
    if res == 0:
        return render_template('result.html', result = "REAL")
    else:
        return render_template('result.html', result = "FAKE")

if __name__ == '__main__':
    app.run(debug=True)
