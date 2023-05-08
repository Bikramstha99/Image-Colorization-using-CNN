from flask import Flask,request, url_for, redirect, render_template,request,flash
from werkzeug.utils import secure_filename
from main import getPrediction
import os
import cv2
from skimage.color import rgb2lab,lab2rgb,gray2rgb
from skimage.io import imsave

UPLOAD_FOLDER='web pic/black and white image/'
#UPLOAD_FOLDER1 ='static/img/'


app = Flask(__name__,static_folder="static", template_folder="templates")

app.secret_key ="secret key"

app.config['UPLOAD_FOLDER'] =UPLOAD_FOLDER


@app.route('/')
def hello_world():
    return render_template("index1.html")

@app.route('/',methods=['POST'])
def submit_file():
   if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)  #Use this werkzeug method to secure filename. 
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            flash('static/savedbw/img.jpg')
            #getPrediction(filename)
            label=getPrediction(filename)
            #imsave('C:/Users/Acer/OneDrive/Desktop/check/static/img/img1.jpg',label)
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'],label))
            #colorfilename=file.save(os.path.join(app.config['UPLOAD_FOLDER'],label))
            flash('static/savedcolor/img1.jpg')
            return redirect('/colorimage')

@app.route('/colorimage')
def bikram():
    return render_template("response.html")

@app.route('/About')
def about():
    return render_template("about.html")

@app.route('/team')
def team():
    return render_template("ourteam.html")

@app.route('/Services')
def services():
    return render_template("services.html")

           

if __name__ == '__main__':
    app.run(port=8000)