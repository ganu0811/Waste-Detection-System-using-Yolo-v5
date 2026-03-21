from WasteDetection.logger import logging
from WasteDetection.pipeline.training_pipeline import TrainingPipeline
from WasteDetection.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request,jsonify, render_template, Response
from flask_cors import CORS, cross_origin
from WasteDetection.constant.application import APP_HOST, APP_PORT
import os
import sys


app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        
        

@app.route("/train")
def trainroute():
    obj = TrainingPipeline()
    obj.run_pipeline()
    return "Training Completed"


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ['POST', 'GET'])
@cross_origin()

def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)
        
        os.system("cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source ../data/inputImage.jpg")
        
        
        opencodedbase64 = encodeImageIntoBase64("yolov5/runs/detect/exp/inputImage.jpg" )
        result = {"image": opencodedbase64.decode('utf-8')}
        os.system("rm -f yolov5/runs")
        
    except ValueError as val:
        print(val)
        return Response("Value not found inside json data")
    
    except KeyError:
        return Response("key value error incorrect key passed")
    
    except Exception as e:
        print(e)
        result = "Invalid Input"
    
    return jsonify(result)            

# For live camera

@app.route("/live", methods= ['GET'])
@cross_origin()

def predictlive():
    try:
        os.system("cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source 0")
        os.system("rm -rf yolov5/runs")
        
        return "Camera Starting"
    
    
    except ValueError as val:
        print(val)
        return Response("Value not found inside json data")


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host=APP_HOST, port=APP_PORT, debug=True)