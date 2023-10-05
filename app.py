from flask import Flask, request
from flask_cors import CORS
import json
from ultrasound import *

app = Flask(__name__)

CORS(app)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if request.method == 'POST':
        print("request files", request.files)
        file = request.files['file']
        original_img, predicted_img = predict(img_path, device)
        merged_img = merge_img(original_img, predicted_img)
        path_to_save_file = ''
        save_img(merged_img, path_to_save_file)
        return json.dumps({"saved_file_path": path_to_save_file})
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_path = '/Users/bishwakarki/Downloads/ultrasound/Polymyositis_2.png'
        original_img, predicted_img = predict_(img_path, device)
        save_img(predicted_img, 'predicted_2.png')

if __name__ == "__main__":
    app.run(debug=True)