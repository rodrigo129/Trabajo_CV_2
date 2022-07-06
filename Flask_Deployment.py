import os
from PIL import Image
from NetworkModel import NetworkModel
import io
from Transform import GetTransform
from flask import Flask, flash, request, redirect, url_for, jsonify
import typer
from pathlib import Path
from Dataset_Flores_Class import get_labels

model = NetworkModel()

classes = get_labels()
# Instance Flask
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = './Archivos subidos'
shell = typer.Typer()


def allowed_file(filename):
    """
    function that chek if a file is an image that is allowed for the page
    :param filename:
    :return:
    """
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def transform_image(image_bytes):
    """
    function that transform an image in a bytes format to a torch tensor representation

    :rtype: image in a format that the neural network can interpretate
    """
    transform = GetTransform()
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)


def get_prediction(image_tensor):
    """
    function that use the neural network moder to predict which flower is in the picture
    :rtype: index of the class more likely for the prediction and probability for each class
    """
    predictions = model.model(image_tensor)
    predictions = predictions.cpu().detach().numpy()
    more_likely = int(predictions.argmax())
    probability_per_class = predictions.tolist()
    return [more_likely, probability_per_class]


@app.route("/")
def upload_file():
    """
    function that flask has associated with the root path of the page
    :return: html template for the root page
    """
    return """<form action="/predict" method="POST" enctype="multipart/form-data">
 <input type="file" name="file">
 <input type="submit">
</form>"""


@app.route('/predict', methods=['POST'])
def predict():
    print(request)
    if request.method == 'POST':
        print(request.values)

        file = request.files.get("file")

        print(file)

    if file is None or file.filename == "":
        return jsonify({'status': False, 'msj': 'no file'})
    if not allowed_file(file.filename):
        return jsonify({'status': False, 'msj': 'format not supported'})
    try:
        img_bytes = file.read()

        tensor = transform_image(img_bytes)

        prediction = get_prediction(tensor)
        probability_per_class = prediction[1]
        more_likely = prediction[0]
        return jsonify({'status': True, 'data': {'classes': classes, 'class_index_more_likely': more_likely,
                                                 'probability_per_class': probability_per_class[0]}})
    except:
        return jsonify({'status': False, 'msj': 'error during prediction'})


def main(ip: str = typer.Option("127.0.0.1", help="ip where the REST API will run"),
         port: int = typer.Option(8080, help="Port for the REST API"),
         check_point: Path = typer.Option(Path(os.path.join('', 'log_final')), help="file or folder for the checkpoint "
                                                                                    "file")):
    """
    Function Dedicate to load the neural network model and initiating the flask instance
    :param ip: ip for the flask server
    :param port: por for the flask server
    :param check_point: checkpoint file or folder containing the checkpoint file
    :return:
    """
    global model
    if check_point.is_file():
        model = NetworkModel.load_from_checkpoint(os.fspath(check_point))
        print('load_file')
    elif check_point.is_dir():
        print('is folder')
        file_list = sorted(check_point.rglob("*.ckpt"))
        if len(file_list) == 0:
            print("no checkpoint file found")
            return
        else:
            print(f"{file_list[0]} found")
            check_point = os.fspath(file_list[0])
            model = NetworkModel.load_from_checkpoint(check_point)
    model.model.eval()

    app.run(debug=False, host=ip, port=port)


if __name__ == '__main__':
    # Iniciamos la aplicación
    typer.run(main)
