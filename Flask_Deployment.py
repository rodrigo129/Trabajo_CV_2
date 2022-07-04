import os
from PIL import Image
from Modelo import modelo
import io
from Transform import GetTransform
from flask import Flask, flash, request, redirect, url_for, jsonify

# Cargar Modelo
model = modelo.load_from_checkpoint(os.path.join('', 'log_antiguo_t2_sh1_test', 'checkpoints', 'epoch=4-step=95.ckpt'))
model.model.eval()

clases = {0:'son rosas',1:'son calas rosa',2:'son cardenales rojas',3:'son orejas de oso',4:'no se reconoce la flor'}

print('Modelo Cargado')
# Intancia Flask
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = './Archivos subidos'


def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transform_image(image_bytes):
    transform = GetTransform()

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    #images = image_tensor.reshape(-1, 28*28)
    #outputs = model(images)

    predictions = model.model(image_tensor)
    #predicted = predictions
    predicted = predictions.argmax()



        # max returns (value ,index)
    #_, predicted = torch.max(outputs.data, 1)
    return [predicted, predictions]

@app.route("/")
def upload_file():
    # renderiamos la plantilla "formulario.html"

    html = """<form action="/predecir" method="POST" enctype="multipart/form-data">
 <input type="file" name="file">
 <input type="submit">
</form>"""
    return html


@app.route('/predecir', methods=['POST'])
def predecir():
    print(request)
    if request.method == 'POST':
        print(request.values)

        file = request.files.get("file")

        print(file)

    if file is None or file.filename == "":
        return jsonify({'error': 'no file'})
    if not allowed_file(file.filename):
        return jsonify({'error': 'format not supported'})
    try:
        img_bytes = file.read()
        #(img_bytes)

        tensor = transform_image(img_bytes)
        #print (tensor)

        prediction = get_prediction(tensor)
        print (prediction)
        """
        #data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
        return jsonify(data)
        """
        return f"<h1>Imagen subida, {clases[int(prediction[0])]}</h1>"
    except:
        return jsonify({'error': 'error during prediction'})


if __name__ == '__main__':
    # Iniciamos la aplicación
    app.run(debug=True,host="192.168.10.3", port=8080)
