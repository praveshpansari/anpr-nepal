#import Flask
from flask import Flask, render_template, request
from model.main import predictPlate
from tensorflow.keras import models
import time

# create an instance of Flask
app = Flask(__name__)
app.jinja_env.cache = {}


def load_model():
    global model
    model = models.load_model('model/trained')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict/', methods=['POST'])
def predict():

    if request.method == "POST":

        # get form data
        start_time = time.time()
        image = request.files['image']

        if(image):
            image.save('static/images/vehicle.'+image.filename.split('.')[-1])

            try:
                (number, line, predict_time, color, color_time, confidence) = predictPlate('vehicle.' +
                                                                                     image.filename.split('.')[-1], None, model)

                color_info = {
                    'red': ('Private', '#FF0000'),
                    'yellow': ('National Coorporation', '#FFFF00'),
                    'black': ('Public', '#000000'),
                    'green': ('Tourist', '#00FF00'),
                    'white': ('Government', '#FFFFFF'),
                    'blue': ('Diplomatic''#0000FF')
                }

                if line > 1:
                    orientation = 'Back'
                else:
                    orientation = 'Front'

                prediction = {
                    'number': number,
                    'color': color,
                    'vtype': color_info[color][0],
                    'hex': color_info[color][1],
                    'predict_time': round(predict_time, 2),
                    'color_time': round(color_time, 2),
                    'total_time': round(time.time() - start_time, 2),
                    'confidence': confidence,
                    'orientation': orientation}

                return render_template('index.html', prediction=prediction)
            except Exception as e:
                print(e.args)
                return render_template('index.html', err='Error! The image probably has no license plate.')
        else:
            return render_template('index.html', err='Please upload image')
    pass


if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run(debug=True)
