
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import numpy as np
import pandas as pd
import pickle
#import pandas as pd

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            # #  reading the inputs given by the user

            Age = str(request.form['Age'])
            Occupation = int(request.form['Occupation'])
            City = str(request.form['City'])
            City_Stay = str(request.form['City_Stay'])

            Category1 = int(request.form['Category1'])
            Category2 = float(request.form['Category2'])

            Purchase = float(request.form['Purchase'])
            Marital_Status = int(request.form['Marital_Status'])

            # Loading the saved models into memory
            filename_scaler = 'scaler_model.pickle'
            filename = 'KnnClassifier_model.pickle'
            filename_encoder = 'OH_encoder.pickle'

            encoder = pickle.load(open(filename_encoder,'rb'))
            scaler_model = pickle.load(open(filename_scaler, 'rb'))
            loaded_model = pickle.load(open(filename, 'rb'))

            encoded_data = encoder.transform([[Occupation,Age, City,Category1, Category2,City_Stay]]).toarray()

            # predictions using the loaded model file

            dft = pd.concat([pd.DataFrame([[Purchase,Marital_Status]]), pd.DataFrame(encoded_data)], sort=False, axis=1)


            scaled_data = scaler_model.transform(dft)
            prediction = loaded_model.predict(scaled_data)
           # print('prediction is', prediction[0])
            if prediction[0] == 1:
                result = 'Man'
            else:
                result = 'Woman'




            print('prediction is', prediction[0])
            # showing the prediction results in a UI



            return render_template('results.html',prediction=result)


        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app