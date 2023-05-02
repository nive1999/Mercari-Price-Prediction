from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle


app = Flask(__name__)

model = pickle.load(open('Voting_model.pkl', 'rb'))


#reg.pickle.load(open('Voting_model1.pkl', 'rb'))

# create a LabelEncoder object for categorical columns
categorical_encoder = LabelEncoder()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # get the input values from the HTML form
    name = request.form.get('name')
    category_name = request.form.get('category_name')
    brand_name = request.form.get('brand_name')
    item_description = request.form.get('item_description')
    item_condition_id = request.form.get('item_condition_id')
    shipping = request.form.get('shipping')

    # create a pandas dataframe with the input values
    data = pd.DataFrame({'name':[name],'item_condition_id':[item_condition_id], 'category_name':[category_name], 'brand_name':[brand_name], 
                         'shipping':[shipping],'item_description':[item_description]})
    categorical_columns = ["name", "category_name", "brand_name", "item_description"]
    # convert the categorical variables to numerical using label encoding
    data[categorical_columns] = categorical_encoder.fit_transform(data[categorical_columns].astype(str).values.ravel())
    print("above predict--------------------")
    data['item_condition_id'] = pd.to_numeric(data['item_condition_id'])
    data['shipping'] = pd.to_numeric(data['shipping'])
    # make predictions using the trained model
    prediction = model.predict(data)
    print("below predict--------------------")

    # format the prediction as a string
    prediction_str = f'${round(prediction[0], 2)}'

    # render the HTML template with the prediction
    return render_template('index.html', prediction_text='  Predicted Price: {}  '.format(prediction_str))

if __name__ == "__main__":
    app.run(debug=True)
