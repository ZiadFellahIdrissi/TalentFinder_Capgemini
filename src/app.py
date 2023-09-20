from flask import Flask, render_template, request, jsonify
import chatbot as cs


## ============================================================================= ## 
##          This code was made by MATTI, Manal and SIRSALANE, Keltoum
## ============================================================================= ## 
 

app = Flask(__name__)
server = app.server
 

@app.route('/', methods=['GET', 'POST'])

def index():

    if request.method == 'POST':

        description = request.form['description']

        profiles = cs.get_similar_profiles(description)

        return render_template('home.html', profiles=profiles)

    return render_template('home.html')

 


@app.route('/predict', methods=['POST'])

def predict():

    data = request.get_json()

    description = data.get('description')

    profiles = cs.get_similar_profiles(description)

    if not profiles:

        return jsonify({"message": "Sorry, there are no similar profiles."})


    return jsonify(profiles)

 

 

if __name__ == "__main__":

    app.run(debug=True)

 