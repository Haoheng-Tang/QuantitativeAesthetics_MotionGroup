from flask import Flask
app = Flask(__name__)

from flask import jsonify

@app.route('/person/')
def hello():
    return jsonify({'name':'Jimit',
                    'address':'India'})