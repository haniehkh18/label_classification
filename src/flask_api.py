import os
from pathlib import Path
# import joblib
from flask import Flask, request
from svm_reproduce import get_label

APP = Flask(__name__)


@APP.route('/test_api', methods=['POST', 'GET'])
def predict():
    print("* " * 10, request.get_json())
    INPUT_PATH = request.get_json()['input_text']
    label = \
        get_label(input_path=INPUT_PATH)

    return {'label': label}


if __name__ == '__main__':
    APP.run(debug=False, host='0.0.0.0', port=5602, threaded=True)
