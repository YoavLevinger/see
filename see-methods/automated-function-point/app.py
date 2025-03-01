import os
import re
import json
import numpy as np
import pandas as pd
import ast
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from flask import Flask, request, jsonify, render_template
import threading
import webbrowser


# Function to extract function points from Python code
def extract_function_points(code):
    try:
        tree = ast.parse(code)
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        return len(functions)
    except SyntaxError:
        return 0


# Load sample dataset (simulating historical software effort data)
def generate_synthetic_data():
    np.random.seed(42)
    data = {
        'FunctionPoints': np.random.randint(5, 100, 200),
        'EffortHours': np.random.randint(10, 500, 200) + np.random.randn(200) * 20
    }
    df = pd.DataFrame(data)
    return df


data = generate_synthetic_data()
X = data[['FunctionPoints']]
y = data['EffortHours']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a CART model with AdaBoost
base_model = DecisionTreeRegressor(max_depth=4)
adaboost_model = AdaBoostRegressor(base_model, n_estimators=50, random_state=42)
adaboost_model.fit(X_train, y_train)


def predict_effort(function_points):
    return adaboost_model.predict(np.array([[function_points]]))[0]


# Flask API and Frontend
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/estimate', methods=['POST'])
def estimate_effort():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    code = file.read().decode('utf-8')
    function_points = extract_function_points(code)
    effort_estimate = predict_effort(function_points)

    return jsonify({
        'FunctionPoints': function_points,
        'EstimatedEffortHours': round(effort_estimate, 2)
    })


# Open browser on startup
def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')


if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()
    app.run(debug=True)