# Automated Function Point Estimation (AFP + ABCART)

This project implements an **Automated Function Point Estimation System** using:
- **AdaBoost + CART (Classification and Regression Tree)** for effort prediction.
- **Flask** as a backend API to estimate software development effort.
- **A web-based frontend** to upload Python scripts and get estimates.

## Features
- **Function Point Extraction:** Analyzes Python code to count function definitions.
- **Effort Estimation Model:** Uses **AdaBoost with Decision Trees** to predict software effort in hours.
- **REST API & Web Interface:** Allows users to upload Python scripts for estimation.
- **Automatic Browser Launch:** Opens the web app on startup.

---

## Installation
To run this project, you need **Python 3.8+** installed.

### **1. Clone the Repository**
```sh
 git clone <repository-url>
 cd <repository-folder>
```

### **2. Install Dependencies**
Run the following command to install required libraries:
```sh
 pip install -r requirements.txt
```

If you don't have `requirements.txt`, install dependencies manually:
```sh
 pip install numpy pandas flask scikit-learn
```

or, if working with virtual environment venv:

```sh
pip install --no-cache-dir pandas numpy scikit-learn flask
```


---

## Usage
### **1. Start the Server**
```sh
 python app.py
```
- This will start a Flask web server at `http://127.0.0.1:5000/`
- The browser will **automatically open** the web interface.

### **2. Upload a Python File**
- Select a **`.py` file** from your system.
- Click the **Estimate** button.
- The web app will display:
  - **Extracted Function Points**
  - **Estimated Development Effort (in hours)**

---

## API Usage (Optional)
If you prefer using the API directly, you can send a **POST request**:

```sh
curl -X POST -F "file=@your_script.py" http://127.0.0.1:5000/estimate
```

### Example Response
```json
{
  "FunctionPoints": 10,
  "EstimatedEffortHours": 45.6
}
```

---

## Folder Structure
```
project/
│── app.py              # Main Flask API and frontend
│── templates/
│   ├── index.html      # Web UI for file upload
│── static/             # (Optional) Add CSS/JS files here
│── requirements.txt    # Dependencies (Flask, scikit-learn, etc.)
```

---

## Future Enhancements
- **Improve ML Model:** Add more features like lines of code (LOC), cyclomatic complexity.
- **Support More Languages:** Extend function point extraction beyond Python.
- **Database Integration:** Store historical estimates for better predictions.

---

## License
This project is licensed under the **Apache License 2.0**.

---

