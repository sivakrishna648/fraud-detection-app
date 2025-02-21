from flask import Flask, request, jsonify, render_template
import pandas as pd
import mysql.connector
import pickle

# Load trained XGBoost model
model = pickle.load(open("fraud_model.pkl", "rb"))

app = Flask(__name__)

# MySQL Connection
def get_db_connection():
    return mysql.connector.connect(host='localhost', user='root', password='Sivaanju@6', database='fraud_detection')

# Debugging: Print all available routes
@app.before_first_request
def list_routes():
    print("\n📌 Available Routes:")
    for rule in app.url_map.iter_rules():
        print(f"➡ {rule}")

# ✅ Serve Webpage (Updated Home Route)
@app.route('/')
def home():
    return render_template("index.html")  # Load the webpage

# ✅ Fraud Prediction API
# ✅ Fraud Prediction API with Error Handling
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # ✅ Handle missing input data
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # ✅ Ensure all expected features are included
        required_columns = ['Amount', 'TransactionType', 'Location', 'CustomerID', 'MerchantID']
        for col in required_columns:
            if col not in data or data[col] == "":
                return jsonify({"error": f"Missing required field: {col}"}), 400

        # ✅ Convert data types correctly
        df = pd.DataFrame([data])
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['TransactionType'] = pd.to_numeric(df['TransactionType'], errors='coerce')
        df['Location'] = pd.to_numeric(df['Location'], errors='coerce')
        df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')
        df['MerchantID'] = pd.to_numeric(df['MerchantID'], errors='coerce')

        # 🚀 Handle invalid numeric values
        if df.isnull().values.any():
            return jsonify({"error": "Invalid input values. Please enter correct data."}), 400

        # ✅ Make fraud prediction
        prediction = model.predict(df)[0]
        return jsonify({"Fraud Prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Get Transactions from MySQL
@app.route('/transactions', methods=['GET'])
def get_transactions():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM Transactions LIMIT 10000")
    transactions = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(transactions)

if __name__ == '__main__':
    app.run(debug=True)


