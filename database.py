import mysql.connector
import pandas as pd

# Connect to MySQL
conn = mysql.connector.connect(host='localhost', user='root', password='Sivaanju@6', database='fraud_detection')
cursor = conn.cursor()

# Load dataset
df = pd.read_csv("fraud_transactions.csv")

# Insert data
for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO Transactions (TransactionID, DATETIME, Amount, CustomerID, MerchantID, Location, TransactionType, FraudFlag)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, tuple(row))

conn.commit()
cursor.close()
conn.close()
print("Data uploaded successfully!")
