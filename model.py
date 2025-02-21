import pandas as pd
import numpy as np
import random
from faker import Faker

fake = Faker()
np.random.seed(42)

# Generate random data
num_records = 10000

data = {
    "TransactionID": np.arange(1, num_records + 1),
    "Date": [fake.date_time_this_decade() for _ in range(num_records)],
    "Amount": np.round(np.random.uniform(1, 5000, num_records), 2),
    "CustomerID": np.random.randint(1000, 5000, num_records),
    "MerchantID": np.random.randint(200, 1000, num_records),
    "Location": np.random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Miami"], num_records),
    "TransactionType": np.random.choice(["Online", "ATM", "POS", "Mobile", "Wire Transfer"], num_records),
    "FraudFlag": np.random.choice([0, 1], num_records, p=[0.60, 0.40])  # 5% fraud cases
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv("fraud_transactions.csv", index=False)
print("Dataset created successfully!")
