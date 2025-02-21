import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # Handle class imbalance
from xgboost import XGBClassifier
import pickle

# ✅ Step 1: Load & Preprocess Data
df = pd.read_csv("fraud_transactions.csv")

# ✅ Keep only the necessary features (Including CustomerID & MerchantID)
df = df[['Amount', 'CustomerID', 'MerchantID', 'TransactionType', 'Location', 'FraudFlag']]

# ✅ Convert categorical columns using Label Encoding
encoder = LabelEncoder()
df["TransactionType"] = encoder.fit_transform(df["TransactionType"])
df["Location"] = encoder.fit_transform(df["Location"])

# ✅ Save cleaned data
df.to_csv("fraud_transactions_cleaned.csv", index=False)
print("✅ Data cleaned and saved successfully!")

# ✅ Step 2: Load Cleaned Dataset
df = pd.read_csv("fraud_transactions_cleaned.csv")

# ✅ Step 3: Split Features and Target
X = df[['Amount', 'CustomerID', 'MerchantID', 'TransactionType', 'Location']]  # ✅ Include all 5 features
y = df["FraudFlag"]  # Target variable

# ✅ Step 4: Handle Class Imbalance with SMOTE
print("🔍 Checking class distribution before SMOTE:")
print(y.value_counts())

if y.value_counts()[1] < 10:
    print("❌ Not enough fraud cases! Skipping SMOTE.")
    X_resampled, y_resampled = X, y  # Use original data
else:
    print("✅ Applying SMOTE to balance dataset...")
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("✅ SMOTE applied successfully!")

# ✅ Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ✅ Step 6: Train XGBoost Model
print("🚀 Training XGBoost model with updated features...")
model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# ✅ Step 7: Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ✅ Step 8: Save Model
with open("fraud_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained successfully with updated features and saved as fraud_model.pkl")
