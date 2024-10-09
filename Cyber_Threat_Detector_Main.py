import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('data/raw/network_traffic.csv')
print(data.head())
print(data.info())
print(data['label'].value_counts())
# Handling missing values
data = data.dropna()
data = data.drop_duplicates()

# Feature engineering for IP addresses can be complex; for simplicity, consider hashing or encoding
data['source_ip'] = data['source_ip'].apply(lambda x: hash(x) % 100000)
data['destination_ip'] = data['destination_ip'].apply(lambda x: hash(x) % 100000)
data['packet_size'] = data['packet_size'].astype(float)
data['time_diff'] = data['timestamp'].diff().fillna(0)
data = data.drop(['timestamp', 'other_irrelevant_columns'], axis=1)
scaler = StandardScaler()
feature_columns = ['packet_size', 'time_diff', 'source_ip', 'destination_ip']
data[feature_columns] = scaler.fit_transform(data[feature_columns])
X = data[feature_columns]
y = data['label']  # 0 for benign, 1 for malicious
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Evaluation
print("Logistic Regression Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr)}")
print(f"Precision: {precision_score(y_test, y_pred_lr)}")
print(f"Recall: {recall_score(y_test, y_pred_lr)}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr)}")
from sklearn.ensemble import RandomForestClassifier

# Initialize and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation
print("Random Forest Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"Precision: {precision_score(y_test, y_pred_rf)}")
print(f"Recall: {recall_score(y_test, y_pred_rf)}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf)}")
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input_dim = X_train.shape[1]
encoding_dim = 14  # You can experiment with this

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='relu')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_data=(X_test, X_test),
    verbose=1
)

# Plot training & validation loss
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Autoencoder Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
# Reconstruction error
X_test_pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)
threshold = np.percentile(mse, 95)  
y_pred_ae = (mse > threshold).astype(int)
# Evaluation
print("Autoencoder Anomaly Detection Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ae)}")
print(f"Precision: {precision_score(y_test, y_pred_ae)}")
print(f"Recall: {recall_score(y_test, y_pred_ae)}")
print(f"F1-Score: {f1_score(y_test, y_pred_ae)}")
combined_predictions = y_pred_rf | y_pred_ae  # Logical OR to flag if either model detects anomaly

# Evaluation
print("Combined Model Metrics:")
print(f"Accuracy: {accuracy_score(y_test, combined_predictions)}")
print(f"Precision: {precision_score(y_test, combined_predictions)}")
print(f"Recall: {recall_score(y_test, combined_predictions)}")
print(f"F1-Score: {f1_score(y_test, combined_predictions)}")
from sklearn.metrics import confusion_matrix, classification_report

conf_matrix = confusion_matrix(y_test, combined_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Combined Model')
plt.show()
from sklearn.metrics import roc_curve, auc

# For Random Forest
y_prob_rf = rf_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'Random Forest ROC curve (area = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()



