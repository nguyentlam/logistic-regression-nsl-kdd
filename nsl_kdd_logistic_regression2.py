import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score 
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('KDDTrain+.txt', header=None)

# Add column names (adjust these names according to the actual dataset)
columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
           "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
           "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
           "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
           "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
           "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
           "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
           "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
           "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "level"]

data.columns = columns

# Encode categorical features using OrdinalEncoder
encoder = OrdinalEncoder()
data[['protocol_type', 'service', 'flag']] = encoder.fit_transform(data[['protocol_type', 'service', 'flag']])

# Encode the labels (normal or attack)
data['label'] = data['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Split data into features and labels
X = data.drop(['label', 'level'], axis=1)
y = data['label']

print(X.head())
print(y.head())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values
# scaler = StandardScaler()
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Encoded categories:")
print(encoder.categories_)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=10000, verbose=1)
# model = LogisticRegression(solver='liblinear', tol=0.00000005, max_iter=10000, verbose=1)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_true=y_test, y_pred=y_pred)
recall = recall_score(y_true=y_test, y_pred=y_pred)
f1score = f1_score(y_true=y_test, y_pred=y_pred)
report = classification_report(y_test, y_pred)


print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1_score: {f1score}')
# print(f'Classification Report:\n{report}')



# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')

# Save the figure
plt.savefig('ROC_curve.png')

plt.show()


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
# Define annotations for TP, TN, FP, FN
annotations = [['TN', 'FP'], ['FN', 'TP']]

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])

# Add annotations to the cells
# for i in range(2):
#     for j in range(2):
#         plt.text(j+0.5, i+0.5, annotations[i][j], ha='center', va='center', color='red', fontsize=12)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# Save the confusion matrix figure
plt.savefig('Confusion_Matrix.png')

# Show the plot
plt.show()