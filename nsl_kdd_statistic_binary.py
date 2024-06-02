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

# Count the occurrences of each class label
label_counts = y.value_counts()

print('label_counts', label_counts)

# Calculate the ratio between normal and attack instances
normal_count = label_counts[0]
attack_count = label_counts[1]
total_count = len(data)
normal_ratio = normal_count / total_count
attack_ratio = attack_count / total_count

# Create a pie chart
labels = ['Normal', 'Attack']
# sizes = [normal_ratio, attack_ratio]
sizes = [normal_count, attack_count]
colors = ['blue', 'red']
explode = (0, 0.1)  # "explode" the second slice (i.e., 'Attack')

plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', textprops={'fontsize': 20}, shadow=True, startangle=140)
plt.title('Ratio of Normal and Attack')
# plt.text(0, 0, 'Total Count: {}'.format(total_count), fontsize=12, color='black', ha='center')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

plt.savefig('Normal_Attack_Statistic.png')

plt.show()