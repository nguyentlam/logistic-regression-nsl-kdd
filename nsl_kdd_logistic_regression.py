from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def parseNumber(s):
    try:
        return float(s)
    except ValueError:
        return s

data_train = np.loadtxt('./KDDTrain+.txt', dtype =object, delimiter=',', encoding='latin1', converters=parseNumber)
data_test = np.loadtxt('./KDDTest+.txt', dtype =object, delimiter=',', encoding='latin1', converters=parseNumber)
print('len(data_train)', len(data_train))
print('len(data_test)', len(data_test))

X_train_raw = data_train[:, 0:41]
y_train_raw = data_train[:, [41]]
print('X_train_raw[0:3]===========', X_train_raw[0:3])
print('y_train_raw[0:5]===========', y_train_raw[0:5])
print('=================')

X_test_raw = data_test[:, 0:41]
y_test_raw = data_test[:, [41]]
print('X_test_raw[0:3]===========', X_test_raw[0:3])
print('y_test_raw[0:3]===========', y_test_raw[0:3])
print('=================')

x_columns = np.array(list(range(41)))
print('x_columns', x_columns)
categorical_x_columns = np.array([1, 2, 3])
numberic_x_columns = np.delete(x_columns, categorical_x_columns)
print('numberic_x_columns', numberic_x_columns)
x_ct = ColumnTransformer(transformers = [("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_x_columns),
                                         ('normalize', Normalizer(norm='l2'), numberic_x_columns)], remainder = 'passthrough')

x_ct.fit(np.concatenate((X_train_raw, X_test_raw), axis = 0))
X_train = x_ct.transform(X_train_raw)
#X_train = X_train.astype('float')
print('X_train[0:3]', X_train[0:3])
print('len(X_train[0])', len(X_train[0]))

X_test = x_ct.transform(X_test_raw)
#X_train = X_train.astype('float')
print('X_train[0:3]', X_train[0:3])
print('len(X_train[0])', len(X_train[0]))

categorical_y_columns = [0]
print('categorical_y_columns', categorical_y_columns)
y_ct = ColumnTransformer(transformers = [("label", OrdinalEncoder(), categorical_y_columns)], remainder = 'passthrough')

y_ct.fit(np.concatenate((y_train_raw, y_test_raw), axis = 0))
y_train = y_ct.transform(y_train_raw)
y_train = y_train.astype('float')
print('y_train[0:2]===', y_train[0:2])
# (name, enc, _columns) = y_ct.transformers_[0]
# print(enc.categories_)

y_test = y_ct.transform(y_test_raw)
y_test = y_test.astype('float')
print('y_test[0:2]===', y_test[0:2])

# Split the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_transformed, test_size=0.2, random_state=42)

# Train a logistic regression classifier on the training set
clf = LogisticRegression(penalty=None, C=1e-6, solver='saga', multi_class='ovr', max_iter = 100)
#clf = DecisionTreeClassifier()
clf.fit(X_train, y_train.ravel())

# Use the trained classifier to predict the classes of the test set
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)