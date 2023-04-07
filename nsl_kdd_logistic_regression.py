from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def parseNumber(s):
    try:
        return float(s)
    except ValueError:
        return s

data = np.loadtxt('./KDDTest+.txt', dtype =object, delimiter=',', encoding='latin1', converters=parseNumber)
print('len(data)', len(data))
X = data[:, 0:41]
y = data[:, [41]]
print('X[0:5]===========', X[0:5])
print('y[0:5]===========', y[0:5])
print('=================')


x_columns = np.array(list(range(41)))
print('x_columns', x_columns)
categorical_x_columns = np.array([1, 2, 3])
numberic_x_columns = np.delete(x_columns, categorical_x_columns)
print('numberic_x_columns', numberic_x_columns)
x_ct = ColumnTransformer(transformers = [("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_x_columns),
                                         ('normalize', Normalizer(norm='l2'), numberic_x_columns)], remainder = 'passthrough')

X_transformed = x_ct.fit_transform(X)
#X_transformed = X_transformed.astype('float')
print('X_transformed[0:5]', X_transformed[0:5])
print('X_transformed[0]', len(X_transformed[0]))

categorical_y_columns = [0]
print('categorical_y_columns', categorical_y_columns)
y_ct = ColumnTransformer(transformers = [("label", OrdinalEncoder(), categorical_y_columns)], remainder = 'passthrough')

y_transformed = y_ct.fit_transform(y)
y_transformed = y_transformed.astype('float')
print('y_transformed[0:2]===', y_transformed[0:2])
# (name, enc, _columns) = y_ct.transformers_[0]
# print(enc.categories_)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_transformed, test_size=0.2, random_state=42)

# Train a logistic regression classifier on the training set
clf = LogisticRegression(penalty=None, C=1e-6, solver='lbfgs', multi_class='multinomial', max_iter = 100)
clf.fit(X_train, y_train.ravel())

# Use the trained classifier to predict the classes of the test set
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)