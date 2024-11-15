# DecisionTree
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
col_names = ['id', 'full_name', 'age', 'gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day', 'click']
df = pd.read_csv('ad_click_dataset.csv', names=col_names, header=None)
df = df.sort_values(by="id", ascending=True)
df = df.drop(['id', 'full_name'], axis=1)
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age'].fillna(df['age'].mean(), inplace=True)
df = pd.get_dummies(df, columns=['gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day'])
X = df.drop('click', axis=1)
Y = df['click']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=class_names, filled=True)
plt.show()
