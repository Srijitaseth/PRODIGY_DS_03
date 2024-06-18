import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


directory_path = '/Users/srijitaseth/Downloads/bank+marketing'
extracted_files = os.listdir(directory_path)
print("Files in the directory:", extracted_files)
csv_file_path = os.path.join(directory_path, 'bank-additional/bank-additional-full.csv')  # Adjust this if the file name is different
df = pd.read_csv(csv_file_path, delimiter=';')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
df = pd.get_dummies(df, drop_first=True)
print(df.head())

X = df.drop(columns=['y_yes'])
y = df['y_yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Display the classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
try:
    import graphviz
    dot_data = export_graphviz(clf, out_file=None, 
                               feature_names=X.columns,  
                               class_names=['No', 'Yes'],  
                               filled=True, rounded=True,  
                               special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph.render("decision_tree")
    graph.view()
except ImportError:
    
    tree_text = export_text(clf, feature_names=list(X.columns))
    print(tree_text)
