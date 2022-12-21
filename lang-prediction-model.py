# Importing necessary Python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix

# Structuring the dataset
ling_data = pd.read_csv('/content/ling_data.csv', header=None)
ling_data.head()

# Understanding the dataset
ling_data.shape
ling_data.describe()

ling_data[6].value_counts()

# Visualizing the dataset
x = ling_data[0]
y = ling_data[6]

plt.scatter(x, y, c=y, cmap='Wistia')
plt.title('Scatter PLot of Logistic Regression')
plt.show()

#Splitting the dataset
x = ling_data.drop(columns=6, axis=1)
y = ling_data[6]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)

model = LogisticRegression()
model.fit(x_train, y_train)

x_train_predict = model.predict(x_train)
training_data_acc = accuracy_score(x_train_predict, y_train)
print(f'Training data accuracy: {training_data_acc}')

input_data = str(input(''))
input_data_as_array = np.asarray(input_data)
input_data_reshape = input_data_as_array.reshape(1, -1)
predict = model.predict(input_data_reshape)

if predict[0] == '0':
  print('Intendency to become multilingual')
else:
  print('Tendency to become multilingual')

#Visualizing with confusion matrix
disp = plot_confusion_matrix(model, x_test, y_test, cmap='Wistia', values_format='.3g')
disp.confusion_matrix
