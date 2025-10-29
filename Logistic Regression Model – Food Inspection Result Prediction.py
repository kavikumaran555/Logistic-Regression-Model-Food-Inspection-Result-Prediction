import pandas
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot

data = pandas.read_csv('food inspection data.csv')
print(data)

x = data[[ 'Restaurant_Age','Employees_Count','Average_Customers_Per_Day','Cleanliness_Score','Previous_Violations']]
y = data['Inspection_Result']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)
model = LogisticRegression()
model.fit(x_train,y_train)
predicted_y = model.predict(x_test)
print(predicted_y)

cm = confusion_matrix(y_test,predicted_y)
chart = ConfusionMatrixDisplay(confusion_matrix = cm)

chart.plot(cmap='Blues')
pyplot.text(0, 0.2, 'False & Predicted Correct', ha='center', va='center', color='white', fontsize=10)
pyplot.text(1, 1.2, 'True & Predicted Correct', ha='center', va='center', color='white', fontsize=10)
pyplot.text(0, 1.2, 'False & Predicted Wrong', ha='center', va='center', color='black', fontsize=10)
pyplot.text(1, 0.2, 'True & Predicted Wrong', ha='center', va='center', color='black', fontsize=10)
pyplot.title('Confusion Matrix with Labels')
pyplot.show()

print(classification_report(y_test,predicted_y))