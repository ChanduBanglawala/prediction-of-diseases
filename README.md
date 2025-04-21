# prediction-of-diseases
#in this i created a model which can predict the possibility of heart disease, Parkinson's and  diabetes

from pandas import *
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
from matplotlib.pyplot import *
import pickle
import xlrd

parkinson_data=read_csv('D:\disease prediction\parkinsons.csv')
parkinson_data

parkinson_data.shape

parkinson_data.info()

parkinson_data.isnull().sum()

x=parkinson_data.drop(columns=['name','status'], axis=1)
y=parkinson_data['status']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

print(x_train,x_test,y_train,y_test)

model=svm.SVC(kernel='linear')
model.fit(x_train,y_train)

x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('accuraccy of model on train data set:%.2f'%training_data_accuracy)

x_test_prediction=model.predict(x_test)
training_score=accuracy_score(x_test_prediction,y_test)
print('accuracy of model on test data set:%.2f'%training_score)

file_name='D:\disease prediction\parkinson.sav'
pickle.dump(model,open(file_name,'wb'))
model=pickle.load(open(file_name,'rb'))
print(model)

#HEART MODEL
from pandas import *
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
from matplotlib.pyplot import *
import pickle
heart_data=read_csv('D:\disease prediction\heart.csv')
heart_data
heart_data.info()
heart_data.shape
heart_data.head()
heart_data.tail()
heart_data.isnull().sum()
heart_data.describe()
heart_data['target'].value_counts()
bar(heart_data['target'],heart_data['chol'])
xlabel('0 means no heart_attack,1 means heart_attack')
ylabel('cholestrol')
show()
bar(heart_data['target'],heart_data['trestbps'])
xlabel('0 means no heart_attack,1 means heart_attack')
ylabel('blood presssure')
show()
x=heart_data.drop(columns='target',axis=1)
print(x.head())
y=heart_data['target']
print(y.head())
spliiting data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
print(x_train,x_test,y_train,y_test)
model=svm.SVC(kernel='linear')
model.fit(x_train,y_train)
x_test_prediction=model.predict(x_test)
training_score=accuracy_score(x_test_prediction,y_test)
print('accuracy of model on test data set:%.2f'%training_score)
training_score=accuracy_score(x_test_prediction,y_test)
print('accuracy of model on test data set:%.2f'%training_score)
list=[62,0,0,140,268,0,0,160,0,3.6,0,2,2]
input_data_as_numpy=array(list).reshape(1,-1)
prediction=model.predict(input_data_as_numpy)
if prediction==1:
    print('person may suffer with heart attack')
else:
    print('person is not suffering from heart attack')
file_name='D:\disease prediction\heart.sav'
pickle.dump(model,open(file_name,'wb'))
a=pickle.load(open(file_name,'rb'))
print(a)
print(a.predict(input_data_as_numpy))


#DIABETES MODEL
from numpy import *
from pandas import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
from matplotlib.pyplot import *
warnings.filterwarnings('ignore')
diabetes_data=read_csv('D:\disease prediction\diabetes.csv')
diabetes_data.head()
diabetes_data.info()
diabetes_data.shape
diabetes_data.isnull().sum()
graphs of the data who dont have diabetes
p=diabetes_data[diabetes_data['Outcome']==0].hist(figsize=(20,20))
graphs of the data who has diabetes
p=diabetes_data[diabetes_data['Outcome']==1].hist(figsize=(20,20))
x=diabetes_data.drop(columns='Outcome',axis=1)
y=diabetes_data['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=45)
model=LogisticRegression()
model.fit(x_train,y_train)
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('training data accuraacy%.2f'%training_data_accuracy)
x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('test data accuracy%.2f'%test_data_accuracy)
input_data_str=[5,166,72,19,175,25.8,0.587,51]
input_data=array(input_data_str,dtype=float).reshape(1,-1)
prediction=model.predict(input_data)
print(prediction)
if prediction==0:
    print('person dont have diabetes')
else:
    print('person has diabetes')

import pickle
filename='D:\disease prediction\diabetes.sav'
pickle.dump(model,open(filename,'wb'))
loaded_model=pickle.load(open('D:\disease prediction\diabetes.sav','rb'))j
