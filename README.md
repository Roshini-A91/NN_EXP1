# NN_EXP-1
<H3>NAME: ROSHINI A</H3>
<H3>REGISTER NO: 212224230233</H3>
<H3>DATE : 03.08.2026</H3>
<H1>EX. NO.1Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
- Import the required libraries and load the Churn_Modelling.csv dataset.
- Explore and preprocess the dataset by checking missing values, duplicate values, and removing unnecessary columns (Surname, Geography, Gender).
- Normalize the dataset using MinMaxScaler to scale the feature values.
- Split the dataset into Training (80%) and Testing (20%) sets using train_test_split().
- Train the Random Forest Classifier using the training dataset and build the prediction model.
- Predict customer churn on the testing dataset and evaluate the model using Accuracy Score.
##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
print(data.head())
print(data.tail())

X=data.iloc[:,:-1].values
print(X)

y=data.iloc[:,-1].values
print(y)

data.info()

print("Missing Values: \n ",data.isnull().sum())

print("Duplicate values:\n ")
print(data.duplicated())

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print("Normalized data \n" , df1)

X = data.drop('Exited', axis=1)  
y = data['Exited'] 

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Training data")
print(X_train)
print(y_train)

print("Testing data")
print(X_test)
print(y_test)
print("Length of X_test: ", len(X_test))


```
## OUTPUT:
<img width="756" height="442" alt="1" src="https://github.com/user-attachments/assets/5d3764a0-0306-4c71-b596-718835bfc516" />
<img width="751" height="446" alt="2" src="https://github.com/user-attachments/assets/08b742b0-94ec-4cef-9884-d41767fefe12" />
<img width="463" height="182" alt="3" src="https://github.com/user-attachments/assets/9cb3b0f7-d1d4-43d2-857c-12dca88111ca" />
<img width="528" height="476" alt="4" src="https://github.com/user-attachments/assets/29855ac8-e19c-405c-bcb6-1dddb67091b2" />
<img width="368" height="367" alt="5" src="https://github.com/user-attachments/assets/53ed49cb-d214-49ea-b255-9c81014908e4" />
<img width="426" height="320" alt="6" src="https://github.com/user-attachments/assets/7e729629-9737-4b41-9444-19552dec2d6b" />
<img width="1246" height="235" alt="7" src="https://github.com/user-attachments/assets/fbd79281-b818-4f87-9518-a4cbac7bc665" />
<img width="762" height="607" alt="8" src="https://github.com/user-attachments/assets/da88cbe0-43f6-4d15-bfa1-5310ddcdef3f" />
<img width="696" height="575" alt="9" src="https://github.com/user-attachments/assets/18bfdf8d-d29f-4215-828d-243df14e8f5e" />
<img width="666" height="585" alt="10" src="https://github.com/user-attachments/assets/40981cbc-4476-4049-98d0-93c0c1923d05" />
<img width="667" height="597" alt="11" src="https://github.com/user-attachments/assets/ff96f205-e73b-47b3-b568-9ad1271794af" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
