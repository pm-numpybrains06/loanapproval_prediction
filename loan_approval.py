#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

df = pd.read_csv("loan.csv")  # or whatever the filename is
df.head()


# In[4]:


df.info()
df.describe()
df['Loan_Status'].value_counts()


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(df['Loan_Status'])
sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=df)


# In[7]:


df.isnull().sum()


# In[8]:


df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)


# In[9]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    df[col] = le.fit_transform(df[col])


# In[10]:


df.drop('Loan_ID', axis=1, inplace=True)


# In[11]:


df.head()


# In[12]:


X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

string_columns = X_train.select_dtypes(include=['object']).columns
for col in string_columns:

    X_train[col] = pd.to_numeric(X_train[col].str.replace('+', ''), errors='coerce')


    X_train[col] = X_train[col].fillna(X_train[col].mean())

model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[16]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

print("Sample of X_test:", X_test.iloc[:5] if hasattr(X_test, 'iloc') else X_test[:5])

X_test_processed = X_test.copy()


for col in X_test_processed.columns:
    if X_test_processed[col].dtype == 'object':

        X_test_processed[col] = X_test_processed[col].str.replace('+', '')

        X_test_processed[col] = pd.to_numeric(X_test_processed[col], errors='ignore')


        if X_test_processed[col].dtype == 'object':
            le = LabelEncoder()
            X_test_processed[col] = le.fit_transform(X_test_processed[col])


y_pred = model.predict(X_test_processed)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:




