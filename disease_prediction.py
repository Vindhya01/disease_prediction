#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix,classification_report,confusion_matrix,precision_score,roc_auc_score,roc_curve
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import pickle
import time


# In[2]:


disease_symptom_data = pd.read_csv("D:\\Ai_internship\\disease_symptom_dataset\\dataset.csv")
print(disease_symptom_data)


# In[3]:


for col in disease_symptom_data.columns: 
    disease_symptom_data[col] = disease_symptom_data[col].str.replace('_',' ')


# In[4]:


cols = disease_symptom_data.columns
data = disease_symptom_data[cols].values.flatten()

s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(disease_symptom_data.shape)


# In[5]:


disease_symptom_data = pd.DataFrame(s, columns=disease_symptom_data.columns)
disease_symptom_data = disease_symptom_data.fillna(0)


# In[6]:


symptom_severity_data = pd.read_csv("D:\\Ai_internship\\disease_symptom_dataset\\symptom_severity.csv")
symptoms=symptom_severity_data['Symptom']


# In[7]:


disease_symptom_and_severity=pd.DataFrame()
disease_symptom_and_severity["Disease"]=disease_symptom_data["Disease"]
y=0
disease_symptom_and_severity[symptoms]=0
for index, row in disease_symptom_data.iterrows():
    for symptom in disease_symptom_data.columns[1:]:
        if row[symptom] != 0:
            disease_symptom_and_severity.loc[index, row[symptom]] = 1
disease_symptom_and_severity = disease_symptom_and_severity.fillna(0)
disease_symptom_and_severity[disease_symptom_and_severity.columns[1:]]=disease_symptom_and_severity[disease_symptom_and_severity.columns[1:]].astype('int')


# In[8]:


disease_symptom_and_severity.columns = disease_symptom_and_severity.columns.str.strip()


# In[9]:


disease_symptom_and_severity.drop(disease_symptom_and_severity.columns[-4:], axis=1, inplace=True)
disease_symptom_and_severity


# In[10]:


columns_to_drop = ['foul_smell_ofurine', 'dischromic_patches', 'spotting_urination']
disease_symptom_and_severity = disease_symptom_and_severity.drop(columns=columns_to_drop)


# In[11]:


disease_symptom_and_severity[disease_symptom_and_severity.columns[1:]].sum(axis=0).sort_values()


# In[12]:


y=disease_symptom_data['Disease'].unique()
y


# In[13]:


data = disease_symptom_and_severity.iloc[:,1:].values
labels = disease_symptom_and_severity['Disease'].values


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.7,random_state=42)
x_train, x_val, y_train,y_val=train_test_split(data,labels,test_size=0.2,random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, x_val.shape,y_val.shape)


# In[15]:


le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_val=le.transform(y_val)


# In[16]:


y=le.classes_
y


# In[17]:


# Initialize the Decision Tree classifier
dt = DecisionTreeClassifier(random_state=1)

# Define the parameter grid for GridSearchCV
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Grid Search with cross-validation
start_time = time.time()
grid_search_dt = GridSearchCV(estimator=dt, param_grid=param_grid_dt, cv=10, scoring="accuracy", verbose=1)
grid_search_dt.fit(x_train, y_train)
dt_time = time.time() - start_time

# Best model and its parameters
best_dt = grid_search_dt.best_estimator_
print("Best parameters for Decision Tree:", grid_search_dt.best_params_)
print(f"Decision Tree Training Time: {dt_time:.2f} seconds")

# Train and test the model
best_dt.fit(x_train, y_train)

# Test set evaluation
test_predictions = best_dt.predict(x_test)
test_f1 = f1_score(y_test, test_predictions, average='weighted')
test_roc = roc_auc_score(y_test, best_dt.predict_proba(x_test), multi_class='ovr')
print(f"Decision Tree Test F1 Score: {test_f1:.4f}, AUC-ROC Score: {test_roc:.4f}")

# Validation set evaluation
val_predictions = best_dt.predict(x_val)
val_f1 = f1_score(y_val, val_predictions, average='weighted')
val_roc = roc_auc_score(y_val, best_dt.predict_proba(x_val), multi_class='ovr')
print(f"Decision Tree Validation F1 Score: {val_f1:.4f}, AUC-ROC Score: {val_roc:.4f}")

# Save the trained model
pickle.dump(best_dt, open("D:\\Ai_internship\\disease_symptom_dataset\\decision_tree_model.pkl", "wb"))

# Print accuracy and classification report
y_pred_dt = best_dt.predict(x_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))


# In[18]:


desc = pd.read_csv("D:\\Ai_internship\\disease_symptom_dataset\\symptom_description.csv")


# In[19]:


desc.head()


# In[20]:


prec = pd.read_csv("D:\\Ai_internship\\disease_symptom_dataset\\symptom_precaution.csv")


# In[21]:


prec.head()


# In[22]:


def predd(m, X):
    # Get probabilities for each class
    proba = m.predict_proba(X)

    # Get the indices and probabilities of the top 5 classes
    top5_idx = np.argsort(proba[0])[-5:][::-1]
    top5_proba = np.sort(proba[0])[-5:][::-1]

    # Get the names of the top 5 diseases
    top5_diseases = y[top5_idx]

    for i in range(5):
        
        disease = top5_diseases[i]
        probability = top5_proba[i]
        # print(f"{disease}={probability}" )
        
        print("Disease Name: ", disease)
        print("Probability: ", probability)
        if(disease in desc["Disease"].unique()):
            disp = desc[desc['Disease'] == disease]
            disp = disp.values[0][1]
            print("Disease Description: ", disp)
        
        if(disease in prec["Disease"].unique()):
            c = np.where(prec['Disease'] == disease)[0][0]
            precuation_list = []
            for j in range(1, len(prec.iloc[c])):
                precuation_list.append(prec.iloc[c, j])
            print("Recommended Things to do at home: ")
            for precaution in precuation_list:
                print(precaution)
        
        print("\n")


# In[23]:


prec


# In[24]:


x=disease_symptom_and_severity.columns[1:]


# In[25]:


x


# In[26]:


y


# In[29]:


t=pd.Series([0]*222, index=x)
m=DecisionTreeClassifier()
with open("D:\\Ai_internship\\disease_symptom_dataset\\decision_tree_model.pkl", "rb") as f:
    m =  pickle.load(f)
t.loc["chest_pain"]=1
t.loc["phlegm"]=1
t.loc["runny_nose"]=1
t.loc["high_fever"]=1
t.loc["throat_irritation"]=1
t.loc["congestion"]=1
t.loc["redness_of_eyes"]=1
t=t.to_numpy()
print(t.shape)
t=t.reshape(1,-1)
predd(m,t)


# In[ ]:




