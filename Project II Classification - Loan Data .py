#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loan Data Set
# ### Dependent and Indenpendent Variabels 
# #### x0 = credit.policy (ผู้ยืมผ่านนโยบายไหม)
# #### x1 = purpose (จุดประสงค์ของการยืม)
# #### x2 = init.rate (ดอกเบี้ย)
# #### x3 = installment (ระยะเวลา)
# #### x4 = log.annual.inc (natural log ของรายได้ต่อปี)
# #### x5 = dti (อัตราส่วนหนี้ต่อรายได้)
# #### x6 = fico (คะแนน FICO ของผู้ยืม)
# #### x7 = days.with.cr.line (จำนวนวันที่ผู้ยืมมี credit line)
# #### x8 = revol.bal (เงินที่ไม่จ่ายในบัตรเครดิต)
# #### x9 = revol.util (จำนวน credit line ที่ถูกใช้)
# #### x10 = inq.last.6mths (คำถามจากธนาคารภายใน 60 วันล่าสุด)
# #### x11 = delinq.2yrs (ค้างชำระเกิน 30 วัน)
# #### x12 = pub.rec (ประวัติสาธารณะ เช่น ล้มละลาย)
# #### y = not.fully.paid (จ่ายคืนครบไหม (label)) 1 = not fully paid , 0 = fully paid

# # 1.Data Acquistion

# In[2]:


df = pd.read_csv('../Desktop/DataCamp/loan_data.csv')
df


# In[3]:


df.shape


# #### There are 9578 records and 14 features in this data set.

# # 2.Data Cleaning

# In[4]:


fig = plt.figure(figsize = (12,8))
sns.heatmap(df.isnull(), cbar = False, cmap = 'Blues_r')


# #### No null values in this data set. Let's dive into data analysis.

# # 3.Data Analysis

# In[5]:


df.info()


# #### We have both numerical and categorical data here. Only purpose column is categorical data.

# In[6]:


df.describe()


# In[7]:


df.corr()


# In[8]:


fig = plt.figure(figsize = (12,8))
sns.countplot(data = df,x = 'purpose', palette = 'rainbow')


# #### Majority of purpose to get loan is debt collection, others and  credit card.

# In[9]:


fig = plt.figure(figsize = (12,8))
sns.countplot(data = df,x = 'purpose', palette = 'rainbow', hue = 'not.fully.paid')


# #### The number of people who have fully paid off is more than those who not fully paid off for all categories 

# ## Create dummy variables for purpose column

# In[11]:


df


# In[12]:


df_real = pd.get_dummies(df,drop_first = True)
df_real


# In[13]:


df_real.info()


# In[15]:


fig = plt.figure(figsize = (12,8))
sns.heatmap(df_real.corr(), annot = df_real.corr())


# # 4.Feature Selection

# #### We can see from the correlation table that all independent variables are  low linear correlated with the column 'not fully paid' . In this case, I decided to use all variables in my models 

# In[16]:


X = df_real.drop(['not.fully.paid'], axis = 1)
y = df['not.fully.paid']


# # 5.Model Building 

# #### I will build Logistic Regression, K-NN, SVM, Naive Bayes and Decision Tree.

# In[66]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# In[67]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[68]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[69]:


logistic_regression = LogisticRegression()
logistic_regression.fit(X_train,y_train)

knn = KNeighborsClassifier(n_neighbors = 29)
knn.fit(X_train,y_train)

nb = GaussianNB()
nb.fit(X_train,y_train)

svc = SVC()
svc.fit(X_train,y_train)

dtree =  DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[70]:


predicted_logistic = logistic_regression.predict(X_test)
predicted_knn = knn.predict(X_test)
predicted_nb = nb.predict(X_test)
predicted_svm = svc.predict(X_test)
predicted_dtree = dtree.predict(X_test)


# In[71]:


print('predicted Logistic Regresion', predicted_logistic) 
print('predicted KNN', predicted_knn) 
print('predicted Naive Bayes', predicted_nb) 
print('predicted SVM', predicted_svm) 
print('predicted Decision Tree', predicted_dtree) 


# # 6.Model Evalution (defult)

# In[72]:


print('confusion matrix Logistic Regresion')
print(confusion_matrix(y_test,predicted_logistic)) 
print('confusion matrix KNN')
print(confusion_matrix(y_test,predicted_knn)) 
print('confusion matrix Naive Bayes')
print(confusion_matrix(y_test,predicted_nb)) 
print('confusion matrix SVM')
print(confusion_matrix(y_test,predicted_svm)) 
print('confusion matrix Decision Tree')
print(confusion_matrix(y_test,predicted_dtree))


# In[74]:


print('Logistic Regresion')
print('accuracy score',accuracy_score(y_test,predicted_logistic))
print('precision score',precision_score(y_test,predicted_logistic))
print('recall_score',recall_score(y_test,predicted_logistic))
print('f1 score',f1_score(y_test,predicted_logistic))


# In[75]:


print('KNN')
print('accuracy score',accuracy_score(y_test,predicted_knn))
print('precision score',precision_score(y_test,predicted_knn))
print('recall_score',recall_score(y_test,predicted_knn))
print('f1 score',f1_score(y_test,predicted_knn))


# In[77]:


print('Naive Bayes')
print('accuracy score',accuracy_score(y_test,predicted_nb))
print('precision score',precision_score(y_test,predicted_nb))
print('recall_score',recall_score(y_test,predicted_nb))
print('f1 score',f1_score(y_test,predicted_nb))


# In[79]:


print('SVM')
print('accuracy score',accuracy_score(y_test,predicted_svm))
print('precision score',precision_score(y_test,predicted_svm))
print('recall_score',recall_score(y_test,predicted_svm))
print('f1 score',f1_score(y_test,predicted_svm))


# In[80]:


print('Decision Tree')
print('accuracy score',accuracy_score(y_test,predicted_dtree))
print('precision score',precision_score(y_test,predicted_dtree))
print('recall_score',recall_score(y_test,predicted_dtree))
print('f1 score',f1_score(y_test,predicted_dtree))


# # 7.Further Study

# # K-NN 

# #### Which n gives the highest accuracy score (range from 1 to 30)? 

# In[81]:


accuracy_lst = []

for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train,y_train)
    predicted_i = knn.predict(X_test)
    accuracy_lst.append(accuracy_score(y_test,predicted_i))


# In[82]:


accuracy_lst


# In[83]:


plt.figure(figsize = (12,8))
plt.plot(range(1,30), accuracy_lst,color = 'black',linestyle = 'dashed', marker = 'o', 
         markerfacecolor = 'blue', markersize = 7)
plt.xlabel('Number of K')
plt.ylabel('Accuracy')


# #### From the result above, we can use number of K 20 or more.

# # Decision Tree

# #### We can see that Decision Tree results are worse than other models' results. I will do GridSeach hyperparameter tuning in this case.

# In[84]:


from sklearn.model_selection import GridSearchCV


# In[85]:


param_combination = {'max_depth': [4,8,16,32,64], 'min_samples_leaf':[1,2,4,8,16]}


# In[86]:


grid_search = GridSearchCV(DecisionTreeClassifier(), param_combination,verbose = 3)
grid_search.fit(X_train,y_train)


# In[87]:


grid_search.best_params_


# In[88]:


grid_predicted_dtree = grid_search.predict(X_test)
grid_predicted_dtree


# In[89]:


confusion_matrix(y_test,grid_predicted_dtree)


# In[90]:


print('accuracy score',accuracy_score(y_test,grid_predicted_dtree))
print('precision score',precision_score(y_test,grid_predicted_dtree))
print('recall_score',recall_score(y_test,grid_predicted_dtree))
print('f1 score',f1_score(y_test,grid_predicted_dtree))


# #### We can clealy see that the accuracy score after adjusting hyperparameter is way higher

# # Normalization ( all 5 models)

# In[95]:


from sklearn.preprocessing import MinMaxScaler


# In[96]:


min_max_scaler =  MinMaxScaler()


# In[97]:


X_minmax = min_max_scaler.fit_transform(df_real.drop('not.fully.paid', axis =1))
X_minmax


# In[98]:


not_fully_paid = df_real['not.fully.paid']
not_fully_paid


# In[99]:


arr_not_fully_paid = np.array(not_fully_paid).reshape(-1,1)
arr_not_fully_paid


# In[100]:


y_minmax = min_max_scaler.fit_transform(arr_not_fully_paid)
y_minmax


# In[101]:


X_train,X_test,y_train,y_test = train_test_split(X_minmax, y_minmax,test_size =0.3, random_state = 20)


# In[106]:


logistic_regression2 = LogisticRegression()
logistic_regression2.fit(X_train,y_train)

knn2 = KNeighborsClassifier(n_neighbors = 29)
knn2.fit(X_train,y_train)

nb2 = GaussianNB()
nb2.fit(X_train,y_train)

svc2 = SVC()
svc2.fit(X_train,y_train)

dtree2 =  DecisionTreeClassifier()
dtree2.fit(X_train,y_train)


# In[107]:


predicted_logistic2 = logistic_regression.predict(X_test)
predicted_logistic2 = logistic_regression.predict(X_test)
predicted_knn2 = knn.predict(X_test)
predicted_nb2 = nb.predict(X_test)
predicted_svm2 = svc.predict(X_test)
predicted_dtree2 = dtree.predict(X_test)


# In[108]:


print('predicted Logistic Regresion 2', predicted_logistic) 
print('predicted KNN 2', predicted_knn) 
print('predicted Naive Bayes 2', predicted_nb) 
print('predicted SVM 2', predicted_svm) 
print('predicted Decision Tree 2', predicted_dtree)


# In[109]:


print('confusion matrix Logistic Regresion')
print(confusion_matrix(y_test,predicted_logistic2)) 
print('confusion matrix KNN')
print(confusion_matrix(y_test,predicted_knn2)) 
print('confusion matrix Naive Bayes')
print(confusion_matrix(y_test,predicted_nb2)) 
print('confusion matrix SVM')
print(confusion_matrix(y_test,predicted_svm2)) 
print('confusion matrix Decision Tree')
print(confusion_matrix(y_test,predicted_dtree2))


# In[110]:


print('Logistic Regresion')
print('accuracy score',accuracy_score(y_test,predicted_logistic2))
print('precision score',precision_score(y_test,predicted_logistic2))
print('recall_score',recall_score(y_test,predicted_logistic2))
print('f1 score',f1_score(y_test,predicted_logistic2))


# In[111]:


print('KNN')
print('accuracy score',accuracy_score(y_test,predicted_knn2))
print('precision score',precision_score(y_test,predicted_knn2))
print('recall_score',recall_score(y_test,predicted_knn2))
print('f1 score',f1_score(y_test,predicted_knn2))


# In[112]:


print('Naive Bayes')
print('accuracy score',accuracy_score(y_test,predicted_nb2))
print('precision score',precision_score(y_test,predicted_nb2))
print('recall_score',recall_score(y_test,predicted_nb2))
print('f1 score',f1_score(y_test,predicted_nb2))


# In[113]:


print('SVM')
print('accuracy score',accuracy_score(y_test,predicted_svm2))
print('precision score',precision_score(y_test,predicted_svm2))
print('recall_score',recall_score(y_test,predicted_svm2))
print('f1 score',f1_score(y_test,predicted_svm2))


# In[114]:


print('Decision Tree')
print('accuracy score',accuracy_score(y_test,predicted_dtree2))
print('precision score',precision_score(y_test,predicted_dtree2))
print('recall_score',recall_score(y_test,predicted_dtree2))
print('f1 score',f1_score(y_test,predicted_dtree2))


# #### Normalization method: we have got worse results except Naive Bayes model 

# # 8.Suggestion

# #### I performed 5 models: Logistic  Regression, K-NN, SVM, Naive Bayes and Decision Tree. Decision Tree with hyperparameter tuning got the best result ( accuracy score = 0.83) , So I would use it to predict the classification of people who have fully paid / have not fully paid their loan. 
# 
# #### I would use all variables  due to the fact that all predictor variables are low linearly correlated.
