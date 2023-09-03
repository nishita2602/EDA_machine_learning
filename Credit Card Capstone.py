#!/usr/bin/env python
# coding: utf-8

# # <font color='Purple'> Project Description:
# In this project, I am developing an automated credit card approval predictor using advanced machine learning techniques. By analyzing factors like income levels, and credit inquiries, we're creating a model to accurately evaluate credit card applications. 
#     
# Leveraging the Credit Card Approval dataset, My goal is to streamline and enhance the credit application process, offering faster, more consistent, and data-driven decisions. 
#     
# Through this project, We aim to showcase the potential of AI-driven solutions in revolutionizing critical financial decision-making processes and contributing to the advancement of financial technology.
#     
# 

# ### Why is the proposal important in today's world?
# 
# In today's world, financial institutions face the challenge of efficiently assessing credit card applications. Our proposal holds significance as it employs data analysis to predict creditworthiness, enhancing the decision-making process.
# 
# As financial transactions continue to digitalize, accurate credit predictions become crucial for risk management and customer satisfaction.
# 
# 

# ### How predicting a good client is worthy for a bank?
# 
# Predicting a reliable client is invaluable for banks. It minimizes the risk of default on loans, reduces bad debts, and ensures a healthy loan portfolio. 
# 
# This predictive ability streamlines the lending process, improves customer experience, and ultimately enhances the bank's financial stability and reputation.
# 

# ### How is it going to impact the banking sector?
# 
# Our proposal impacts the banking sector by introducing data-driven precision to credit evaluations. It revolutionizes decision-making, making it faster and more accurate. 
# 
# By minimizing defaults and bad loans, banks can save resources and focus on strategic growth. Moreover, customers benefit from quicker and fairer loan approvals, fostering trust in the banking system.

# ### If any, what is the gap in the knowledge or how my proposed method can be helpful if required in the future for any bank in India.
# 
# There exists a gap in traditional credit assessment methods, which might overlook subtle patterns in vast datasets. 
# 
# My method bridges this gap by harnessing advanced analytics, uncovering hidden insights, and making more informed credit predictions.
# 
# In the future, this approach can serve as a template for other banks in India to adopt data-centric strategies for risk assessment.
# 
# 
# 
# 

# ## Our initial hypotheses are:
# 
# Hypothesis 1: Through data analysis, we anticipate discovering significant patterns that correlate with creditworthiness.
# 
# 
# Hypothesis 2: Machine learning models, particularly those based on ensemble methods, will outperform individual algorithms in credit prediction.
# 
# 
# Hypothesis 3: Key features such as annual income, employment duration, and credit history will emerge as pivotal indicators for credit approval.
# 
# 
# As we explore the data and test different models, we will refine these hypotheses and extract actionable insights to develop an efficient credit prediction model tailored for the banking sector. The model's effectiveness will be justified through relevant cost functions and visualized using graphs, showcasing its superiority over other potential models.

# ## <font color='Green'> Importing Libraries:
# 
# We will import all of the primary packages into our python environment.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# ## <font color='green'> Loading Data:
# 
# We start the project by loading the dataset in our Jupyter notebook. The dataset is loaded into a pandas dataframe named data. 

# In[2]:


data=pd.read_csv('Credit_card[1].csv')
data.head() #looking at the dataset, we print the first five rows using .head().


# ### <font color='Red'> To prove or disprove our hypotheses:
# 
# We will employ an exploratory data analysis (EDA) approach. We'll begin by comprehensively examining the dataset's distribution, identifying outliers, and addressing missing values. We'll analyze the relationships between features and target variables, employing visualization techniques such as histograms and correlation matrices.
# 
#     
# Feature engineering techniques like scaling, one-hot encoding for categorical variables, and handling missing values will be essential to prepare the data for modeling.
# 
# Our data analysis approach is justified as it enables us to uncover hidden patterns, understand the significance of features, and lay the groundwork for the subsequent machine learning phase.

# ## Let's Start :

# ## <font color='green'> Knowing Our Data:
# 
# To understand our data better, we use pandas features .info(),.describe(),shape.
# 
# We are also checking if we have any null data using .isnull().sum()

# In[3]:


data.info()


# <font color='blue'> From the output we get the following information about the data: Data has a total of 1548 entries i.e. approval or rejection data of 1548 credit card applications with a total of 17 columns or input features.
# 
# <font color='blue'> From the output’s Dtype column, we see several features with Dtype as object (string or mixed), which we will have to convert into int64 in later stage for ML Algorithms.

# In[4]:


data.shape


# In[5]:


data.describe()


# In[6]:


data.isnull().sum()


# ## <font color='green'> Visualizing the Data:
#     
#     

# In[7]:


data.hist(figsize=(10,10))


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Drop rows with missing values for the relevant columns
data_cleaned = data.dropna(subset=['Annual_income', 'Birthday_count'])

# Scatter plot: Annual Income vs. Age
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_cleaned, x='Birthday_count', y='Annual_income')
plt.title('Scatter Plot: Annual Income vs. Age')
plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.show()


# In[9]:


data.head()


# In[10]:


# Drop rows with missing values for the relevant columns
data_cleaned_gender = data.dropna(subset=['GENDER', 'Annual_income'])

# Box plot: Gender vs. Annual Income
plt.figure(figsize=(8, 6))
sns.boxplot(data=data_cleaned_gender, x='GENDER', y='Annual_income')
plt.title('Box Plot: Gender vs. Annual Income')
plt.xlabel('Gender')
plt.ylabel('Annual Income')
plt.show()


# In[11]:


import pandas as pd
from scipy.stats import pearsonr



# Drop rows with missing values for the relevant columns
data_cleaned = data.dropna(subset=['Annual_income', 'Birthday_count'])

# Calculate Pearson correlation and p-value
correlation, p_value = pearsonr(data_cleaned['Annual_income'], data_cleaned['Birthday_count'])

print("Pearson Correlation:", correlation)
print("P-value:", p_value)


# <font color='blue'>The analysis revealed a statistically significant positive correlation (correlation coefficient ≈ 0.112) between 'Annual Income' and 'Age.' 
# 
# <font color='blue'>The low p-value (≈ 1.44e-05) indicates that as individuals' age increases, their annual income tends to rise. However, the correlation is weak, suggesting that age explains only a small portion of the income variation.
# 
# <font color='blue'>While the results show a meaningful link, it's important to remember that correlation doesn't imply causation, and other unexplored factors might influence this relationship."

# In[12]:


data['GENDER'].value_counts()


# In[13]:


from scipy.stats import ttest_ind

# Drop rows with missing values for the relevant columns
data_cleaned_gender = data.dropna(subset=['GENDER', 'Annual_income'])

# Separate data by gender
female_income = data_cleaned_gender[data_cleaned_gender['GENDER'] == 'F']['Annual_income']
male_income = data_cleaned_gender[data_cleaned_gender['GENDER'] == 'M']['Annual_income']

# Perform independent samples t-test
t_statistic, p_value = ttest_ind(female_income, male_income)

print("T-statistic:", t_statistic)
print("P-value:", p_value)


# <font color='blue'> The conducted independent samples t-test revealed a substantial and statistically significant difference in 'Annual Income' based on gender. 
#     
# <font color='blue'>The negative t-statistic (≈ -8.58) and very low p-value (≈ 2.30e-17) indicate that, on average, one gender's annual income is significantly lower than the other. 
#     
#     
# <font color='blue'>This underscores the role of gender in influencing income disparities within the dataset. It's important to consider broader factors that might contribute to these observed differences.

# In[14]:


data.duplicated()


# In[15]:


data.corr()


# <font color='blue'> Data.corr() is used to find the pairwise correlation of all columns in the Pandas Dataframe in Python.
# 
# <font color='blue'> Any NaN values are automatically excluded. Any non-numeric data type or columns in the Dataframe, it is ignored.

# In[16]:


data.isnull().sum()


# <font color='blue'> Data.isnull().sum() provides us with missing values in the dataset , with above data we can see that there are missing data in 'Gender','Annual_income','Birthday_count','Type_Occupation'.
#  
# <font color='blue'> We can either remove or replace the missing values with any imputation method in later stage.

# ## <font color='green'> Data Cleaning:
# 
# In the world of data analysis, dealing with messy data is a reality we can't escape.
# Every dataset, without exception, may contain missing values across various columns, each corresponding to a data entry. Before diving into data analysis and drawing conclusions.
# 
# It's crucial to recognize the existence of these missing values in our dataset. The way missing values are represented can vary, such as using symbols like ?, NaN. When a column's data type is numeric (int or float), missing values are often indicated using NaN.
# On the other hand, for columns with categorical data types, we display the distinct values present. As we inspect the dataset, we notice the presence of missing values, marked with the label '?'.
# 
# Also there are several columns in the raw data which does not add any value to the data, we are dropping all such columns which has very less or no impact on our output.
# 
# 

# In[17]:


data.drop('Mobile_phone',inplace=True,axis=1)
data.drop('Work_Phone',inplace=True,axis=1)
data.drop('Phone',inplace=True,axis=1)
data.drop('EMAIL_ID',inplace=True,axis=1)


# <font color='blue'> Here, we dropped columns like 'Mobile_phone','Work_Phone','Phone','EMAIL_ID' as they have no impact on the Output.

# In[18]:


data['Age'] = data['Birthday_count']/365*(-1)  #Changing the Birthdaycount column into Age


# <font color='blue'> Here, We change the 'Birthday_count' column into 'Age' so that our data works and looks better.

# In[19]:


data['Experience'] = data['Employed_days']/365*(-1) #Changing the Employed days column into Experience


# In[20]:


data['Experience'] = data['Experience'].apply(lambda x: round(x, 2))


# <font color='blue'> As we changed for 'Age' column likewise We are converting  the 'Employed_days' column into 'Experience' so that our data works and looks better.

# In[21]:


data.drop('Birthday_count',inplace=True,axis=1)
data.drop('Employed_days',inplace=True,axis=1)


# <font color='blue'>  We dropped both the columns 'Birthday_count' and 'Employed_days' from the original data as we have changed the data into 'Age' and 'Experience'.

# ## <font color='green'> Finding and Handling Missing Number:
# 
# Missing data is probably one of the most common issues when working with real datasets. Data can be missing for a multitude of reasons, including sensor failure, data vintage, improper data management, and even human error. Missing data can occur as single values, multiple values within one feature, or entire features may be missing.
# 
# It is important that missing data is identified and handled appropriately prior to further data analysis or machine learning. Many machine learning algorithms can’t handle missing data and require entire rows, where a single missing value is present, to be deleted or replaced (imputed) with a new value.
# 

# ## <font color='green'>  Median Imputation on 'Annual_income' column:

# Median imputation is a technique used to replace missing values in a dataset with the median value of the available data. In the context of annual income, median imputation involves replacing missing income values with the median income of the individuals or cases for which income data is available. 
# 
# This method is often used to handle missing data in a way that avoids extreme outliers and maintains the overall distribution of income.

# In[22]:


data['Annual_income'].fillna(data['Annual_income'].median())


# <font color='blue'> Imputed the 23 missing annual income values with the median income to mitigate the influence of outliers

# In[23]:


data.dropna(inplace=True)


# <font color='blue'> 'Data.dropna(inplace=True)' will remove all rows from the data DataFrame that contain at least one missing value and update the DataFrame itself.
#  
# <font color='blue'> This can be a useful step when you want to clean your data by getting rid of rows that have incomplete information. 
#  
#  <font color='Grey'>Just make sure to use this method with caution, as removing rows with missing data might lead to loss of valuable information, and it's important to consider the impact on your analysis or model.

# In[24]:


data = data.astype({'Age':'int'})


# <font color='blue'> We are changing the datatype into 'int' for Age column from 'float'.

# In[25]:


get_ipython().system('pip install missingno')


# In[26]:


data.head()


# In[27]:


import missingno as msno


# In[28]:


msno.bar(data) # shows how much data is missing

#The amount of empty spaces shows missing data


# We can see that there is no missing values as we have already dealt with the missing values earlier and dropped all the na value in the dataset i.e., the missing values were removed from the original dataset.
# 
# Though we should always be cautious before removing any missing values , either we can handle the missing values by imputation as we did earlier for 'Annual_ income column' and for others we removed the missing data by 'data.dropna'.

# In[29]:


data.shape


# ## <font color='green'> Outlier Treatment:
#     
#  Outlier treatment involves identifying and addressing data points in a dataset that deviate significantly from the overall pattern or distribution of the data.
#     
#  Outliers can be caused by measurement errors, data entry mistakes, or genuine anomalies in the data. Managing outliers is important because they can skew statistical analysis, model performance, and the overall understanding of the data.

# In[30]:


sns.kdeplot(data['Annual_income'])


# In[31]:


sns.boxplot(y=data['Annual_income'])


# <font color= 'blue'> We can easily see the presence of outliers in the given data.
# 
#     
# <font color= 'blue'> Data points that fall significantly above or below certain thresholds based on these methods are often flagged as outliers. we are going to use IQR outlier treatment to deal with the outliers in our data.

# In[32]:


data.describe()


# ## <font color='green'> Outlier detection and Treatment using IQR:
#     
#     
# -Outlier detection using the Interquartile Range (IQR) is a practical method for spotting potential outliers in a dataset. 
#  
# -Compute the IQR by finding the range between the third quartile (Q3) and the first quartile (Q1). 
#     
# -Multiply the IQR by a chosen factor (commonly 1.5 or 3) to determine lower and upper boundaries. 
#     
# -Data points falling below the lower bound or above the upper bound are considered possible outliers.
#     
# -Visualize these outliers through a box plot, where points outside the whiskers indicate potential outliers. 
# 
# -Examine flagged data points, considering their context and whether further investigation or treatment is necessary. 
#     
# -Adjust the multiplier factor based on the data's characteristics and sensitivity requirements. 
#     
# 

# In[33]:


Q1= data['Annual_income'].quantile(0.25)
Q3= data['Annual_income'].quantile(0.75)


# In[34]:


Q1


# In[35]:


Q3


# In[36]:


IQR= Q3 - Q1
IQR  # quantative value which shows variation b/w values.
#The smaller the iqr the better data 


# In[37]:


low_lim =Q1-1.5 * IQR
high_lim= Q3 + 1.5 * IQR
low_lim
high_lim


# In[38]:


data = data[(data['Annual_income']> low_lim) & (data['Annual_income']<high_lim)]


# In[39]:


outlier =[]
def detect_outlier(column):
    for x in data[column]:
        if (x > high_lim) or (x<low_lim):
            outlier.append(x)


# In[40]:


detect_outlier('Annual_income')


# In[41]:


outlier


# In[42]:


detect_outlier('Age')
outlier


# In[43]:


sns.boxplot(y=data['Annual_income'])


# <font color= 'blue'> We observe that the outliers have been removed by the IQR treatment.

# In[44]:


data.head()


# ## <font color='green'>  Encoding: 
#     
#  Encoding is a process used in data preprocessing and machine learning to convert categorical data (non-numeric values) into a numerical format that can be understood by algorithms. 
# 
#  Categorical data includes variables like colors, types of products, or geographic regions.

# In[45]:


get_ipython().system('pip install category_encoders')


# In[46]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce


# ### <font color='green'> Encoded Education column using Ordinal Endcoding:
#     
# Ordinal encoding is a method used to convert categorical variables with an inherent order or ranking into numerical values. This technique is particularly useful when dealing with data where the categories have a meaningful sequence, but the actual numeric values don't hold significant meaning.

# In[47]:


data['EDUCATION'].unique()


# In[48]:


data['EDUCATION']=data['EDUCATION'].map({'Higher education':4,'Secondary / secondary special':3,'Lower secondary':2,'Incomplete higher':1})


# In[49]:


data.head()


# ### <font color='green'> Encoding with Get_dummies:

# In[50]:


df_dummy=pd.get_dummies(data[['GENDER','Type_Income','Car_Owner','Propert_Owner']],drop_first=True)


# In[51]:


data=pd.concat([data,df_dummy],axis=1) 


# In[52]:


data.shape


# In[53]:


data.isnull().sum()


# In[54]:


data.drop(columns =['GENDER','Type_Income','Car_Owner','Propert_Owner'],inplace=True)


# ### <font color='green'> Encoding with OneHotEncoder:

# In[55]:


data['Marital_status'].value_counts()


# In[56]:


OHE=OneHotEncoder(handle_unknown='ignore',sparse=False)


# In[57]:


OHE.fit(data[['Marital_status']])


# In[58]:


encoded=OHE.transform(data[['Marital_status']])


# In[59]:


pd.DataFrame(encoded)


# In[60]:


encoded


# In[61]:


labels_marital=pd.DataFrame()
labels_marital['Marital_status_married']=encoded[:,0]
labels_marital['Marital_status_Single']=encoded[:,1]
labels_marital['Marital_status_Civil_marriage']=encoded[:,2]
labels_marital['Marital_status_Separated']=encoded[:,3]
labels_marital['Marital_status_Widow']=encoded[:,4]


# In[62]:


labels_marital.isnull().sum()


# In[63]:


data=pd.concat([data.reset_index(drop=True),labels_marital.reset_index(drop=True)], axis=1)


# In[64]:


data.drop('Marital_status',inplace=True,axis=1)


# In[65]:


data.isnull().sum()


# In[66]:


data['Housing_type'].value_counts()


# In[67]:


OHE_house=OneHotEncoder(handle_unknown='ignore',sparse=False)


# In[68]:


OHE_house.fit(data[['Housing_type']])


# In[69]:


encoded_house=OHE_house.transform(data[['Housing_type']])


# In[70]:


labels_house=pd.DataFrame()
labels_house['housing_type_house']=encoded_house[:,0]
labels_house['housing_type_with_parents']=encoded_house[:,1]
labels_house['housing_type_Municipal_apartment']=encoded_house[:,2]
labels_house['housing_type_Rented_apartment']=encoded_house[:,3]
labels_house['housing_type_Office_apartment']=encoded_house[:,4]
labels_house['housing_type_Office_Co_op_apartment']=encoded_house[:,5]


# In[71]:


data= pd.concat([data.reset_index(drop=True),labels_house.reset_index(drop=True)], axis=1)


# In[72]:


data.drop('Housing_type',inplace=True,axis=1)


# In[73]:


data['Type_Occupation'].value_counts()


# In[74]:


df_dummy_occ=pd.get_dummies(data[['Type_Occupation']],drop_first=True)


# In[75]:


data=pd.concat([data,df_dummy_occ],axis=1) 


# In[76]:


data.head()


# In[77]:


data.drop('Type_Occupation',inplace=True,axis=1)


# In[78]:


data.head()


# ## <font color='green'> Loading Target Variable Dataset:
#     
# We will load the target table and merge both input and output table for further analysis to split and train our machine learning model.

# In[79]:


data_op= pd.read_csv('Credit_card_label[1].csv')


# In[80]:


data_op.head()


# ## <font color='green'> Visualizing target Variable:

# In[81]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(7,5), sharex=True)
sns.countplot(data=data_op, x='label',edgecolor="white",palette="viridis",order=data_op["label"].value_counts().index)
total = data_op['label'].value_counts().sum()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Label', fontsize=12)
plt.ylabel('count', fontsize=12)

plt.show()


# ## <font color='green'> Merge the target data 

# In[82]:


M_data=pd.merge(data,data_op,on= 'Ind_ID')


# In[83]:


M_data


# In[84]:


M_data.drop('Ind_ID',inplace=True,axis=1)


# In[85]:


M_data


# ## <font color='green'> Train test split
#     
#     
# In our project, we performed a train-test split, a common practice in machine learning. 
# 
#     
#     
#  This involved dividing our dataset into a training set and a testing set. The training set was used to teach our machine learning model the patterns and relationships in the data. 
#     
#     
#  After training, we evaluated the model's performance using the testing set, which contained new, unseen data. This approach helped us ensure that our model could generalize well to real-world situations and avoid overfitting."

# In[86]:


#Training and Testing
from sklearn.model_selection import train_test_split

df_train,df_test=train_test_split(M_data,test_size=0.2,train_size=0.8)


# ## <font color='green'> Feature Scaling: 

# In[87]:


from sklearn.preprocessing import StandardScaler


# In[88]:


scaler_std=StandardScaler()


# In[89]:


numvars=['Annual_income','Age','Experience','Family_Members','CHILDREN'] #only columns which were numerical in start not encoded.

df_train[numvars] = scaler_std.fit_transform(df_train[numvars])


# In[90]:


numvars=['Annual_income','Age','Experience','Family_Members','CHILDREN']

df_test[numvars] = scaler_std.transform(df_test[numvars])


# In[91]:


#splitting the data for testing.
X_test=df_test.drop('label',axis=1)      #Input Testing
Y_test=df_test['label']                 #Output Testing


# In[92]:


#splitting the data for training.
X_train=df_train.drop('label',axis=1) #Input  Training
Y_train=df_train['label']  


# ## <font color='green'> ML Algorithms:

# ## 1.)Logistic Regression:
# 
# 
# We use logistic regression, a statistical method, to analyze and model relationships between variables. Suited for binary classification tasks, it estimates the probability of an outcome based on input features. 
# 
# By fitting a logistic curve to the data, it classifies instances into two classes. Logistic regression aids in understanding factors influencing outcomes and predicting future events

# In[93]:


# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0,1))
rescaledxTrain = scaler.fit_transform(X_train)
rescaledxTest = scaler.transform(X_test)

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression 
# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression()

# Fit logreg to the train set
logreg.fit(rescaledxTrain, Y_train)


# In[94]:


# Import confusion_matrix
from sklearn.metrics import confusion_matrix
# Use logreg to predict instances from the test set and store it
y_pred1 = logreg.predict(rescaledxTest)
y_pred2 = logreg.predict(rescaledxTrain)

# Get the accuracy score of logreg model and print it
print("Test: Accuracy = ", logreg.score(rescaledxTest,Y_test))
print("Train: Accuracy = ", logreg.score(rescaledxTrain,Y_train))

# Print the confusion matrix of the logreg model
confusion_matrix(Y_test,y_pred1)


# In[95]:


Y_test.shape,y_pred1.shape


# ## 2.)Decision Tree:
# 
# Implemented decision tree algorithm, a machine learning technique for classification and regression tasks. The tree-like model makes decisions based on input features, branching to different outcomes. It recursively splits data to maximize information gain and minimize impurity, resulting in a predictive model.
# 
# 
# Decision trees are interpretable, useful for feature selection, and can handle nonlinear relationships. They can be prone to overfitting but are often part of ensemble methods like Random Forests.

# In[96]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from pandas import DataFrame
import matplotlib.pyplot as plt


# In[97]:


train_acc=[]
test_acc=[]
list_score=[]
p=[]
from sklearn import tree
for i in range(1, 10):
    
    dtc = tree.DecisionTreeClassifier(max_depth = i ,random_state = 0)
    dtc.fit(X_train,Y_train)

    train_pred = dtc.predict(X_train)
    #train_acc.append(score(train_pred, yTrain))
    
    test_pred = dtc.predict(X_test)
    #test_acc.append(score(test_pred, yTest))
    test_acc = accuracy_score(Y_test, test_pred)
    train_acc = accuracy_score(Y_train, train_pred)
    print(i,'Train score:',train_acc,'Test score:',test_acc)

    list_score.append([i,accuracy_score(train_pred, Y_train),accuracy_score(test_pred, Y_test)]) 
    
    
df2 = DataFrame(list_score,columns=['Depth','Train Accuracy','Test Accuracy'])
plt.plot(df2['Depth'],df2['Test Accuracy'],label='Test Accuracy')
plt.plot(df2['Depth'],df2['Train Accuracy'],label='Train Accuracy')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.legend()


# In[98]:


dtc = tree.DecisionTreeClassifier(max_depth = 4 ,random_state = 0)
dtc.fit(X_train,Y_train)

train_pred = dtc.predict(X_train)
   #train_acc.append(score(train_pred, yTrain))
   
test_pred = dtc.predict(X_test)
   #test_acc.append(score(test_pred, yTest))
test_acc = accuracy_score(Y_test, test_pred)
train_acc = accuracy_score(Y_train, train_pred)
print('Train score:',train_acc,'Test score:',test_acc)


# ## 3.) Gradient Boost:
# 
# Applied gradient boosting, an ensemble learning technique, to improve model performance. It combines multiple weak learners sequentially, each correcting errors of its predecessor. During training, it assigns higher weights to misclassified instances, focusing on difficult cases. 
# 
# By aggregating predictions, it creates a strong model that excels in predictive accuracy. Gradient boosting is widely used due to its ability to handle complex relationships and reduce overfitting.

# In[99]:


clf = GradientBoostingClassifier(random_state=0)
clf.fit(X_train, Y_train)

train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)


test_acc_grad = accuracy_score(Y_test, test_predict)
train_acc_grad = accuracy_score(Y_train, train_predict)
print('Train score:',train_acc_grad,'Test score:',test_acc_grad)


# ## 4.) Random forest:
# 
# Utilized random forest, an ensemble learning algorithm, for robust predictions. Comprising multiple decision trees, it reduces overfitting by aggregating their outputs. Each tree is trained on a random subset of data and features, enhancing diversity. 
# 
# By averaging or voting over individual tree predictions, random forest provides accurate results, handles noisy data, and identifies important features. It's suitable for classification and regression tasks and is resistant to outliers

# In[100]:


from sklearn.ensemble import RandomForestClassifier  
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
classifier.fit(X_train, Y_train)  


# In[101]:


#Predicting the test set result  
Y_Pred_random= classifier.predict(X_test)  

train_predict_random = classifier.predict(X_train)



# In[102]:


#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(Y_test, Y_Pred_random)
cm


# In[103]:


test_acc_random= accuracy_score(Y_test, Y_Pred_random)
train_acc_random = accuracy_score(Y_train, train_predict_random)
print('Train score:',train_acc_random,'Test score:',test_acc_random)


# ## Conclusion:

# In[104]:


Algo_data=pd.DataFrame()
Algo_data['Model']=['Logistic Regression','Decision Tree','Gradient Boost','Random Forest']
Algo_data['Train_Accuracy']=[logreg.score(rescaledxTrain,Y_train),train_acc,train_acc_grad,train_acc_random]
Algo_data['Test_Accuracy']=[logreg.score(rescaledxTest,Y_test),test_acc,test_acc_grad ,test_acc_random]
Algo_data


# In our analysis, we evaluated four different models on the given dataset.
# 
# -The Logistic Regression model showed solid performance with a train accuracy of 89.8% and a test accuracy of 91.3%.
# 
# -The Decision Tree model achieved a train accuracy of 90.7% and a test accuracy of 90.8%, indicating its effectiveness.
# 
# -Gradient Boost exhibited strong predictive capabilities, achieving a train accuracy of 94.3% and a test accuracy of 91.3%. 
# 
# -However, the Random Forest model emerged as the top performer, attaining a train accuracy of 97.8% and a test accuracy of 94.9%.
# 
# 
# Based on these results, we conclude that the Random Forest model is the most suitable choice for this dataset, offering both high training accuracy and strong generalization to unseen data.

# In[ ]:




