#!/usr/bin/env python
# coding: utf-8

# In[812]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[813]:


import os


# In[814]:


os.listdir("../Data")


# In[815]:


data = pd.read_csv("../Data/heart.csv")


# In[816]:


data.head(4)


# In[817]:


data.describe()


# In[818]:


data.shape


# In[819]:


data.columns


# In[820]:


data.columns = ["age", "sex", "chest_pain_type", "resting_blood_pressure", "serum_cholestoral", "fasting_blood_sugar", "resting_electrocardiographic_results", "maximum_heart_rate_achieved", "exercise_induced_angina", "depression_induced_by_exercise_relative_to_rest", "slope_of_the_peak_exercise", "number_of_major_vessels", "thalassemia", "angiographic_disease_status"]


# In[821]:


data.head(5)


# In[822]:


for col in data.columns:
    print("{}\n\n".format(col), data[col].isnull().value_counts())
#data['age'].isnull().value_counts()


# In[823]:


#data.age.drop_duplicates()
data.rename(columns={"angiographic_disease_status":"target"}, inplace=True)


# In[824]:


data.duplicated().value_counts()


# In[825]:


data.drop_duplicates(inplace=True)
data.duplicated().value_counts()


# In[826]:


data.dtypes


# In[827]:


sns_data = data.corr()
plt.figure(figsize=(15,10))
sns.heatmap(sns_data, annot=True, fmt=".2f")
plt.show()


# In[828]:


#fig , axs = plt.subplot(2,3)
#fig , axs = plt.subplots()
#axs.set_title("mk")
#axs.bo


# In[829]:


plt_data = [data.depression_induced_by_exercise_relative_to_rest, data.target]


# In[830]:


["resting_blood_pressure", "serum_cholestoral", "fasting_blood_sugar", "maximum_heart_rate_achieved", "depression_induced_by_exercise_relative_to_rest", "number_of_major_vessels", "thalassemia"]


# In[831]:


plt.boxplot(plt_data)
plt.show()


# In[832]:


for col in data.columns:
    plt_data = [data[col], data.target]
    plt.title(col)
    plt.boxplot(plt_data, 'gD')
    plt.show()


# In[833]:


box_plot_data = [data.resting_blood_pressure, data.serum_cholestoral, data.maximum_heart_rate_achieved, data.depression_induced_by_exercise_relative_to_rest, data.thalassemia]
#box_plot_data[3]
box_plot_names = ["resting_blood_pressure", "serum_cholestoral", "maximum_heart_rate_achieved"]


# In[834]:


#fig , axis = plt.subplots(3,2)
for col in box_plot_data:
    plt_data = [col]#, data.target]
    plt.title(col.name)
    plt.boxplot(plt_data, 0, 'gD')
    plt.show()


# In[835]:


box_plot_names = ["resting_blood_pressure", "serum_cholestoral", "maximum_heart_rate_achieved"]


# In[836]:


data.resting_blood_pressure.quantile(q=0.95)


# In[837]:


m = data.loc[lambda data: data['resting_blood_pressure'] == 120.0]
m.shape[0]


# In[838]:


for col in data.columns:
    print("{}".format(col))
    m = data.loc[lambda data: data[col] == data[col].quantile(q=0.25)]
    n = data.loc[lambda data: data[col] == data[col].quantile(q=0.75)]
    print("In the lower limit are: {}\nIn the upper limit are: {}".format(m.shape[0], n.shape[0]))


# In[839]:


box_plot_names = ["", "maximum_heart_rate_achieved"]


# In[840]:


"""print("serum_cholestoral")
print(data.shape)
m = data.loc[lambda data: data["serum_cholestoral"] == data["serum_cholestoral"].quantile(q=0.25)]
n = data.loc[lambda data: data["serum_cholestoral"] == data["serum_cholestoral"].quantile(q=0.75)]
print("In the lower limit are: {}\nIn the upper limit are: {}".format(m.shape[0], n.shape[0]))
print(data.shape)"""


# In[841]:


def remove_outliers(df_in, col_name):
    q1 = data[col_name].quantile(q=0.25)
    q4 = data[col_name].quantile(q=0.75)
    interquartile = q4 - q1
    low_lim = q1 - 1.5 * interquartile
    upper_lim = q1 - 1.5 * interquartile
    df_out = df_in.loc[(df_in[col_name] > low_lim) and (df_in[col_name] < upper_lim)]
    return df_out


# In[842]:


for col in data.columns:
    print("{} has {} unique values".format(col, data[col].nunique()))


# In[843]:


data = data[data.thalassemia != 0]
data.head()


# In[844]:


data.number_of_major_vessels.value_counts()


# In[845]:


data = data[data.number_of_major_vessels != 4]
data.shape


# In[846]:


def categorizing(df_in, col_name):
    col_name = str(col_name)
    df_in[col_name].astype("object")
    dummies = pd.get_dummies(df_in[col_name])
    if col_name == "sex":
        dummies.columns = ["female","male"]
    elif col_name == "chest_pain_type":
        dummies.columns = ["typical_angina", "atypical_angina", "non_anginal_pain", "asymptomatic"]
    elif col_name == "fasting_blood_sugar":
        dummies.columns = ["below_120","above_120"]
    elif col_name == "resting_electrocardiographic_results":
        dummies.columns = ['normal','ST_T_abnormality','showing_probable']
    elif col_name == "exercise_induced_angina":
        dummies.columns = ["no_exercise_induced_angina", "yes_exercise_induced_angina"]
    elif col_name == "slope_of_the_peak_exercise":
        dummies.columns = ['upsloping','flat','downsloping']
    elif col_name == "thalassemia":
        dummies.columns = ['normal', 'fixed_defect', 'revesable_defect']
    df_in = pd.concat([df_in, dummies], axis=1)
    df_in.drop(col_name, axis=1, inplace=True)
    return df_in


# In[847]:


m.columns


# In[848]:


cols = ['sex', 'chest_pain_type', 'fasting_blood_sugar','resting_electrocardiographic_results', 'exercise_induced_angina','slope_of_the_peak_exercise', 'thalassemia']
for col in cols:
    data = categorizing(df_in=data, col_name=col)
data.columns


# In[849]:


data.head(3)
#m['thalassemia']


# In[850]:


sns_data = data.corr()
plt.figure(figsize=(15,10))
sns.heatmap(sns_data, annot=True, fmt=".2f")
plt.show()


# In[860]:


counter = 0
for col in data.columns:
    data.fi
    #print(data[data[col] == -9.0])
        #counter = counter + 1
print(counter)


# In[525]:





# In[861]:


data.head()


# In[862]:


data.to_csv("processed_data.csv")


# In[671]:


data.dtypes


# In[ ]:


data['sex'][data['sex'] == 0] = 'female'
data['sex'][data['sex'] == 1] = 'male'

data['chest_pain_type'][data['chest_pain_type'] == 1] = 'typical_angina'
data['chest_pain_type'][data['chest_pain_type'] == 2] = 'atypical_angina'
data['chest_pain_type'][data['chest_pain_type'] == 3] = 'non-anginal_pain'
data['chest_pain_type'][data['chest_pain_type'] == 4] = 'asymptomatic'

data['fasting_blood_sugar'][data['fasting_blood_sugar'] == 0] = 'below_120'
data['fasting_blood_sugar'][data['fasting_blood_sugar'] == 1] = 'above_120'

data['resting_electrocardiographic_results'][data['resting_electrocardiographic_results'] == 0] = 'normal'
data['resting_electrocardiographic_results'][data['resting_electrocardiographic_results'] == 1] = 'ST_T_abnormality'
data['resting_electrocardiographic_results'][data['resting_electrocardiographic_results'] == 2] = 'probable'

data['exercise_induced_angina'][data['exercise_induced_angina'] == 0] = 'no'
data['exercise_induced_angina'][data['exercise_induced_angina'] == 1] = 'yes'

data['slope_of_the_peak_exercise'][data['slope_of_the_peak_exercise'] == 1] = 'upsloping'
data['slope_of_the_peak_exercise'][data['slope_of_the_peak_exercise'] == 2] = 'flat'
data['slope_of_the_peak_exercise'][data['slope_of_the_peak_exercise'] == 3] = 'downsloping'

data['thalassemia'][data['thalassemia'] == 1] = 'normal'
data['thalassemia'][data['thalassemia'] == 2] = 'fixed_defect'
data['thalassemia'][data['thalassemia'] == 3] = 'reversable_defect'

