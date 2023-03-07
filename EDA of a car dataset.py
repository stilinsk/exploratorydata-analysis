#!/usr/bin/env python
# coding: utf-8

# **The begginning of exploratory data analysis using python otherwise known as EDA**
# 

# Start by importing all the required libraries for data analysis in python, 
# if by any chance you ever import a library in any python work space and it does not work start
# with this command '!pip install[the package]'

# In[4]:


import pandas as pd
df=pd.read_csv('car_prices.csv')
df1=pd.read_csv('car_prices_train.csv')
df2=pd.read_csv('car_prices_test.csv')


# after importing al the libraries required we can execute the below command to have aview of the data

# In[5]:


df.head()


# if we are merging different files we need to first read the files individually,
# to determine whether the different data in the data sets is similar or not 

# In[7]:


df1.head()


# In[8]:


df2.head()


# After we have examined all the different values and how they relate to one another,
# the next step is joining the different datasets to create one data set .we use the method [concat] 

# In[10]:


combined_df = pd.concat([df, df1, df2])
combined_df


# The most basic and essential skill as a datascientist is after examining the data how do you clean it 

# we first find whether there are any duplicated values this is neccesary so as to
# remove any unnecessary data

# In[11]:


df.duplicated().value_counts


# we need to keep all the duplicates 'FALSE'so as to put duplicates next to
# each other the sort values after that will be used to group th new data

# In[12]:


df[df.duplicated(keep=False)].sort_values(by='model')


# later we need to drop the duplicates

# In[13]:


new=df.drop_duplicates()
new


# next we look for the new shape of our new data

# In[14]:


combined_df.shape


# we need to check for missing data

# In[15]:


df.isna()


# In[16]:


df.isna().sum()


# we can get the number of all cars sold in a certain year and we can later assign a variable to draw different
# visualizations models

# In[45]:


df["year"].value_counts().sort_index()


# next we need to group or data to dive deep in data analysis
# we can start by grouping the data and aggregating all the necessary data

# In[29]:


grouped_df = df.groupby(['make', 'model']).agg({'price': 'mean','state':'count'})
grouped_df


# After grouping the data it is clearly shown the pefomance of different models of differnt cars in different states

#  **WE NOW NEED TO SET OBJECTIVES WHILE USING THE DATA*
#    We load the data(which is already done)
#   we need to view the distribution of data this is discussed in details below
#   we need to get correlation between different columns
#   
#    However this only applies to our data different datasets will have different objectives in respect to different firms.

# **First we start by importing the required libraries**

# In[1]:


import matplotlib .pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# we need to explore the data set

# In[18]:


combined_df.info


# In[19]:


combined_df.describe


# **EXPLORE DATA DISTRIBUTIONS**

# we create a histogram for price of cars 

# In[20]:


df['price'].plot.hist(title ='price ranges of cars')
plt.axvline(df['price'].mean(),color ='k')


# next we calculate the mean, median and standard deviation for ptice column

# In[21]:


print(df['price'].mean())
print(df['price'].median())
print(df['price'].std())


# we will now create a pivot table to show how tha sales of different cars faired overtime

# In[38]:


# group the data by year and car model and count the number of occurrences
df_grouped = df.groupby(['year', 'model']).size().reset_index(name='count')

# plot the data using a pivot table to create a separate bar for each model
df_pivot = df_grouped.pivot(index='year', columns='model', values='count')
df_pivot.plot(kind='bar', stacked=True)
plt.ylabel('Number of cars sold')
plt.show()


# with the above pivot table we can be able to see how the sales of different cars faired overtime

# here we will be able to use the library known as **seaborn**

# In[46]:


sns.countplot(data=combined_df,x='model')
plt.xticks(rotation=90);


# we can now see the total numbers of cars and we see which model made the most moey overall

# we will now test the correlation between the different columns in the dataset

# In[43]:


# Calculate correlation coefficients
corr_matrix = combined_df.corr()

# Create a heatmap
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# Simon Kamande
