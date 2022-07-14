#!/usr/bin/env python
# coding: utf-8

# # Review module

# **Instructions**
# 
# In order to complete this review module, we recommend you follow these instructions:
# 
# 1. Complete the functions provided to you in this notebook, but do **not** change the name of the function or the name(s) of the argument(s). If you do that, the autograder will fail and you will not receive any points.
# 2. Run all the function-definition cells before you run the testing cells. The functions must exist before they are graded!
# 3. Read the function docstrings carefully. They contain additional information about how the code should look (a [docstring](https://www.datacamp.com/community/tutorials/docstrings-python) is the stuff that comes between the triple quotes).
# 4. Some functions may require several outputs (the docstrings tell you which ones). Make sure they are returned in the right order.
# 5. Remove from each function the code `raise NotImplementedError()` and replace it with your implementation.

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
import pingouin
from scipy.stats import chi2_contingency


# In[2]:


# Loading the data
df = pd.read_csv("data/bank-full.csv")
df.head(3)


# ### Exercise 1
# 
# Calculate the average (mean) balance of married people vs. single people.

# In[3]:


def mean_balance_married_single(df):
    """
    Calculate the mean balance of married people vs. single people.
    
    Arguments:
    df: A Pandas dataframe (the dataset)
    
    Output:
    mean_married: A number of type float or numpy float64. The mean balance for married people.
    mean_single: A number of type float or numpy float64. The mean balance for single people.
    
    The function outputs these two objects in that order.
    
    """
    return df[df["marital"] =='married']['balance'].mean(), df[df["marital"] =='single']['balance'].mean()


# ### Exercise 2
# 
# Conduct a two-sample $t$ - test to determine whether the mean balance of married vs. single people is statistically different. Report the $p$ - value and a boolean that takes the value `True` if the difference is statistically significant at a 0.05 level, and `False` otherwise. We have given you the first few lines of code to help you get started.
# 
# **Hint:** If you find yourself needing to slice a Dataframe, you can use the [`.loc`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html) method. Make sure all the outputs are of the right types before you submit your assignment.

# In[4]:


def test_difference_married_single(df):
    """
    Test whether there is a statistically significant difference
    between the mean balances of married and single people.
    
    Arguments:
    df: A Pandas dataframe (the dataset)
    
    Outputs:
    p: A number of type float, the p-value
    sig: A boolean; True if difference is significant, False otherwise    
    """
    married_balance = df[df['marital']=='married']['balance']
    single_balance = df[df['marital']=='single']['balance']
    stats=pingouin.ttest(married_balance,  single_balance)
    
    return float(stats['p-val'][0]), bool(stats['p-val'][0]<0.05)


# ### Exercise 3
# 
# Determine whether the `job` and `education` variables are independent or not. Use a significance level of 0.05.

# In[5]:


def perform_chisq(series1, series2):
    """
    Perform a chi-square test between `series1` and `series2`
    and report the p-value and whether the difference is significant or not
    
    Arguments:
    series1, series2: Two pandas series (categorical data)
    
    Outputs:
    p: A number of type float, the p-value
    sig: A boolean; True if difference is significant, False otherwise
    """
    my_contingency_table = pd.crosstab(index=series1, columns=series2)
    chi2_contingency(my_contingency_table)[1]
        
    return float(chi2_contingency(my_contingency_table)[1]), bool(chi2_contingency(my_contingency_table)[1]<0.05)


# ## Testing Cells
# 
# Run the below cells to check your answers. Make sure you run your solution cells first before running the cells below, otherwise you will get a `NameError` when checking your answers.

# In[6]:


# Ex 1
tnpf = type(np.float64(4))
tfl = type(15.32)
assert type(mean_balance_married_single(df)[0]) in [tnpf, tfl], "mean_married is not a float or numply float64! Check it with type(). If you have a series s and want to extract only one of its elements, you can use s.loc['x'], where x is the label of the index (x is a string)"
assert type(mean_balance_married_single(df)[1]) in [tnpf, tfl], "mean_single is not a float or numply float64! Check it with type(). If you have a series s and want to extract only one of its elements, you can use s.loc['x'], where x is the label of the index (x is a string)"
mm = 1425.9255897699713
ms = 1301.4976544175138
assert abs(mean_balance_married_single(df)[0] - mm) < 1, "Please check your results. The mean_married you calculated is way off the true value. Are you sure you're using the right column? If you're filtering columns, check out this SO answer for a refresher: https://stackoverflow.com/a/17071908"
assert abs(mean_balance_married_single(df)[1] - ms) < 1, "Please check your results. The mean_single you calculated is way off the true value. Are you sure you're using the right column? If you're filtering columns, check out this SO answer for a refresher: https://stackoverflow.com/a/17071908"
print("Exercise 1 looks correct!")


# In[7]:


# Ex 2
f = 0.12
assert type(test_difference_married_single(df)[0]) == type(f), "Exercise 2: Your p-value isn't a float number. You can check with type()."
assert type(test_difference_married_single(df)[1]) == type(False), "Exercise 2: Your significance flag isn't a boolean. You can check with type()."
assert test_difference_married_single(df)[0] <= 1, "Exercise 2: Your p-value seems to be greater than one. Since p-values are probabilities, they can never be greater than one."
assert test_difference_married_single(df)[1] == True, "Exercise 2: Are you sure you're using a significance level of 0.05? Recall that the difference is significant when your p-value is **lower** than your threshold."
print("Exercise 2 looks correct!")


# In[8]:


# Ex 3
assert perform_chisq(df["job"], df["education"])[0] < 0.05, "Ex. 3 - Your p-value seems to be too large! Did you use the chi2_contingency function and passed a contingency table as its argument?"
assert perform_chisq(df["job"], df["education"])[1] == True, "Ex. 3 - Check your results! Did you use a significance level of 0.05?"
print("Exercise 3 looks correct!")


# ## Attribution
# 
# "Bank Marketing dataset", May 7, 2018, Sandeep Verma, CC0 Public Domain, https://www.kaggle.com/skverma875/bank-marketing-dataset
