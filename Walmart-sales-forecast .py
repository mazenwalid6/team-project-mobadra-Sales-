#!/usr/bin/env python
# coding: utf-8

# In[71]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose


# In[21]:


# 1. Data Overview
""" Which columns are in the dataset? 
What are their data types and shapes? 
Are there any missing values?"""

""" Answer:
What columns do you see?
Are there missing values?
How many stores and departments are there?"""

# Load the data
train_df = pd.read_csv("train.csv")
stores_df = pd.read_csv("stores.csv")
features_df = pd.read_csv("features.csv")

# Check the first few rows
print(f"\n{train_df.head()}")
print(f"\n{stores_df.head()}")
print(f"\n{features_df.head()}")


# In[22]:


print("\nTrain Data Info:")
train_df.info()

print("\nFeatures Data Info:")
features_df.info()

print("\nStores Data Info:")
stores_df.info()


# In[23]:


# Check shape and data types
print(f"\nDataset Shape: {train_df.shape}")


# In[24]:


# Check for missing values
print(f"\nMissing Values: \n{train_df.isnull().sum()} ")
print(f"\nMissing Values: \n{stores_df.isnull().sum()} ")
print(f"\nMissing Values: \n{features_df.isnull().sum()} ")


# In[25]:


# Check for number of stores & dept. in each store
print(f"\nNumber of stores: \n{train_df['Store'].nunique()}")
print(f"\nNumber of departments : \n{train_df['Dept'].nunique()}")
print(f"\nDepartment in each store : \n{train_df.groupby('Store')['Dept'].nunique()}")


# In[81]:


# Merge train and stores data
merged_df = pd.merge(train_df, stores_df, on='Store')

store_dept_table = pd.pivot_table(merged_df, index='Store', columns='Dept',values='Weekly_Sales', aggfunc=np.mean)
display(store_dept_table)


# In[82]:


""""it is obviously seen that there are some wrong values such as there are 0 and minus values for weekly sales. 
But sales amount can not be minus. Also, it is impossible for one department not to sell anything whole week.
So, change this values."""

merged_df.loc[merged_df['Weekly_Sales']<=0]

"""1358 rows in 421570 rows means 0.3%, 
so I can delete and ignore these rows which contains wrong sales values."""

merged_df = merged_df.loc[merged_df['Weekly_Sales'] > 0]

merged_df.shape 


# In[83]:


# 2. Sales Trends & Distributions
"""What are the overall sales trends over time? 
How do sales vary across stores?"""

# Convert date to datetime format
train_df['Date'] = pd.to_datetime(train_df['Date'])

# Group sales by date
sales_trend = train_df.groupby('Date')['Weekly_Sales'].sum().reset_index()

# Plot sales over time
plt.figure(figsize=(12, 6))
plt.plot(sales_trend['Date'], sales_trend['Weekly_Sales'])
plt.xlabel('Date')
plt.ylabel('Total Weekly Sales')
plt.title('Overall Sales Trend Over Time')
plt.grid()
plt.show()

"""Answer:
Do you notice any trends or seasonality?
Are there sharp peaks (holiday seasons)?"""


# In[85]:


# Convert date to datetime format
train_df['Date'] = pd.to_datetime(train_df['Date'])

# Group sales by date
sales_trend = train_df.groupby('Date')['Weekly_Sales'].sum().reset_index()
holiday_sales_trend = train_df[train_df['IsHoliday']].groupby('Date')['Weekly_Sales'].sum().reset_index()


# Plot sales over time
sns.set(style="whitegrid")
plt.figure(figsize=(14, 8), dpi=100)
plt.plot(sales_trend['Date'], sales_trend['Weekly_Sales'], label='Overall Sales', color='blue')
plt.plot(holiday_sales_trend['Date'], holiday_sales_trend['Weekly_Sales'], color='red', label='Holiday Sales', linestyle='--')
plt.fill_between(holiday_sales_trend['Date'], 0, holiday_sales_trend['Weekly_Sales'], color='red', alpha=0.3)

# Title and labels
plt.title('Overall Sales Trend Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Weekly Sales', fontsize=14)

# Grid and spines
plt.grid(True, linestyle='--', linewidth=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Format x-axis
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.legend()

# Add annotations for major holidays 
major_holidays = ['2010-12-25', '2011-12-25', '2012-12-25']
for holiday in major_holidays:
    plt.axvline(pd.to_datetime(holiday), color='grey', linestyle='--', linewidth=1)
    plt.text(pd.to_datetime(holiday), max(sales_trend['Weekly_Sales']), 'Christmas', rotation=45, fontsize=12, ha='right')


# In[106]:


# Group sales by date and ensure the index is in datetime format
sales_trend = train_df.groupby('Date')['Weekly_Sales'].sum()
sales_trend.index = pd.to_datetime(sales_trend.index)

# Time series decomposition
decomposition = seasonal_decompose(sales_trend, model="additive", period=12)  
# Plot the three components: Trend, Seasonality, and Residuals
plt.figure(figsize=(14, 10))

# Sales trend
plt.subplot(3, 1, 1)
plt.plot(decomposition.trend, label="Trend", color="blue")
plt.title("Sales Trend ")
plt.legend()

# Seasonality
plt.subplot(3, 1, 2)
plt.plot(decomposition.seasonal, label="Seasonality", color="green")
plt.title("Seasonality ")
plt.legend()

# Residuals
plt.subplot(3, 1, 3)
plt.plot(decomposition.resid, label="Residuals", color="red")
plt.title("Residuals ")
plt.legend()

plt.tight_layout()
plt.show()


# In[91]:


#  3. Store-Level Analysis
"""How do sales differ across stores? 
Are larger stores making more sales?"""

# Calculate average sales per store
store_sales = merged_df.groupby('Store')['Weekly_Sales'].mean().reset_index()

# Calculate Average sales of store according to store size
Store_size_sales = merged_df.groupby('Size')['Weekly_Sales'].mean().reset_index()

# Sort and visualize
store_sales = store_sales.sort_values(by='Weekly_Sales', ascending=False)
Store_size_sales = Store_size_sales.sort_values(by='Weekly_Sales', ascending=True)


plt.figure(figsize=(12, 6))
plt.bar(store_sales['Store'], store_sales['Weekly_Sales'], color='skyblue')
plt.xlabel('Store')
plt.ylabel('Average Weekly Sales')
plt.title('Average Weekly Sales by Store')

# Store-wise sales
store_sales = merged_df.groupby("Store")["Weekly_Sales"].sum().reset_index().sort_values("Weekly_Sales", ascending=False)
print("Top 5 Stores by Total Sales:\n", store_sales.head())

""" Answer:
Which stores perform the best?
Does store size influence sales?"""


# In[86]:


# Store type vs. sales
store_type_sales = merged_df.groupby("Type")["Weekly_Sales"].mean().reset_index()
print("Average Sales by Store Type:\n", store_type_sales)

# Visualize store size vs. sales
plt.figure(figsize=(8, 5))
sns.barplot(x="Size", y="Weekly_Sales", data=merged_df, alpha=0.5)
plt.title("Store Size vs. Weekly Sales")
plt.show()


# In[60]:


store_size_sales


# In[87]:


# Store department vs. sales
sales_dept = train_df.groupby('Dept')['Weekly_Sales'].sum().reset_index()

# Plot sales over time
plt.figure(figsize=(12, 6))
plt.bar(sales_dept['Dept'], sales_dept['Weekly_Sales'])
plt.xlabel('Dept')
plt.ylabel('Total Weekly Sales')
plt.title('Overall Sales on number of deptartment')
plt.grid()
plt.show() 


# In[64]:


# Merge train and features data
features_df['Date'] = pd.to_datetime(features_df['Date'])
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
full_df = pd.merge(merged_df, features_df, on=['Store', 'Date'])

full_df.drop(columns=["IsHoliday_y"], inplace=True)
full_df.rename(columns={"IsHoliday_x": "IsHoliday"}, inplace=True)
print(full_df)


# In[98]:


#   4. Feature Impact on Sales
"""How do features like temperature, fuel price, and holidays affect sales?"""

# Check correlations
correlation = full_df[['Weekly_Sales', 'Temperature', 'Fuel_Price', 'Unemployment']].corr()
print(correlation)

# Scatter plot for sales vs. temperature
plt.figure(figsize=(8, 6))
plt.scatter(full_df['Temperature'], full_df['Weekly_Sales'], alpha=0.3)
plt.xlabel('Temperature')
plt.ylabel('Weekly Sales')
plt.title('Sales vs. Temperature')
plt.grid()
plt.show()

# Pairplot visualization
sns.pairplot(full_df[["Weekly_Sales", "Temperature", "Fuel_Price", "Unemployment"]], diag_kind="kde")
plt.show()
"""Answer:

Is there any correlation between sales and external factors?
How does seasonality impact sales?"""


# In[93]:


#  5. Holiday Sales Impact
"""Do sales spike during holidays? 
Which holidays have the biggest impact?"""

# Check sales on holidays
holiday_sales = full_df[full_df['IsHoliday'] == True].groupby('Date')['Weekly_Sales'].sum().reset_index()

# Plot holiday sales
plt.figure(figsize=(12, 6))
plt.plot(holiday_sales['Date'], holiday_sales['Weekly_Sales'], color='red', label='Holiday Sales')
plt.xlabel('Date')
plt.ylabel('Total Weekly Sales')
plt.title('Sales During Holiday Weeks')
plt.legend()
plt.grid()
plt.show()

# Check holiday impact
holiday_sales = full_df.groupby("IsHoliday")["Weekly_Sales"].mean()
print("Average Sales During Holidays vs. Non-Holidays:\n", holiday_sales)
"""Answer:

Do sales consistently spike on holidays?
Which holiday weeks have the highest sales?"""


# In[96]:


# Visualize holiday effect
plt.figure(figsize=(8, 4))
sns.boxplot(x="IsHoliday", y="Weekly_Sales", data=full_df)
plt.title("Sales Distribution During Holidays vs. Non-Holidays")
plt.show()


# In[92]:


# Convert date to datetime format
full_df['Date'] = pd.to_datetime(full_df['Date'])

# Group sales by date and holiday status
sales_trend = full_df.groupby(['Date', 'IsHoliday'])['Weekly_Sales'].sum().reset_index()

# Split the sales data into holiday and non-holiday sales
holiday_sales = sales_trend[sales_trend['IsHoliday'] == True]
non_holiday_sales = sales_trend[sales_trend['IsHoliday'] == False]

# Calculate average sales for holidays and non-holidays
average_holiday_sales = holiday_sales['Weekly_Sales'].mean()
average_non_holiday_sales = non_holiday_sales['Weekly_Sales'].mean()

# Set Seaborn style
sns.set(style="whitegrid")

# Plot sales over time
plt.figure(figsize=(14, 8), dpi=100)

# Non-holiday sales
plt.plot(non_holiday_sales['Date'], non_holiday_sales['Weekly_Sales'], label='Non-Holiday Sales', color='blue')

# Holiday sales
plt.plot(holiday_sales['Date'], holiday_sales['Weekly_Sales'], label='Holiday Sales', color='red', linestyle='--')

# Highlight holiday weeks
plt.fill_between(holiday_sales['Date'], 0, holiday_sales['Weekly_Sales'], color='red', alpha=0.3)

# Title and labels
plt.title('Sales Trend Over Time with Holiday Impact', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Weekly Sales', fontsize=14)

# Grid and spines
plt.grid(True, linestyle='--', linewidth=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Format x-axis
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)


plt.legend(fontsize=12)

# Add horizontal lines for average sales
plt.axhline(average_holiday_sales, color='red', linestyle='--', linewidth=1, label='Average Holiday Sales')
plt.axhline(average_non_holiday_sales, color='blue', linestyle='--', linewidth=1, label='Average Non-Holiday Sales')

# Add annotations for major holidays
major_holidays = {
    'Christmas': '2010-12-25',
    'Thanksgiving': '2010-11-26',
    'Easter': '2011-04-24'
}
for holiday, date in major_holidays.items():
    plt.axvline(pd.to_datetime(date), color='grey', linestyle='--', linewidth=1)
    plt.text(pd.to_datetime(date), max(holiday_sales['Weekly_Sales']), holiday, rotation=45, fontsize=12, ha='right')

plt.show()


# In[ ]:


""" Next Steps
Outlier Detection: Are there abnormal sales spikes/drops?
Data Cleaning & Transformation: Handle missing data, create new features."""

