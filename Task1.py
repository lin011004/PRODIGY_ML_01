#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the training data
train_data = pd.read_csv("C:\\Users\\Lingesh\\OneDrive\\Desktop\\ML Intern\\house-prices-advanced-regression-techniques\\train.csv")

# Select features (square footage, bedrooms, bathrooms) and target (prices)
X_train = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath','HalfBath']]
y_train = train_data['SalePrice']

# Load the testing data
test_data = pd.read_csv("C:\\Users\\Lingesh\\OneDrive\\Desktop\\ML Intern\\house-prices-advanced-regression-techniques\\test.csv")

# Select features (square footage, bedrooms, bathrooms) for testing
X_test = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath','HalfBath']]

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
predicted_prices = model.predict(X_test)

# Create a DataFrame to store predicted prices with 'ID' and 'SalePrice' columns
submission_df = pd.DataFrame({
    'ID': test_data['Id'],
    'SalePrice': predicted_prices
})

# Save the DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

# Print the contents of the CSV file
print("The Predicted Price:")
print(submission_df)

# Plot a line graph of the predicted prices
plt.plot(submission_df['SalePrice'])
plt.xlabel('House Index')
plt.ylabel('Predicted Price')
plt.title('Predicted Prices of Houses')
plt.show()


# In[ ]:




