# Step 1: Gather Required Resources

# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('data.csv')

# Check the data for missing values
print(data.isnull().sum())

# Drop the rows with missing values
data = data.dropna()

# Check the data for outliers
sns.boxplot(data=data)
plt.show()

# Remove the outliers
data = data[~((data - data.mean()).abs() > (3 * data.std())).any(axis=1)]

# Check the data for normality
sns.distplot(data['feature_1'])
plt.show()

# Transform the data to normality
data['feature_1'] = np.log(data['feature_1'])

# Check the data for normality again
sns.distplot(data['feature_1'])
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Evaluate the model on the test data
score = model.score(X_test, y_test)
print('The score of the model is:', score)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Plot the actual and predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()

# Calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print('The mean absolute error of the model is:', mae)

# Calculate the root mean squared error
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('The root mean squared error of the model is:', rmse)

# Save the model
joblib.dump(model, 'model.joblib')