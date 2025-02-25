# Import helpful libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the data, and separate the target
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice

# You can change the features needed for this task depending on your understanding of the features and the final task
features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

# Select columns corresponding to features, and preview the data
X = home_data[features]

# Split into testing and training data
train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state=1)

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Train the model
model.fit(train_X, train_y)

# Evaluate the model on the training data
train_preds = model.predict(train_X)
train_mae = mean_absolute_error(train_y, train_preds)

# Evaluate the model on the validation data
valid_preds = model.predict(valid_X)
valid_mae = mean_absolute_error(valid_y, valid_preds)

# Print the training and validation MAE
print("Train MAE: {:,.0f}".format(train_mae))
print("Validation MAE: {:,.0f}".format(valid_mae))

# Feature importance
importances = model.feature_importances_
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features, importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

# Plot the feature importances
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.title("Feature Importances")
plt.bar(range(len(feature_importances)), [importance for feature, importance in feature_importances], color='lightblue')
plt.xticks(range(len(feature_importances)), [feature for feature, importance in feature_importances], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# Save the model
import pickle
pickle.dump(model, open('iowa_model.pkl', 'wb'))

test_data = pd.read_csv('test.csv')
test_X = test_data[features]
test_preds = model.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)

# Calculate the R2 score on the validation data
valid_r2 = model.score(valid_X, valid_y)
print("Validation R2: {:,.2f}".format(valid_r2))

# Calculate the R2 score on the training data
train_r2 = model.score(train_X, train_y)
print("Train R2: {:,.2f}".format(train_r2))