# Import helpful libraries
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the data, and separate the target
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice

# Select only the following features: 'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars'
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars']

# Select columns corresponding to features, and preview the data
X = home_data[features]

# Split into testing and training data
train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state=1)

# Train a Gradient Boosting Regressor model on the training data and evaluate it on the validation data
model = GradientBoostingRegressor(n_estimators=100, random_state=1)
model.fit(train_X, train_y)
train_mae = mean_absolute_error(train_y, model.predict(train_X))
valid_mae = mean_absolute_error(valid_y, model.predict(valid_X))

# Print the results of training and validation
print("Train MAE: {:,.0f}".format(train_mae))
print("Validation MAE: {:,.0f}".format(valid_mae))

# Make predictions on the test data
test_data = pd.read_csv('test.csv')
test_X = test_data[features]
test_preds = model.predict(test_X)

# Create a submission file
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)