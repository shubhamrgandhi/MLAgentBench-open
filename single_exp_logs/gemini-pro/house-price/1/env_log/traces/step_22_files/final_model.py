# Import helpful libraries
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
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

# ***********************************************
# In this part of the code, write and train the model on the above dataset to perform the task.
# This part should populate the variable train_mae and valid_mae on the model selected
# ***********************************************

# Create a Gradient Boosting Regressor model
gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=1)

# Train the Gradient Boosting Regressor model on the training data
gbr_model.fit(train_X, train_y)

# Evaluate the Gradient Boosting Regressor model on the training data
train_gbr_mae = mean_absolute_error(train_y, gbr_model.predict(train_X))

# Evaluate the Gradient Boosting Regressor model on the validation data
valid_gbr_mae = mean_absolute_error(valid_y, gbr_model.predict(valid_X))

# ***********************************************
# End of the main training module
# ***********************************************

print("Gradient Boosting Regressor Train MAE: {:,.0f}".format(train_gbr_mae))
print("Gradient Boosting Regressor Validation MAE: {:,.0f}".format(valid_gbr_mae))

test_data = pd.read_csv('test.csv')
test_X = test_data[features]
test_preds = gbr_model.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)