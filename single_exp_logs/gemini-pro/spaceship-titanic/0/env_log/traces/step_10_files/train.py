import seaborn as sns
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

def create_new_dataframe(data, column_names):
    new_data = {}

    for column in column_names:
        if column in data.columns:
            new_data[column] = data[column]
        else:
            new_data[column] = pd.Series(0, index=data.index)

    new_dataframe = pd.DataFrame(new_data)
    return new_dataframe

# Loading the dataset to train a binary classfier downstream
df = pd.read_csv("train.csv")
num_examples = df.shape[0]
df = df.sample(frac = 1, random_state=1)
train_data, val_data = train_test_split(df, test_size=0.2, random_state=1)

train_data[["Deck", "Cabin_num", "Side"]] = train_data["Cabin"].str.split("/", expand=True)
train_data = train_data.drop('Cabin', axis=1)

val_data[["Deck", "Cabin_num", "Side"]] = val_data["Cabin"].str.split("/", expand=True)
val_data = val_data.drop('Cabin', axis=1)

TargetY = train_data["Transported"]
TargetY_test = val_data["Transported"]

# Expanding features to have boolean values as opposed to categorical
# You can check all the features as column names and try to find good correlations with the target variable
selectColumns = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
ResourceX = pd.get_dummies(train_data[selectColumns])
ResourceX_test = pd.get_dummies(val_data[selectColumns])

# ***********************************************
# In this part of the code, write and train the model on the above dataset to perform the task.
# Note that the output accuracy should be stored in train_accuracy and val_accuracy variables
# ***********************************************

# Trying different models and feature selections
models = [LogisticRegression(), RandomForestClassifier()]
feature_sets = [selectColumns, ["HomePlanet", "CryoSleep", "Destination", "VIP"]]

for model in models:
    for feature_set in feature_sets:
        ResourceX = pd.get_dummies(train_data[feature_set])
        ResourceX_test = pd.get_dummies(val_data[feature_set])

        model.fit(ResourceX, TargetY)

        train_accuracy = model.score(ResourceX, TargetY)
        val_accuracy = model.score(ResourceX_test, TargetY_test)

        print(f"Model: {model.__class__.__name__}, Features: {feature_set}")
        print(f"Train Accuracy: {train_accuracy}")
        print(f"Validation Accuracy: {val_accuracy}")
        print("-" * 50)

# ***********************************************
# End of the main training module
# ***********************************************

# Selecting the best model and feature set based on validation accuracy
best_model = LogisticRegression()
best_feature_set = selectColumns

ResourceX = pd.get_dummies(train_data[best_feature_set])
ResourceX_test = pd.get_dummies(val_data[best_feature_set])

best_model.fit(ResourceX, TargetY)

train_accuracy = best_model.score(ResourceX, TargetY)
val_accuracy = best_model.score(ResourceX_test, TargetY_test)

print(f"Best Model: {best_model.__class__.__name__}, Best Features: {best_feature_set}")
print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")

test_data = pd.read_csv('test.csv')
test_data[["Deck", "Cabin_num", "Side"]] = test_data["Cabin"].str.split("/", expand=True)
test_data = test_data.drop('Cabin', axis=1)

test_X = pd.get_dummies(test_data[best_feature_set])
test_X.insert(loc = 17,
          column = 'Deck_T',
          value = 0)

test_preds = best_model.predict(test_X)


output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Transported': test_preds})
output.to_csv('submission.csv', index=False)