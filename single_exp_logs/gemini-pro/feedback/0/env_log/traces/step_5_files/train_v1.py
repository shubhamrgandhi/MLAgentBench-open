import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DIMENSIONS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics_for_regression(y_test, y_test_pred):
    metrics = {}
    for task in DIMENSIONS:
        targets_task = [t[DIMENSIONS.index(task)] for t in y_test]
        pred_task = [l[DIMENSIONS.index(task)] for l in y_test_pred]
        
        rmse = mean_squared_error(targets_task, pred_task, squared=False)

        metrics[f"rmse_{task}"] = rmse
    
    return metrics

def train_model(X_train, y_train, X_valid, y_valid):
    # define and train the model
    # should return the trained model
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-squadv1")
    model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-squadv1")

    train_encodings = tokenizer(X_train, truncation=True, padding=True)
    valid_encodings = tokenizer(X_valid, truncation=True, padding=True)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']), torch.tensor(train_encodings['attention_mask']), torch.tensor(y_train))
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid_encodings['input_ids']), torch.tensor(valid_encodings['attention_mask']), torch.tensor(y_valid))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(3):
        model.train()
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            for batch in valid_dataloader:
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                outputs = model(**inputs)
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    return model

def predict(model, X):
    # TODO. predict the model
    # should return an array of predictions
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-squadv1")
    test_encodings = tokenizer(X, truncation=True, padding=True)
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_encodings['input_ids']), torch.tensor(test_encodings['attention_mask']))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_pred = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
            outputs = model.generate(**inputs)
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            y_pred.extend(preds)

    y_pred = np.array([[float(pred.split()[i]) for i in range(1, len(pred.split()))] for pred in y_pred])
    return y_pred

if __name__ == '__main__':

    ellipse_df = pd.read_csv('train.csv', 
                            header=0, names=['text_id', 'full_text', 'Cohesion', 'Syntax', 
                            'Vocabulary', 'Phraseology','Grammar', 'Conventions'], 
                            index_col='text_id')
    ellipse_df = ellipse_df.dropna(axis=0)


    # Process data and store into numpy arrays.
    data_df = ellipse_df
    X = list(data_df.full_text.to_numpy())
    y = np.array([data_df.drop(['full_text'], axis=1).iloc[i] for i in range(len(X))])

    # Create a train-valid split of the data.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    # define and train the model
    # should fill out the train_model function
    model = train_model(X_train, y_train, X_valid, y_valid)

    # evaluate the model on the valid set using compute_metrics_for_regression and print the results
    # should fill out the predict function
    y_valid_pred = predict(model, X_valid)
    metrics = compute_metrics_for_regression(y_valid, y_valid_pred)
    print(metrics)
    print("final MCRMSE on validation set: ", np.mean(list(metrics.values())))

    # save submission.csv file for the test set
    submission_df = pd.read_csv('test.csv',  header=0, names=['text_id', 'full_text'], index_col='text_id')
    X_submission = list(submission_df.full_text.to_numpy())
    y_submission = predict(model, X_submission)
    submission_df = pd.DataFrame(y_submission, columns=DIMENSIONS)
    submission_df.index = submission_df.index.rename('text_id')
    submission_df.to_csv('submission.csv')