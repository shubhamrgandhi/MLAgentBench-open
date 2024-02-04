from datasets import load_dataset
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

if __name__ == "__main__":
    imdb = load_dataset("imdb")

    # Preprocess data
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    imdb = imdb.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)

    # Define model
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    # Train model
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=imdb["train"],
        eval_dataset=imdb["test"],
    )

    trainer.train()

    # Evaluate model and print accuracy on test set, also save the predictions of probabilities per class to submission.csv
    submission = pd.DataFrame(columns=list(range(2)), index=range(len(imdb["test"])))
    acc = 0
    for idx, data in enumerate(imdb["test"]):
        text = data["text"]
        label = data["label"]
        pred = model(torch.tensor([data["input_ids"]])).logits # TODO: replace with proper prediction
        pred = torch.softmax(pred, dim=0)
        submission.loc[idx] = pred.tolist()
        acc += int(torch.argmax(pred).item() == label)
    print("Accuracy: ", acc/len(imdb["test"]))

    submission.to_csv('submission.csv', index_label='idx')