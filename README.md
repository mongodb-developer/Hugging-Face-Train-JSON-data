<img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.png" alt="Huggingface Logo" width="300">


**Using HuggingFace AutoTrain JSON chunks**

Extract Data from MongoDB:

Use a MongoDB client (like pymongo) to query and retrieve data from your MongoDB collections.
Convert the retrieved data into a format suitable for training, like a pandas DataFrame.
Preprocess and Train:

Preprocess the data (e.g., tokenization, label encoding) using Hugging Face transformers and datasets.
Train the model as you've done in your script.
Push to Hub or Deploy:

Once trained, you can push the model to the Hugging Face Hub or deploy it in your environment.

First off install the required packages
```
pip install -r requirements.txt
```

In your project folder name it what you wish add some JSON data
```
[
    {
        "text": "This is a positive example.",
        "label": "positive"
    },
    {
        "text": "This is a negative example.",
        "label": "negative"
    },
    {
        "text": "This is another positive example.",
        "label": "positive"
    },
    {
        "text": "This is another negative example.",
        "label": "negative"
    }
]
```
Then run main training script - make sure you are in the same folder as your simple_test_data.json or pass the path to it

```
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from huggingface_hub import login
import torch

# Step 1: Authenticate with Hugging Face using your write token
login(token="put your WRITE token here")  # Replace with your actual Hugging Face write token

# Step 2: Load the JSON file into a DataFrame
df = pd.read_json('simple_test_data.json')

# Step 3: Convert labels to numerical format
label_map = {'positive': 1, 'negative': 0}
df['label'] = df['label'].map(label_map)

# Step 4: Split the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2)

# Step 5: Convert DataFrame to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Step 6: Load the tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Step 7: Tokenize the input
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Step 8: Set the format for PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Step 9: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Step 10: Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Step 11: Train the model
trainer.train()

# Step 12: Evaluate the model
trainer.evaluate()

# Step 13: Save the model and tokenizer locally
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

# Step 14: Push the model and tokenizer to Hugging Face Hub
model.push_to_hub("your_model_name")  # Replace with your desired model name on Hugging Face
tokenizer.push_to_hub("your_model_name")

# Step 15: Load the model and tokenizer for inference
model = BertForSequenceClassification.from_pretrained("./saved_model")
tokenizer = BertTokenizer.from_pretrained("./saved_model")

# Step 16: Example text for inference
text = "This is a new example for prediction."

# Step 17: Tokenize and prepare for inference
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
outputs = model(**inputs)

# Step 18: Get the predicted label
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
predicted_label = torch.argmax(predictions, dim=1).item()

print(f"Predicted label: {predicted_label}")
```



