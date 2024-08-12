<img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.png" alt="Huggingface Logo" width="300">


# Converting the MongoDB data into a Hugging Face Dataset

Extract Data from MongoDB:

Use a MongoDB client (like pymongo) to query and retrieve data from your MongoDB collections.
Convert the retrieved data into a format suitable for training, like a pandas DataFrame.
Preprocess and Train:

Preprocess the data (e.g., tokenization, label encoding) using Hugging Face transformers and datasets.
Train the model.

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
Then run hugfaceAuto.py training script - make sure you are in the same folder as your simple_test_data.json or pass the path to it

```
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from huggingface_hub import login
import torch
```
# Step 1: Authenticate with Hugging Face using your write token
```
login(token="put your WRITE token here")  # Replace with your actual Hugging Face write token
```
# Step 2: Load the JSON file into a DataFrame
```
df = pd.read_json('simple_test_data.json')
```
# Step 3: Convert labels to numerical format
```
label_map = {'positive': 1, 'negative': 0}
df['label'] = df['label'].map(label_map)
```
# Step 4: Split the data into train and test sets
```
train_df, test_df = train_test_split(df, test_size=0.2)
```
# Step 5: Convert DataFrame to Hugging Face Dataset
```
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
```
# Step 6: Load the tokenizer and model
```
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
```
# Step 7: Tokenize the input
```
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)
```
# Step 8: Set the format for PyTorch tensors
```
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
```
# Step 9: Define training arguments
```
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)
```
# Step 10: Define the trainer
```
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
```
# Step 11: Train the model
```
trainer.train()
```
# Step 12: Evaluate the model
```
trainer.evaluate()
```
# Step 13: Save the model and tokenizer locally
```
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
```

# Step 14: Push the model and tokenizer to Hugging Face Hub
```
model.push_to_hub("your_model_name")  # Replace with your desired model name on Hugging Face
tokenizer.push_to_hub("your_model_name")
```
# Step 15: Load the model and tokenizer for inference
```
model = BertForSequenceClassification.from_pretrained("./saved_model")
tokenizer = BertTokenizer.from_pretrained("./saved_model")
```
# Step 16: Example text for inference
```
text = "This is a new example for prediction."
```
# Step 17: Tokenize and prepare for inference
```
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
outputs = model(**inputs)
```
# Step 18: Get the predicted label
```
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
predicted_label = torch.argmax(predictions, dim=1).item()
print(f"Predicted label: {predicted_label}")
```
# Output should look like this Predicted label score should be 1=positive (or) 0=negative

```
➜  huggingface python3 hugfaceAuto.py
Token is valid (permission: write).
Your token has been saved to /Users/jeffery.schmitz/.cache/huggingface/token
Login successful
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 567.13 examples/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 460.00 examples/s]
{'eval_loss': 0.6930631399154663, 'eval_runtime': 0.1085, 'eval_samples_per_second': 9.217, 'eval_steps_per_second': 9.217, 'epoch': 1.0}              
{'eval_loss': 0.7359107732772827, 'eval_runtime': 0.0149, 'eval_samples_per_second': 67.151, 'eval_steps_per_second': 67.151, 'epoch': 2.0}            
{'eval_loss': 0.7567771673202515, 'eval_runtime': 0.0144, 'eval_samples_per_second': 69.668, 'eval_steps_per_second': 69.668, 'epoch': 3.0}            
{'train_runtime': 1.4217, 'train_samples_per_second': 6.33, 'train_steps_per_second': 2.11, 'train_loss': 0.735368569691976, 'epoch': 3.0}             
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.12it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 624.62it/s]
model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 438M/438M [05:20<00:00, 1.37MB/s]
Predicted label: 0
```




