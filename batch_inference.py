import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained("./saved_model")
tokenizer = BertTokenizer.from_pretrained("./saved_model")

# Load the test data
df = pd.read_json("test_data.json", lines=True)

# Function to predict the label
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return torch.argmax(predictions, dim=1).item()

# Apply the prediction to each text
df["predicted_label"] = df["text"].apply(predict)

# Save the results
df.to_json("inference_results.json", orient="records", lines=True)
print("Batch inference complete.")
