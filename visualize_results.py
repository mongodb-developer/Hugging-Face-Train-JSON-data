import matplotlib.pyplot as plt

# Load the inference results
df = pd.read_json("inference_results.json", lines=True)

# Plot the distribution of true vs predicted labels
plt.figure(figsize=(10, 5))
df.groupby("label").size().plot(kind='bar', color='blue', alpha=0.7, label='True Labels')
df.groupby("predicted_label").size().plot(kind='bar', color='orange', alpha=0.5, label='Predicted Labels')
plt.legend()
plt.title("True vs Predicted Label Distribution")
plt.show()
