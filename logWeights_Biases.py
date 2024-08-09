import wandb

# Initialize Weights & Biases
wandb.init(project="your_project_name")

# Log model metrics
wandb.log({"train_loss": 0.719, "eval_loss": 0.814})
