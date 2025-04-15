# Imports and Setup
import pandas as pd
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"?? Using device: {device}")

# Load and Preprocess Data
DATA_PATH = 'data/latest.csv'
df = pd.read_csv(DATA_PATH).dropna()
df = df[df['Sentiment'] != 1]  # Remove neutral class
df = df.sample(n=200000, random_state=42)  # Downsize to 200k rows
df['Sentiment'] = df['Sentiment'].map({0: 0, 2: 1})  # Map labels: 0 (negative) -> 0, 2 (positive) -> 1

# Split dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['Text'].tolist(), df['Sentiment'].tolist(), test_size=0.2, random_state=42
)
print(f"?? Training samples: {len(train_texts)}, Testing samples: {len(test_texts)}")
print(f"Average text length (tokens): {df['Text'].str.split().apply(len).mean()}")
print("Label distribution in training set:")
print(pd.Series(train_labels).value_counts())

# Model and Tokenizer
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)
model.to(device)

# Layer Freezing Strategy
# Freeze the RoBERTa base model
for param in model.roberta.parameters():
    param.requires_grad = False

# Unfreeze the last 6 layers 
for i in range(-6, 0):
    for param in model.roberta.encoder.layer[i].parameters():
        param.requires_grad = True

# Prepare Data for Training
# Tokenize the data
train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
test_encodings = tokenizer(test_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")

# Convert to PyTorch tensors
train_dataset = TensorDataset(
    train_encodings['input_ids'],
    train_encodings['attention_mask'],
    torch.tensor(train_labels)
)
test_dataset = TensorDataset(
    test_encodings['input_ids'],
    test_encodings['attention_mask'],
    torch.tensor(test_labels)
)

# Optimizer and Scheduler Setup
# DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

# Define the optimizer with better parameters
optimizer = torch.optim.AdamW(
    [param for param in model.parameters() if param.requires_grad],
    lr=2e-5,
    weight_decay=0.01  # Add weight decay for regularization
)

# Add learning rate scheduler  
num_epochs = 10  # Fewer epochs but more effective training
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
    num_training_steps=total_steps
)

# Training
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters 
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()  # Adjust learning rate
        
        # Track loss
        total_loss += loss.item()
        
        # Track accuracy
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    # Compute metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

# Testing with Examples
example_texts = [
    "I love you",
    "This product is amazing",
    "I hate this experience",
    "The service was terrible"
]

model.eval()
for text in example_texts:
    # Tokenize
    encoding = tokenizer(text, padding=True, truncation=True, max_length=32, return_tensors="pt")
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
    
    # Print result
    label = "Positive" if prediction == 1 else "Negative"
    print(f"Text: '{text}'")
    print(f"Predicted: {label} (Confidence: {confidence:.2f})")
    print("-" * 40)

# Save Model(train in another server, path might conflict, but model is saved)
model_save_path = r"roberta_sentiment_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")