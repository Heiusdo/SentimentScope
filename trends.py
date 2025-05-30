import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import os

# Set style
sns.set(style="whitegrid")

# Ensure pic folder exists
PIC_DIR = "src/pic"
os.makedirs(PIC_DIR, exist_ok=True)

# Load dataset
DATA_PATH = r'C:\Users\hieud\Documents\draft thesis\thesis\src\data\latest.csv'
df = pd.read_csv(DATA_PATH).dropna()
df = df[df['Sentiment'] != 1]  # Remove neutral class
df['Sentiment'] = df['Sentiment'].map({0: 'Negative', 2: 'Positive'})

# Plot 1: Sentiment Distribution
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='Sentiment', data=df, order=df['Sentiment'].value_counts().index, palette="viridis")
plt.title('Sentiment Distribution in Dataset')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.savefig(os.path.join(PIC_DIR, "sentiment_distribution.png"), dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Accuracy over Epochs (Old vs New Model)
epochs = list(range(1, 11))
old_model_acc = [0.7187, 0.7678, 0.7838, 0.7974, 0.8087, 0.8211, 0.8315, 0.8399, 0.8489, 0.8531]
new_model_acc = [0.7581, 0.7923, 0.8098, 0.8259, 0.8418, 0.8569, 0.8714, 0.8811, 0.8895, 0.8957]

plt.figure(figsize=(10, 6))
sns.lineplot(x=epochs, y=old_model_acc, label='Old Model (Facebook RoBERTa)', marker='o')
sns.lineplot(x=epochs, y=new_model_acc, label='New Model (CardiffNLP)', marker='s')
plt.title('Training Accuracy Across Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PIC_DIR, "accuracy_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Loss Curves
old_model_loss = [0.5376, 0.4817, 0.4566, 0.4343, 0.4129, 0.3922, 0.3734, 0.3570, 0.3404, 0.3325]
new_model_loss = [0.4926, 0.4444, 0.4131, 0.3843, 0.3555, 0.3276, 0.3001, 0.2799, 0.2625, 0.2495]

plt.figure(figsize=(10, 6))
sns.lineplot(x=epochs, y=old_model_loss, label='Old Model (Facebook RoBERTa)', marker='o')
sns.lineplot(x=epochs, y=new_model_loss, label='New Model (CardiffNLP)', marker='s')
plt.title('Training Loss Across Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PIC_DIR, "loss_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()