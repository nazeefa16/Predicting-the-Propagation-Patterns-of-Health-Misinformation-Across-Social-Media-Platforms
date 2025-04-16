import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
import matplotlib.pyplot as plt
import seaborn as sns

# ================= TF-IDF BASELINE =================
print("===> Loading dataset", flush=True)
df = pd.read_csv("merged_dataset.csv")

print("===> Running TF-IDF + Logistic Regression", flush=True)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['content'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
preds = model_lr.predict(X_test)
print("TF-IDF Model Accuracy:", round(accuracy_score(y_test, preds) * 100, 3), "%")

# ================= BERT FINE-TUNING =================
print("===> Preparing for BERT fine-tuning", flush=True)

label_map = {'Reliable': 0, 'Misinformation': 1}
df['label'] = df['label'].map(label_map).astype(int)
train_texts, val_texts, train_labels, val_labels = train_test_split(df['content'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

class PostDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            **{key: torch.tensor(val[idx]) for key, val in self.encodings.items()},
            'labels': torch.tensor(self.labels[idx])
        }

train_dataset = PostDataset(train_encodings, train_labels)
val_dataset = PostDataset(val_encodings, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 20
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training loop
print("===> Starting BERT training", flush=True)
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1} complete",flush=True)

# Evaluation
print("===> Evaluating model", flush=True)
model.eval()
preds, true = [], []
with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true.extend(batch['labels'].cpu().numpy())

print(classification_report(true, preds))

# Save model
save_directory = "./bert_misinformation_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# Confusion Matrix
cm = confusion_matrix(true, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Misinformation'], yticklabels=['Real', 'Misinformation'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")  # Save instead of show

# Classification report plot
report = classification_report(true, preds, target_names=["Real", "Misinformation"], output_dict=True)
report_df = pd.DataFrame(report).transpose()
metrics = report_df.loc[["Real", "Misinformation"], ["precision", "recall", "f1-score"]]

metrics.plot(kind='bar', figsize=(8, 6), colormap="Set2")
plt.title("Precision, Recall & F1-Score per Class")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.savefig("metrics_plot.png")  # Save instead of show
