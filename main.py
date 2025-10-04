import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.pneumonia_cnn import PneumoniaCNN
from utils.dataset_loader import PneumoniaDataset
from utils.train_eval import train_one_epoch, evaluate
from utils.metrics import compute_metrics
from utils.visualize import plot_confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 15
LR = 0.001

# Datasets & Loaders
train_set = PneumoniaDataset(TRAIN_DIR)
val_set = PneumoniaDataset(VAL_DIR)
test_set = PneumoniaDataset(TEST_DIR)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

# Model
model = PneumoniaCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop
best_acc = 0
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, DEVICE)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "models/best_pneumonia_model.pth")
        print("✅ Saved new best model")

# Final Evaluation
model.load_state_dict(torch.load("models/best_pneumonia_model.pth"))
_, _, y_true, y_pred = evaluate(model, test_loader, criterion, DEVICE)
metrics = compute_metrics(y_true, y_pred)
print(metrics)
plot_confusion_matrix(y_true, y_pred)
