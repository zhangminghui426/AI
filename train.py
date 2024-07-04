import torch
import torch.nn as nn
from torch import optim
import timeit
from tqdm import tqdm
from dataset.utils import get_image_loaders
from model.VisionTransormer import Vit


class VisionModelConfig:
    def __init__(self, cfg=None):
        # Hyper Parameters
        self.epochs = 50
        self.batch_size = 16
        self.train_dir = "./data/train.csv"
        self.test_dir = "./data/test.csv"

        # Model Parameters
        self.in_channels = 1
        self.image_size = 28
        self.patch_size = 4
        self.embed_dim = (self.patch_size ** 2) * self.in_channels
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.dropout = 0.001

        self.num_heads = 4
        self.activation = "gelu"
        self.num_layers = 16
        self.num_classes = 10

        self.learning_rate = 1e-4
        self.adam_weight_decay = 0
        self.adam_betas = (0.9, 0.999)


def train(epochs):
    start = timeit.default_timer()
    for epoch in tqdm(range(epochs), position=0, leave=True):
        model.train()
        train_labels = []
        train_preds = []
        train_running_loss = 0
        for idx, img_label in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            img = img_label["image"].float().to(device)
            label = img_label["label"].type(torch.uint8).to(device)
            y_pred = model(img)
            y_pred_label = torch.argmax(y_pred, dim=1)

            train_labels.extend(label.cpu().detach())
            train_preds.extend(y_pred_label.cpu().detach())

            loss = criterion(y_pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
        train_loss = train_running_loss / (idx + 1)

        model.eval()
        val_labels = []
        val_preds = []
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_label in enumerate(tqdm(val_dataloader, position=0, leave=True)):
                img = img_label["image"].float().to(device)
                label = img_label["label"].type(torch.uint8).to(device)
                y_pred = model(img)
                y_pred_label = torch.argmax(y_pred, dim=1)

                val_labels.extend(label.cpu().detach())
                val_preds.extend(y_pred_label.cpu().detach())

                loss = criterion(y_pred, label)
                val_running_loss += loss.item()
        val_loss = val_running_loss / (idx + 1)

        print("-" * 30)
        print(f"Train Loss Epoch {epoch + 1} : {train_loss:.4f}")
        print(f"Val   Loss Epoch {epoch + 1} : {val_loss:.4f}")
        print(f"Train Accuracy EPOCH {epoch + 1}: {sum(1 for x, y in zip(train_preds, train_labels) if x == y) / len(train_labels):.4f}")
        print(f"Val Accuracy EPOCH {epoch + 1}: {sum(1 for x, y in zip(val_preds, val_labels) if x == y) / len(val_labels):.4f}")
        print("-" * 30)

    stop = timeit.default_timer()
    print(f"Training Time:{stop - start:.2f}s")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = VisionModelConfig()

    train_dataloader, val_dataloader, test_dataloader = get_image_loaders(cfg.train_dir, cfg.test_dir, cfg.batch_size)

    model = Vit(in_channels=cfg.in_channels, 
                patch_size=cfg.patch_size, 
                embed_dim=cfg.embed_dim, 
                num_patches=cfg.num_patches,
                dropout=cfg.dropout,
                num_heads=cfg.num_heads,
                activation=cfg.activation,
                num_layers=cfg.num_layers,
                num_classes=cfg.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), betas=cfg.adam_betas, lr=cfg.learning_rate, weight_decay=cfg.adam_weight_decay)
    train(epochs=cfg.epochs)


