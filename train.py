import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
from models.Descriptor import Descriptor, OneDeepDescriptor, TwoDeepDescriptor
from models.CLAPModel import CLAPModel
from utils.dataset import MusicDataset, EmbeddingDataset
from utils.loss import MSEMusicDescriptorLoss, AdaptedMusicDescriptorLoss
import wandb
from datetime import datetime

def train(model, device, train_loader, val_loader, num_epochs, optimizer, criterion):
    wandb.init(project="audio-conditioner", name=f"audio-conditioner-run-{datetime.now().strftime('%Y%m%d_%H%M%S')}", config={
        "num_epochs": num_epochs,
        "optimizer": optimizer.__class__.__name__,
        "criterion": criterion.__class__.__name__,
    })
    wandb.watch(model, criterion, log="all",log_freq=10)
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            wandb.log({"batch_loss": loss.item()})
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        val_loss_total = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                val_loss_total += val_loss.item()
        
        avg_val_loss = val_loss_total / len(val_loader)
        wandb.log({"avg_loss": avg_loss, "val_loss": avg_val_loss, "epoch": epoch+1})

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        torch.save(model.state_dict(), "saves/model_checkpoint.pt")
        print("Model saved successfully!")
    
    wandb.finish()

def main(lr=0.001, num_epochs=10, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = MusicDataset('data/teacher_dataset.jsonl')
    embedding_model = CLAPModel()
    embedding_dataset = EmbeddingDataset(dataset, embedding_model, device=device)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(embedding_dataset))
    val_size = len(embedding_dataset) - train_size
    train_dataset, val_dataset = random_split(embedding_dataset, [train_size, val_size])

    print(f"Total samples: {len(embedding_dataset)}, Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, criterion and optimizer
    model = TwoDeepDescriptor(clap_dim=512, backbone_dim=256).to(device)
    criterion = AdaptedMusicDescriptorLoss() #Weights can be added
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    train(model, device, train_loader, val_loader, num_epochs=num_epochs, optimizer=optimizer, criterion=criterion)

if __name__ == "__main__":
    main(lr=0.0005, num_epochs=50, batch_size=64)