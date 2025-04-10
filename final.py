import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import time
from PIL import Image, ImageFile

# Set this to True to handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom dataset class to handle corrupted images
class CatsDogsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()
        
    def _make_dataset(self):
        samples = []
        for target_class in self.classes:
            class_index = self.class_to_idx[target_class]
            class_dir = os.path.join(self.root_dir, target_class)
            
            if not os.path.isdir(class_dir):
                continue
                
            for fname in os.listdir(class_dir):
                if fname.startswith('.'):
                    continue
                    
                path = os.path.join(class_dir, fname)
                if not os.path.isfile(path):
                    continue
                    
                # Check if the file is a valid image
                try:
                    with open(path, 'rb') as f:
                        img = Image.open(f)
                        img.verify()  # Verify it's an image
                    samples.append((path, class_index))
                except (IOError, SyntaxError, OSError) as e:
                    print(f"Skipping corrupted image {path}: {e}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        
        # Load image safely
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')  # Ensure consistent format
                
            if self.transform is not None:
                img = self.transform(img)
                
            return img, target
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a placeholder image if loading fails
            dummy_img = torch.zeros(3, 224, 224) if self.transform else Image.new('RGB', (224, 224), (0, 0, 0))
            return dummy_img, target

# Function to load and preprocess the dataset
def load_and_preprocess_data(data_dir, img_size=224, batch_size=32):
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset using custom class
    dataset = CatsDogsDataset(root_dir=data_dir, transform=transform)
    
    # Split dataset into train, validation, and test sets (70%, 15%, 15%)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    class_to_idx = dataset.class_to_idx
    
    return train_loader, val_loader, test_loader, class_to_idx

# Function to set up a pre-trained ViT model
def setup_pretrained_model(model_name="google/vit-base-patch16-224-in21k", num_classes=2, fine_tune=True):
    # Load pre-trained ViT model
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    # Freeze some layers if fine_tune is False
    if not fine_tune:
        for param in model.vit.embeddings.parameters():
            param.requires_grad = False
    
    model = model.to(device)
    return model

# Function to train the model
def train_model(model, train_loader, val_loader, num_epochs=5, learning_rate=0.0002):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Train Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Function to evaluate the model
def evaluate_model(model, data_loader, criterion=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate statistics
    loss = running_loss / len(data_loader.dataset)
    accuracy = correct / total
    
    return loss, accuracy

# Function to test the model and generate confusion matrix - FIXED FUNCTION
def test_model(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs).logits
            _, predicted = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print classification report
    # Fixed: Get the actual class names as a list, not treating them as int keys
    class_names_list = [class_names[i] for i in range(len(class_names))]
    report = classification_report(all_labels, all_preds, target_names=class_names_list)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names_list, 
                yticklabels=class_names_list)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save the figure and make sure it doesn't try to display if there's no display
    plt.savefig('confusion_matrix.png')
    
    # Only call plt.show() if you have a display
    try:
        plt.show()
    except:
        plt.close()
    
    return accuracy, cm, report

# Function to plot training results
def plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the figure and make sure it doesn't try to display if there's no display
    plt.savefig('training_results.png')
    
    # Only call plt.show() if you have a display
    try:
        plt.show()
    except:
        plt.close()

# Main function
def main():
    # Define parameters
    data_dir = "PetImages"  # Folder containing Cat and Dog subfolders
    batch_size = 16
    num_epochs = 5
    learning_rate = 0.0002
    img_size = 224  # ViT typically uses 224x224 images
    
    # 1. Load and preprocess the dataset
    print("Loading and preprocessing dataset...")
    train_loader, val_loader, test_loader, class_to_idx = load_and_preprocess_data(
        data_dir, img_size, batch_size
    )
    class_names = {v: k for k, v in class_to_idx.items()}
    print(f"Classes: {class_names}")
    
    # 2. Set up the ViT model
    print("Setting up ViT model...")
    model = setup_pretrained_model(
        model_name="google/vit-base-patch16-224-in21k",  # Pre-trained ViT model
        num_classes=len(class_names),
        fine_tune=True
    )
    
    # 3. Train and evaluate the ViT model
    print("Training model...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs, learning_rate
    )
    
    # Plot training results
    plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Test the model
    print("Testing model...")
    test_accuracy, confusion_mat, classification_rep = test_model(model, test_loader, class_names)
    
    # Save the model
    print("Saving model...")
    torch.save(model.state_dict(), "vit_cats_dogs_model.pth")
    print("Model saved as 'vit_cats_dogs_model.pth'")

if __name__ == "__main__":
    main()
