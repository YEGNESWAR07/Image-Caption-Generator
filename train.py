import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from model.encoder import EncoderCNN
from model.decoder import DecoderRNN
from model.vocab import Vocabulary
from torch.utils.data import DataLoader
from data import ImageCaptionDataset

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),         # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Initialize the dataset and dataloader
dataset = ImageCaptionDataset(
    image_dir="data/Images",
    captions_file="data/captions.txt",
    vocab=Vocabulary(freq_threshold=5),
    transform=transform
)

# Check if the dataset is loaded correctly
# Check if the dataset is loaded correctly
if len(dataset) == 0:
    print("Dataset is empty. Debugging information:")
    print(f"Captions file path: data/captions.txt")
    print(f"Image directory path: data/Images")
    raise ValueError("Dataset is empty. Please check the captions file and image directory.")

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize models, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = EncoderCNN(embed_size=256).to(device)
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=5000).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

num_epochs = 10
# Training loop
for epoch in range(num_epochs):
    for images, captions in dataloader:  # Assume dataloader provides batches of images and captions
        images, captions = images.to(device), captions.to(device)

        # Forward pass
        features = encoder(images)
        outputs = decoder(features, captions)

        # Compute loss
        loss = criterion(outputs.view(-1, outputs.size(2)), captions.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print(f"Dataset size: {len(dataset)}")
# Save the trained models
torch.save(encoder.state_dict(), 'saved_models/encoder.pth')
torch.save(decoder.state_dict(), 'saved_models/decoder.pth')