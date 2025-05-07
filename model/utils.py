import torch
from torchvision import transforms
from PIL import Image

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_bytes).convert("RGB")
    return transform(image).unsqueeze(0)

def generate_caption(image_tensor, encoder, decoder, vocab, max_len=20, device="cpu"):
    result = []
    with torch.no_grad():
        features = encoder(image_tensor.to(device))
        input = torch.tensor([vocab.stoi["<SOS>"]]).to(device).unsqueeze(0)
        states = None

        for _ in range(max_len):
            embedding = decoder.embed(input)
            hiddens, states = decoder.lstm(embedding, states)
            output = decoder.linear(hiddens.squeeze(1))
            predicted = output.argmax(1)
            word = vocab.itos[predicted.item()]

            if word == "<EOS>":
                break
            result.append(word)
            input = predicted.unsqueeze(0)

    return ' '.join(result)
