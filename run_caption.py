import torch
from model.encoder import EncoderCNN
from model.decoder import DecoderRNN
from model.vocab import Vocabulary
from model.utils import transform_image, generate_caption

# Load the models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = EncoderCNN(embed_size=256).to(device)
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=5000).to(device)

encoder.load_state_dict(torch.load("saved_models/encoder.pth.py", map_location=device))
decoder.load_state_dict(torch.load("saved_models/decoder.pth.py", map_location=device))

# Load vocabulary
vocab = Vocabulary(freq_threshold=5)
# Assume vocab is pre-built or load it from a file if saved

# Transform and generate caption
with open("path_to_image.jpg", "rb") as f:
    image_tensor = transform_image(f)
caption = generate_caption(image_tensor, encoder, decoder, vocab, device=device)
print("Generated Caption:", caption)