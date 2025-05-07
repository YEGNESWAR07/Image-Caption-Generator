from flask import Flask, render_template, request
from model.encoder import EncoderCNN
from model.decoder import DecoderRNN
from model.vocab import Vocabulary
from model.utils import transform_image, generate_caption
import torch
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = EncoderCNN(embed_size=256).to(device)
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=5000).to(device)

encoder.load_state_dict(torch.load('saved_models/encoder.pth', map_location=device))
decoder.load_state_dict(torch.load('saved_models/decoder.pth', map_location=device))
encoder.eval()
decoder.eval()

# Dummy vocab — load your trained one
vocab = Vocabulary(freq_threshold=1)
vocab.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>", 4: "a", 5: "cat", 6: "on", 7: "mat"}
vocab.stoi = {v: k for k, v in vocab.itos.items()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        image_tensor = transform_image(open(filepath, 'rb'))
        caption = generate_caption(image_tensor, encoder, decoder, vocab, device=device)
        return render_template('index.html', caption=caption, img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
