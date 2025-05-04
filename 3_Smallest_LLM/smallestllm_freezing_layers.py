import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Step 1: Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Step 2: Define Mini LLM Class ---
class MiniLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # Initialize the layers for the model:
        # Embedding layer for tokenizing inputs (Step 3)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Multihead attention layer (Step 4)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        
        # Feed-Forward network (Step 5)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # First linear layer (Step 6)
            nn.GELU(),  # Activation function (Step 7)
            nn.Linear(embed_dim * 4, embed_dim)  # Second linear layer (Step 8)
        )
        
        # Layer Normalization (Step 9)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Final Linear layer to output the vocabulary size (Step 10)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    # --- Step 3: Forward Pass ---
    def forward(self, input_tokens):
        # Token embeddings (Step 11)
        x = self.token_embedding(input_tokens)
        
        # Attention mechanism (Step 12)
        attn_out, _ = self.attention(x, x, x)
        
        # First Layer Normalization (Step 13)
        x = self.norm1(x + attn_out)
        
        # Feed-Forward Network (Step 14)
        ffn_out = self.ffn(x)
        
        # Second Layer Normalization (Step 15)
        x = self.norm2(x + ffn_out)
        
        # Final output logits (Step 16)
        logits = self.lm_head(x)
        return logits


# --- Step 4: Create a Tiny Vocabulary ---
vocab = ["hello", "world", "i", "am", "chatgpt", "how", "are", "you", "today", "goodbye"]
word_to_id = {word: idx for idx, word in enumerate(vocab)}
id_to_word = {idx: word for word, idx in word_to_id.items()}

# Define the vocab size and embedding dimension
vocab_size = len(vocab)
embed_dim = 64

# --- Step 5: Instantiate the Model ---
model = MiniLLM(vocab_size, embed_dim).to(device)

# --- Step 6: Freeze Most Layers for PEFT ---
def freeze_layers(model):
    """
    Freezes all layers except the lm_head to allow PEFT.
    """
    for name, param in model.named_parameters():
        if name != 'lm_head.weight' and name != 'lm_head.bias':
            param.requires_grad = False
        else:
            param.requires_grad = True

# Freeze model except the last linear layer (lm_head)
freeze_layers(model)

# --- Step 7: Function for Encoding and Decoding ---
# Function to convert a list of words into token ids (Step 20)
def encode(words):
    return torch.tensor([[word_to_id[word] for word in words]], device=device)

# Function to convert token ids back into words (Step 21)
def decode(token_ids):
    return [id_to_word[token.item()] for token in token_ids]

# --- Step 8: Input some words ---
input_words = ["hello", "how", "are"]
input_tokens = encode(input_words)

# --- Step 9: Forward Pass ---
logits = model(input_tokens)

# --- Step 10: Get Predicted Next Word ---
last_token_logits = logits[0, -1]  # (vocab_size,)
predicted_id = torch.argmax(last_token_logits)  # Get the index of the most likely word
predicted_word = id_to_word[predicted_id.item()]  # Convert the index back to a word

# --- Step 11: Display the Results ---
print(f"Input words: {input_words}")
print(f"Predicted next word: {predicted_word}")

