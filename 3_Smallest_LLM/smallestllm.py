import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Mini LLM Class ---
class MiniLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_tokens):
        x = self.token_embedding(input_tokens)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        logits = self.lm_head(x)
        return logits

# --- Create a tiny vocabulary ---
vocab = ["hello", "world", "i", "am", "chatgpt", "how", "are", "you", "today", "goodbye"]
word_to_id = {word: idx for idx, word in enumerate(vocab)}
id_to_word = {idx: word for word, idx in word_to_id.items()}

vocab_size = len(vocab)
embed_dim = 64

# --- Create model
model = MiniLLM(vocab_size, embed_dim).to(device)

# --- Function to encode and decode ---
def encode(words):
    return torch.tensor([[word_to_id[word] for word in words]], device=device)

def decode(token_ids):
    return [id_to_word[token.item()] for token in token_ids]

# --- Input some words ---
input_words = ["hello", "how", "are"]
input_tokens = encode(input_words)

# --- Forward pass
logits = model(input_tokens)

# --- Get predicted next words
# Take last token's prediction (you could also loop for multiple generations)
last_token_logits = logits[0, -1]  # (vocab_size,)
predicted_id = torch.argmax(last_token_logits)
predicted_word = id_to_word[predicted_id.item()]

print(f"Input words: {input_words}")
print(f"Predicted next word: {predicted_word}")
