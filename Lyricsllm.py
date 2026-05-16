"""
Portishead Lyrics Neural Network
=================================
- Custom word-level tokenizer & vocabulary (built from scratch)
- CNN for sentiment/mood classification (auto-labeled)
- LSTM encoder-decoder for lyric generation
- Works on CPU or GPU automatically
"""

import os
import re
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LYRICS_FILE   = "portishead_lyrics.txt"   # one song per block, blank line between songs
                                           # OR a csv with a 'lyrics' column
EMBED_DIM     = 128
HIDDEN_DIM    = 256
NUM_LAYERS    = 2
DROPOUT       = 0.3
SEQ_LEN       = 20          # words per training window (LSTM)
BATCH_SIZE    = 64
EPOCHS_LSTM   = 40
EPOCHS_CNN    = 20
LR            = 0.001
MIN_WORD_FREQ = 2            # ignore words appearing fewer times
GENERATE_LEN  = 60           # words to generate at inference
TEMPERATURE   = 0.8          # sampling temperature (higher = more creative)


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_lyrics(path: str) -> list[str]:
    """Load lyrics from .txt (blank-line separated) or .csv (lyrics column)."""
    if path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        col = next((c for c in df.columns if "lyric" in c.lower()), df.columns[0])
        songs = df[col].dropna().tolist()
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        songs = [s.strip() for s in raw.split("\n\n") if s.strip()]
    print(f"Loaded {len(songs)} songs")
    return songs


# ─────────────────────────────────────────────
# 2. TOKENIZER (built from scratch)
# ─────────────────────────────────────────────

class Tokenizer:
    """
    Word-level tokenizer with special tokens.
    Builds vocabulary from corpus, encodes/decodes sequences.
    """
    PAD, UNK, BOS, EOS = "<PAD>", "<UNK>", "<BOS>", "<EOS>"
    #PAD is for leftover padding in CNN input
    #UNK is for out of vocab words
    #BOS is for beginning of sequence
    #EOS is for end of sequence
    #so the tokenization starts from 4 and up for actual words

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.vocab_size = 0

    def _clean(self, text: str) -> list[str]:
        """Lowercase, strip punctuation (keep apostrophes), split."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    def build(self, corpus: list[str]):
        """Build vocabulary from list of song strings."""
        counts = Counter()
        for song in corpus:
            counts.update(self._clean(song))

        specials = [self.PAD, self.UNK, self.BOS, self.EOS]
        words = [w for w, c in counts.most_common() if c >= self.min_freq]

        vocab = specials + words
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(vocab)
        print(f"Vocabulary size: {self.vocab_size} words")
        return self

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        tokens = self._clean(text)
        ids = [self.word2idx.get(t, self.word2idx[self.UNK]) for t in tokens]
        if add_special:
            ids = [self.word2idx[self.BOS]] + ids + [self.word2idx[self.EOS]]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        specials = {self.PAD, self.UNK, self.BOS, self.EOS}
        words = []
        for i in ids:
            w = self.idx2word.get(i, self.UNK)
            if skip_special and w in specials:
                continue
            words.append(w)
        return " ".join(words)

    def save(self, path="tokenizer.json"):
        with open(path, "w") as f:
            json.dump({"word2idx": self.word2idx, "idx2word": {str(k): v for k, v in self.idx2word.items()}}, f)

    def load(self, path="tokenizer.json"):
        with open(path) as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        self.vocab_size = len(self.word2idx)
        return self


# ─────────────────────────────────────────────
# 3. AUTO-LABELING FOR SENTIMENT (CNN target)
# ─────────────────────────────────────────────

MOOD_SEEDS = {
    "dark":      ["darkness", "shadow", "cold", "dead", "hollow", "empty", "grave", "haunted", "black", "void"],
    "longing":   ["waiting", "gone", "miss", "far", "dream", "remember", "lost", "wish", "return", "distance"],
    "pain":      ["hurt", "pain", "cry", "tears", "break", "wound", "bleed", "sorrow", "ache", "suffer"],
    "hope":      ["light", "rise", "free", "new", "love", "hold", "stay", "together", "believe", "alive"],
}

def auto_label(songs: list[str]) -> list[int]:
    """
    Simple seed-word voting to auto-label mood for CNN training.
    Returns class indices: 0=dark, 1=longing, 2=pain, 3=hope
    """
    labels = []
    for song in songs:
        words = set(song.lower().split())
        scores = {mood: sum(1 for s in seeds if s in words)
                  for mood, seeds in MOOD_SEEDS.items()}
        best = max(scores, key=scores.get)
        labels.append(list(MOOD_SEEDS.keys()).index(best))
    dist = Counter(labels)
    print("Mood distribution:", {list(MOOD_SEEDS.keys())[k]: v for k, v in dist.items()})
    return labels

MOOD_NAMES = list(MOOD_SEEDS.keys())


# ─────────────────────────────────────────────
# 4. DATASETS
# ─────────────────────────────────────────────

class LSTMDataset(Dataset):
    """Sliding-window next-word prediction dataset."""
    def __init__(self, songs: list[str], tokenizer: Tokenizer, seq_len: int):
        self.seq_len = seq_len
        self.pairs: list[tuple[list[int], int]] = []
        for song in songs:
            ids = tokenizer.encode(song)
            for i in range(len(ids) - seq_len):
                self.pairs.append((ids[i:i+seq_len], ids[i+seq_len]))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        x, y = self.pairs[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class CNNDataset(Dataset):
    """Fixed-length sequence + mood label for CNN classifier."""
    MAX_LEN = 100

    def __init__(self, songs: list[str], labels: list[int], tokenizer: Tokenizer):
        self.samples = []
        pad_id = tokenizer.word2idx[Tokenizer.PAD]
        for song, label in zip(songs, labels):
            ids = tokenizer.encode(song)[:self.MAX_LEN]
            ids += [pad_id] * (self.MAX_LEN - len(ids))
            self.samples.append((ids, label))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ─────────────────────────────────────────────
# 5. CNN SENTIMENT MODEL (from scratch)
# ─────────────────────────────────────────────

class LyricsCNN(nn.Module):
    """
    TextCNN: parallel conv filters of sizes 2,3,4 → maxpool → classifier.
    Captures different n-gram patterns simultaneously.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int,
                 filter_sizes=(2, 3, 4), num_filters=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # One conv layer per filter size
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)                    # (batch, seq, embed)
        emb = emb.permute(0, 2, 1)                 # (batch, embed, seq) for Conv1d

        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))                  # (batch, filters, seq-fs+1)
            c = F.max_pool1d(c, c.size(2)).squeeze(2)  # (batch, filters)
            pooled.append(c)

        cat = torch.cat(pooled, dim=1)             # (batch, filters*len(filter_sizes))
        out = self.fc(self.dropout(cat))            # (batch, num_classes)
        return out


# ─────────────────────────────────────────────
# 6. LSTM GENERATION MODEL (from scratch)
# ─────────────────────────────────────────────

class LyricsLSTM(nn.Module):
    """
    Stacked LSTM for next-word language modelling.
    Embedding → LSTM × num_layers → Linear → vocab logits
    """
    def __init__(self, vocab_size: int, embed_dim: int,
                 hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb  = self.dropout(self.embedding(x))     # (batch, seq, embed)
        out, hidden = self.lstm(emb, hidden)        # (batch, seq, hidden)
        logits = self.fc(self.dropout(out))         # (batch, seq, vocab)
        return logits, hidden

    def init_hidden(self, batch_size: int):
        h = torch.zeros(NUM_LAYERS, batch_size, HIDDEN_DIM).to(DEVICE)
        c = torch.zeros(NUM_LAYERS, batch_size, HIDDEN_DIM).to(DEVICE)
        return (h, c)


# ─────────────────────────────────────────────
# 7. TRAINING LOOPS
# ─────────────────────────────────────────────

def train_cnn(model: LyricsCNN, loader: DataLoader, epochs: int):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    print("\n── Training CNN (Sentiment) ──")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            #next part is main loop for training the CNN model
            optimizer.zero_grad()#sets the gradients to 0 
            out  = model(x)#forward pass throught the model to get output
            loss = criterion(out, y)#loss is calculated using cross entropy
            loss.backward()#backpropagation to comput gradients
            optimizer.step()#update parameters using the optimizer
            total_loss += loss.item()#accumulate loss for reporting
            correct    += (out.argmax(1) == y).sum().item()#count correct predictions for accuracy picks mood with highest score
            total      += y.size(0)#count total samples for accuracy
        acc = correct / total * 100#calculate accuracy percentage
        print(f"  Epoch {epoch:02d} | Loss: {total_loss/len(loader):.4f} | Acc: {acc:.1f}%")#reports epoch loss and accuracy
    return model


def train_lstm(model: LyricsLSTM, loader: DataLoader, epochs: int):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    #necessary for training LSTM it halves the learning rate every 10 epochs to help convergence
    print("\n── Training LSTM (Generation) ──")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(x)
            # Use last timestep prediction vs target
            loss = criterion(logits[:, -1, :], y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)#sets a max norm of 1.0 for exploding gradient
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"  Epoch {epoch:02d} | Loss: {total_loss/len(loader):.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")
    return model


# ─────────────────────────────────────────────
# 8. INFERENCE
# ─────────────────────────────────────────────

def classify_mood(text: str, model: LyricsCNN, tokenizer: Tokenizer) -> str:
    """Run CNN sentiment classification on a text snippet."""
    model.eval()
    pad_id = tokenizer.word2idx[Tokenizer.PAD]
    ids = tokenizer.encode(text)[:CNNDataset.MAX_LEN]
    ids += [pad_id] * (CNNDataset.MAX_LEN - len(ids))
    x   = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    for i, p in enumerate(probs):
        print(f"  {MOOD_NAMES[i]:10s}: {p.item()*100:.1f}%")
    return MOOD_NAMES[probs.argmax().item()]


def generate_lyrics(seed: str, model: LyricsLSTM, tokenizer: Tokenizer,
                    length: int = GENERATE_LEN, temperature: float = TEMPERATURE) -> str:
    """
    Auto-regressive generation: encode seed → LSTM → sample next word → repeat.
    Temperature controls randomness: lower = safer, higher = more creative.
    """
    model.eval()
    ids = tokenizer.encode(seed)
    if not ids:
        ids = [tokenizer.word2idx[Tokenizer.BOS]]

    generated = list(ids)
    x = torch.tensor([ids], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        _, hidden = model(x)

        for _ in range(length):
            last = torch.tensor([[generated[-1]]], dtype=torch.long).to(DEVICE)
            logits, hidden = model(last, hidden)
            logits = logits[0, 0] / temperature
            probs  = F.softmax(logits, dim=0).cpu().numpy()
            next_id = np.random.choice(len(probs), p=probs)
            generated.append(int(next_id))

    text = tokenizer.decode(generated)

    # Format into lines of ~7 words (Portishead-style sparse lines)
    words = text.split()
    lines = [" ".join(words[i:i+7]) for i in range(0, len(words), 7)]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────

def main():
    # ── Load data ──
    songs  = load_lyrics("portishead_data.csv")
    labels = auto_label(songs)

    # ── Build tokenizer ──
    tokenizer = Tokenizer(min_freq=MIN_WORD_FREQ)
    tokenizer.build(songs)
    tokenizer.save("tokenizer.json")

    # ── CNN Training ──
    cnn_dataset = CNNDataset(songs, labels, tokenizer)
    cnn_loader  = DataLoader(cnn_dataset, batch_size=BATCH_SIZE, shuffle=True)

    cnn_model = LyricsCNN(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        num_classes=len(MOOD_NAMES)
    )
    cnn_model = train_cnn(cnn_model, cnn_loader, EPOCHS_CNN)
    torch.save(cnn_model.state_dict(), "cnn_sentiment.pt")
    print("CNN model saved → cnn_sentiment.pt")

    # ── LSTM Training ──
    lstm_dataset = LSTMDataset(songs, tokenizer, SEQ_LEN)
    lstm_loader  = DataLoader(lstm_dataset, batch_size=BATCH_SIZE, shuffle=True)

    lstm_model = LyricsLSTM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    lstm_model = train_lstm(lstm_model, lstm_loader, EPOCHS_LSTM)
    torch.save(lstm_model.state_dict(), "lstm_generation.pt")
    print("LSTM model saved → lstm_generation.pt")

    # ── Demo: Mood Classification ──
    print("\n── Mood Classification Demo ──")
    test_text = "darkness surrounds me cold and hollow waiting in the shadow"
    print(f"Input: \"{test_text}\"")
    mood = classify_mood(test_text, cnn_model, tokenizer)
    print(f"Predicted mood: {mood}")

    # ── Demo: Lyric Generation ──
    print("\n── Lyric Generation Demo ──")
    seed = "the darkness"
    print(f"Seed: \"{seed}\"\n")
    lyrics = generate_lyrics(seed, lstm_model, tokenizer)
    print(lyrics)

    # ── Interactive loop ──
    print("\n── Interactive Mode (type 'quit' to exit) ──")
    while True:
        mode = input("\n[g]enerate or [c]lassify? ").strip().lower()
        if mode == 'quit':
            break
        elif mode == 'g':
            seed = input("Seed phrase: ").strip()
            temp = input(f"Temperature (default {TEMPERATURE}): ").strip()
            temp = float(temp) if temp else TEMPERATURE
            print("\n" + generate_lyrics(seed, lstm_model, tokenizer, temperature=temp))
        elif mode == 'c':
            text = input("Paste lyrics snippet: ").strip()
            classify_mood(text, cnn_model, tokenizer)


if __name__ == "__main__":
    main()