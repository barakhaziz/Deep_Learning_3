from tqdm import tqdm
import os
import io
import re
import random
import string
import zipfile
import requests
import numpy as np
import pandas as pd
import pretty_midi
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import List
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec, KeyedVectors
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_song(lyrics: List[str]):
    """
    Helper function that takes a list of strings including special tokens and returns the song as a single string
    """
    song = " ".join(lyrics).replace("eol ", "\n").replace("eos", "").strip()
    return song

class CosineSimilarityLoss(nn.Module):
    """
    Cosine Similarity value range is [-1, 1] and Loss value range is [0, 1]
    Similarity = 1 => loss = 0 and Similarity = -1 => loss = 1
    """
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.input_type = "embedding"
        self.similarity = nn.CosineSimilarity(dim=2)

    def forward(self, output, target, not_pad_mask):
        res = self.similarity(output, target)
        masked_res = res * not_pad_mask
        loss = (1 - masked_res.sum() / not_pad_mask.sum()) / 2
        return loss

class SentimentLoss(nn.Module):
    """
    Calculate loss based on a pre-trained Sentiment analysis model
    sentiment model polarity scores has compound values [-1, 1],
    and negative, positive, neutral values [0, 1].
    Sentiment loss per sample in batch are calculated as follows:
        sum(MSE(score) per score in [compound, negative, positive, neutral]) / max_error (= 7)
    This gives sample loss in range [0, 1]
    return average of losses in batch
    """
    def __init__(self, sentiment_model=SentimentIntensityAnalyzer()):
        super(SentimentLoss, self).__init__()
        self.input_type = "words"
        self.sid = sentiment_model

    def _get_scores(self, sequences):
        batch_scores = [
            list(self.sid.polarity_scores(
                get_song(sequences[seq_ind])
            ).values()) for seq_ind in range(sequences.shape[0])
        ]
        return np.array(batch_scores)

    def forward(self, output, target):
        # get score per sample in batch and calculate the average score
        output_scores = self._get_scores(output)
        target_scores = self._get_scores(target)

        # weighted_sentiment_loss range is [0, 2**2 + 3 = 7], divide by 7 for [0, 1] range
        sentiment_losses = np.mean(np.sum((output_scores - target_scores) ** 2, axis=1) / 7)
        return sentiment_losses

class LineLenLoss(nn.Module):
    """
    LineLenLoss: MSE between average line len per song in batch compared to target average line len - loss range [0, 1]
    """
    def __init__(self, end_of_line_token="eol", max_seq_len=600):
        super(LineLenLoss, self).__init__()
        self.input_type = "words"
        self.end_of_line_token = end_of_line_token
        self.max_seq_len = max_seq_len

    def _get_avg_line_lens(self, sequences):
        eol_mask = (sequences == self.end_of_line_token)
        eol_indices = [
            np.where(seq_eol_mask)[0] if np.any(seq_eol_mask) else np.array([self.max_seq_len])
            for seq_eol_mask in eol_mask
        ]
        avg_line_len = [
            np.mean(np.insert(np.diff(eol_index) - 1, 0, eol_index[0]))
            for eol_index in eol_indices
        ]
        return np.array(avg_line_len)

    def forward(self, output, target):
        # line length loss -
        output_avg_line_lens = self._get_avg_line_lens(output)
        target_avg_line_lens = self._get_avg_line_lens(target)

        line_len_loss = np.mean(
            abs(output_avg_line_lens - target_avg_line_lens) / self.max_seq_len
        )

        return line_len_loss

class SongLenLoss(nn.Module):
    """
    SongLenLoss: MSE error between song len and target song len - loss range [0, 1]
    """
    def __init__(self, end_of_song_token="eos", max_seq_len=600):
        super(SongLenLoss, self).__init__()
        self.input_type = "words"
        self.end_of_song_token = end_of_song_token
        self.max_seq_len = max_seq_len

    def _get_song_len(self, sequences):
        eos_mask = (sequences == self.end_of_song_token)
        return np.where(np.any(eos_mask, axis=1), np.argmax(eos_mask, axis=1), self.max_seq_len)

    def forward(self, output, target):
        # song length loss -
        output_song_len = self._get_song_len(output)
        target_song_len = self._get_song_len(target)

        max_len_error = np.maximum(target_song_len, self.max_seq_len - target_song_len) ** 2
        song_len_loss = np.mean((output_song_len - target_song_len) ** 2 / max_len_error)

        return song_len_loss

class DiversityLoss(nn.Module):
    """
    Penalize songs which use a small number of unique words to counter repeated word usage
    """
    def __init__(self):
        super(DiversityLoss, self).__init__()
        self.input_type = "words"

    def forward(self, output, target):
        unique_word_counts = [len(set(sequence)) for sequence in output]
        total_word_counts = [len(sequence) for sequence in output]

        # Calculate diversity as the ratio of unique words to total words
        diversity_ratios = np.array(unique_word_counts) / np.array(total_word_counts)
        diversity_loss = 1 - np.mean(diversity_ratios)

        return diversity_loss

class EntropyRegularizationLoss(nn.Module):
    """
    Penalize low entropy in song generation to promote less deterministic word selection
    """
    def __init__(self, vocab_size):
        super(EntropyRegularizationLoss, self).__init__()
        self.input_type = "logits"
        self.max_entropy = np.log(vocab_size)

    def forward(self, logits, target):
        # Compute softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
        # Normalize entropy by the maximum entropy
        normalized_entropy = entropy / self.max_entropy
        entropy_loss = 1 - normalized_entropy

        return entropy_loss

class CombinedLyricsLoss(nn.Module):
    def __init__(
        self,
        word2vec,
        loss_func_list: List[nn.Module],
        loss_weights: List[float]
    ):
        super(CombinedLyricsLoss, self).__init__()
        self.word2vec = word2vec
        self.loss_func_list = loss_func_list
        self.loss_weights = loss_weights
        self.summed_weights = sum(loss_weights)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.pad_idx = self.word2vec.key_to_index["pad"]

    def forward(self, logits, target, embedding_layer):
        device = logits.device
        batch_size, seq_len, vocab_size = logits.size()

        # Calculate CrossEntropyLoss
        ce_loss = self.cross_entropy(logits.view(-1, vocab_size), target.view(-1))

        # Get predicted words
        predicted_indices = torch.argmax(logits, dim=-1)
        predicted_words = np.array([
            [self.word2vec.index_to_key[idx.item()]
             for idx in sequence] for sequence in predicted_indices
        ])

        # Get target words
        target_words = np.array([
            [self.word2vec.index_to_key[idx.item()] for idx in sequence]
            for sequence in target
        ])

        # Get embeddings for predicted and target words
        predicted_embeddings = embedding_layer(predicted_indices)
        target_embeddings = embedding_layer(target)
        not_pad_mask = (target != self.pad_idx)

        weighted_loss = ce_loss.clone()

        for ind, loss_func in enumerate(self.loss_func_list):
            if self.loss_weights[ind] == 0:
                continue
            if loss_func.input_type == "words":
                loss = torch.tensor(loss_func(predicted_words, target_words), device=device)
            elif loss_func.input_type == "embedding":
                loss = loss_func(predicted_embeddings, target_embeddings, not_pad_mask)
            elif loss_func.input_type == "logits":
                loss = loss_func(logits, target)
            else:
                raise AssertionError("Loss function is missing input_type attribute")

            weighted_loss += self.loss_weights[ind] * loss

        return weighted_loss / (self.summed_weights + 1)  # +1 for CrossEntropyLoss



class LyricsGAN(nn.Module):
    def __init__(
            self,
            word2vec,
            hidden_dim: int = 256,
            melody_hidden_dim: int = 128,
            rnn_type: str = "gru",
            n_rnn_layers: int = 3,
            dropout: float = 0.6,
            use_attention: bool = False,
            criterion=nn.CrossEntropyLoss(),
    ):
        """
        Generative Lyrics model including train and song generation methods
        :param word2vec: an initial word2vec model to use for the embedding layer
        :param melody_hidden_dim: hidden size of the melody layer
        :param hidden_dim: number of hidden nodes in the RNN architecture
        :param rnn_type: type of RNN cell ("lstm" or "gru")
        :param n_rnn_layers: Number of RNN layers in the RNN block
        :param dropout: dropout probability in the RNN cell
        :param use_attention: whether to use attention mechanism
        :param criterion: Loss function to use in training
        """
        super(LyricsGAN, self).__init__()

        self.word2vec_model = word2vec
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion.to(self.device)
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.melody_hidden_dim = melody_hidden_dim
        self.writer = SummaryWriter()
        self.vector_size = word2vec.vector_size
        self.vocab_size = len(word2vec)

        embedding_matrix = np.zeros((self.vocab_size, self.vector_size))
        for word, idx in word2vec.key_to_index.items():
            embedding_matrix[idx] = word2vec[word]

        embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Sequential(nn.Embedding.from_pretrained(embedding_matrix, freeze=True),
                                       nn.ReLU(inplace=True))

        # The "encoding" layer of the melody input
        self.melody_encoder = nn.Sequential(
            nn.Linear(12, melody_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        if self.use_attention:
            # attention mechanisms for melody input
            self.melody_attention = nn.MultiheadAttention(melody_hidden_dim, num_heads=1, batch_first=True)
            # self-attention for sequence
            self.word_attention = nn.MultiheadAttention(self.vector_size, num_heads=1, batch_first=True)

        if rnn_type == "lstm":
            self.rnn_block = nn.LSTM(
                input_size=self.vector_size + melody_hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_rnn_layers,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.rnn_block = nn.GRU(
                input_size=self.vector_size + melody_hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_rnn_layers,
                dropout=dropout,
                batch_first=True,
            )

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dense = nn.Linear(hidden_dim, self.vocab_size)

        self.to(self.device)

    def forward(self, x, melody_state, prev_state=None):
        b_size, seq_len = x.size()
        x_embbed = self.embedding(x)
        melody_state = self.melody_encoder(melody_state)

        if self.use_attention:
            if prev_state is None:
                query = torch.zeros(b_size, 1, self.hidden_dim).to(self.device)
            else:
                query = prev_state[0][-1].view(b_size, 1, self.melody_hidden_dim)

            melody_context, melody_attention = self.melody_attention(query, melody_state, melody_state)
            melody_context = melody_context.repeat(1, seq_len, 1)

            # mask future words so model won't rely on them when generating
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(self.device)
            word_context, word_attention = self.word_attention(x_embbed, x_embbed, x_embbed, attn_mask=causal_mask)

            combined_input = torch.cat((word_context, melody_context), dim=2)
        else:
            combined_input = torch.cat((x_embbed, melody_state), dim=2)

        output, state = self.rnn_block(combined_input, prev_state)

        output = self.bn(output.transpose(1, 2)).transpose(1, 2)  # apply batchnorm to the feature space
        logits = self.dense(output)
        return logits, state

    def train_model(
            self,
            train_loader,
            validation_loader,
            epochs: int,
            patience: int = 10,
            lr: float = 0.001
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        self.train()
        losses = {"Train": [], "Validation": []}
        best_val_loss = float('inf')
        patience_counter = 0
        steps = 1
        with tqdm(total=epochs, desc='Training') as pbar:
            for epoch in range(epochs):
                total_loss = torch.tensor(0.0, device=self.device)

                # Iterate over the train_loader (batches)
                for inputs, targets, midi_vectors in train_loader:
                    steps += 1
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    midi_vectors = midi_vectors.to(self.device)

                    optimizer.zero_grad()
                    logits, _ = self(inputs, midi_vectors)

                    loss = self.criterion(logits, targets, self.embedding)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss

                avg_train_loss = total_loss.item() / len(train_loader)
                # Log training loss to TensorBoard
                self.writer.add_scalar('Loss/train_step', loss.item(), steps)

                # Validation loop
                self.eval()
                total_val_loss = torch.tensor(0.0, device=self.device)
                with torch.no_grad():
                    for inputs, targets, midi_vectors in validation_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        midi_vectors = midi_vectors.to(self.device)

                        logits, _ = self(inputs, midi_vectors)

                        val_loss = self.criterion(logits, targets, self.embedding)
                        total_val_loss += val_loss

                avg_val_loss = total_val_loss.item() / len(validation_loader)

                # Log epoch losses to TensorBoard
                self.writer.add_scalars('Loss/epoch', {
                    'train': avg_train_loss,
                    'validation': avg_val_loss
                }, epoch)
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    # Save the best model
                    torch.save(self.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1

                pbar.set_description(
                    f'Epoch {epoch + 1}/{epochs}, '
                    f'Train Loss: {round(avg_train_loss, 4)}, '
                    f'Val Loss: {round(avg_val_loss, 4)}, '
                    f'BestVal Loss: {round(best_val_loss, 4)} (epoch {best_epoch})'
                )
                pbar.update(1)

                if patience_counter >= patience:
                    print(
                        f'\nEarly stopping triggered after {epoch + 1} '
                        f'epochs with best val loss = {round(best_val_loss, 4)}'
                    )
                    break

                if epoch % 25 == 0:
                    save_path = f'model_epoch_{epoch}.pth'
                    torch.save(self.state_dict(), save_path)

                    for i in range(3):
                        # Choose a random initial word and MIDI vector from the validation dataset to start a song
                        random_index = random.randint(0, len(validation_loader.dataset) - 1)
                        song_seed, _, melody = validation_loader.dataset[random_index]
                        song_seed = validation_loader.dataset.model.index_to_key[song_seed[0].item()]
                        generated_song = self.generate_song(song_seed, melody)
                        print(f"\nEpoch {epoch} - "
                              f"Generated Song (probabilistic) #{i + 1} - "
                              f"word count = {len(generated_song)}:"
                              f"\n{get_song(generated_song)}")

                # Set back to train mode
                self.train()

        # Close the TensorBoard writer
        self.writer.close()

    def predict(self, input_seq, midi_vector, prev_state=None):
        """
        Predict method adds a dimension of probability to the predictions
        by sampling from the output logits.
        """
        self.eval()
        input_seq = input_seq.to(self.device)
        midi_vector = midi_vector.to(self.device)
        with torch.no_grad():
            logits, state = self(input_seq, midi_vector, prev_state=prev_state)

        probabilities = torch.softmax(logits, dim=-1)
        next_word_index = torch.multinomial(probabilities[0, -1], 1).item()

        return next_word_index, state

    def generate_song(self, initial_words: str, melody, max_length=600):
        """
        Support method to generate songs word-by-word starting from an initial word(s) input
        """
        generated_song = initial_words.split()
        current_indices = [
            self.word2vec_model.key_to_index[word] for word in generated_song
        ]
        current_indices = torch.tensor(np.array(current_indices)).unsqueeze(0)
        prev_state = None

        for i in range(max_length - len(generated_song)):
            current_midi = melody[i].unsqueeze(0).unsqueeze(0).to(self.device)
            next_word_index, state = self.predict(current_indices, current_midi, prev_state)

            next_word = self.word2vec_model.index_to_key[next_word_index]

            # if the predicted word is end-of-song token - stop the generation
            if next_word == "eos":
                break

            # Add the generated word to the song
            generated_song.append(next_word)

            # Predict the next word using the last word & the previous state
            current_indices = torch.tensor([next_word_index]).unsqueeze(0).to(self.device)
            prev_state = state

        return generated_song




