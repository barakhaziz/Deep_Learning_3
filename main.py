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
from model import get_song, LyricsGAN, CombinedLyricsLoss, SentimentLoss, LineLenLoss, SongLenLoss, DiversityLoss,EntropyRegularizationLoss, CosineSimilarityLoss
from dataloaders import MIDIDataset
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer



# Load train and test lyrics
def load_lyrics(path: str) -> pd.DataFrame:
  df = pd.read_csv(path, header=None, usecols=[0,1,2])
  df.columns = ["artist", "title", "lyrics"]
  return df

# Load train and test MIDI files, create validation set
train = load_lyrics("lyrics_train_set.csv")
unique_artists = np.array(list(set(train.artist)))
number_of_artists_in_validation = int(len(unique_artists) * 0.1)
random_choice = np.random.choice(range(number_of_artists_in_validation), number_of_artists_in_validation, replace=False)

validation = train[train.artist.isin(unique_artists[random_choice])]
validation.reset_index(drop=True, inplace=True)

train = train[~train.artist.isin(unique_artists[random_choice])]
train.reset_index(drop=True, inplace=True)

train = MIDIDataset("./midi_files/", train)
validation = MIDIDataset("./midi_files/", validation)

test = load_lyrics("lyrics_test_set.csv")
test = MIDIDataset("./midi_files/", test)

print("------------------------------------------\n-----------------Train:-------------------\n------------------------------------------")
#display(train.dataset.info())
print("------------------------------------------\n-----------------Validation:--------------\n------------------------------------------")
#display(validation.dataset.info())
print("------------------------------------------\n-----------------Test:--------------------\n------------------------------------------")
#display(test.dataset.info())


def tokenize_lyrics(lyrics: pd.Series):
  # Maintain line seperation in songs with end-of-line token (eos)
  processed_lyrics = lyrics.apply(lambda song: song.replace(" & ", " eol "))

  # Remove punctuation
  processed_lyrics = processed_lyrics.apply(lambda song: re.sub(r'[^\w\s]', '', song))

  # Remove numbers
  processed_lyrics = processed_lyrics.apply(lambda song: re.sub(r'\d+', '', song))

  # verify all tokens are lower case letters and strip whitespace
  # And add end-of-song token (eos)
  # Remove cases where 'eol' token appears twice in a row
  eol = "eol"
  pattern = rf'\b{re.escape(eol)}\b\s+\b{re.escape(eol)}\b'

  tokens = processed_lyrics.apply(lambda lyrics: [
      word.lower() for word in re.sub(pattern, eol, lyrics).strip().split()
  ] + ["eos"])

  # Modification:
  # Ensure padding so that at least one 'pad' exists in the corpus


  return tokens

tokens = tokenize_lyrics(train.dataset["lyrics"])

song_word_count = tokens.apply(len)

plt.figure()
plt.hist(song_word_count)
quantile_line = song_word_count.quantile(0.9)
plt.plot([quantile_line, quantile_line], [0, 300], ':',
         label=f"90% of songs # Words < {round(quantile_line)}")
plt.xlabel("# Words")
plt.ylabel("# Songs")
plt.title("Number of words for songs in training data")
plt.legend()
plt.show()


special_tokens = pd.Series([["pad", "eos", "eol"]]) # --- Add special tokens before training Word2Vec ---
corpus = pd.concat([
    tokens,
    tokenize_lyrics(validation.dataset["lyrics"]),
    tokenize_lyrics(test.dataset["lyrics"]),
    special_tokens                # <<-- just add to the list!
]).reset_index(drop=True)

word2vec = Word2Vec(
  sentences=corpus,
  vector_size=300,
  window=5, # context window
  min_count=1,
  # min_count=2 would ignore rare words that appear only once in your corpus.
  # min_count=1 means keep all words, even those that appear just once.

)
word2vec = word2vec.wv
word2vec.save("word2vec_lyrics.wordvectors")

word2vec = KeyedVectors.load("word2vec_lyrics.wordvectors", mmap='r')

# seperating melody vectors according to time. We synced the melody to the lyrics by dividing song duration by the number of lyrics, creating a rough estimate of each word’s duration.
class LyricsDataset(Dataset):
  def __init__(
          self,
          tokenized_lyrics: pd.Series(List[str]),
          midi_vectors: pd.Series(List[np.array]),
          word2vec,
          max_seq_length: int = 600,
          auto_encoder=False
  ):
    """
    Dataset object for lyrics.
    :param tokenized_lyrics: list of string tokens representing song lyrics
    :param midi_vectors: preprocess midi vectors
    :param word2vec: trained word2vec model
    :param max_seq_length: maximum length of lyrics tokens
    """
    self.tokenized_lyrics = tokenized_lyrics
    self.max_seq_length = max_seq_length
    self.auto_encoder = auto_encoder
    self.match_seq_length()
    self.mid_vectors = []
    self.divide_midi(midi_vectors)
    self.model = word2vec
    self.vectorize_dataset()

  def match_seq_length(self):
    """
    Add padding or truncate the lyrics to match max_seq_length
    """
    for ind, lyrics in enumerate(self.tokenized_lyrics):
      if len(lyrics) > self.max_seq_length:
        lyrics = lyrics[:self.max_seq_length - len(lyrics) - 1] + ["eos"]
      else:
        lyrics += ["pad"] * (self.max_seq_length - len(lyrics))
      self.tokenized_lyrics[ind] = lyrics

  def vectorize_dataset(self):
    for ind, lyrics in enumerate(self.tokenized_lyrics):
      self.tokenized_lyrics[ind] = [
        self.model.key_to_index[token] for token in lyrics if token in self.model.key_to_index
      ]

  def __len__(self):
    return len(self.tokenized_lyrics)

  def __getitem__(self, index):
    tokens = self.tokenized_lyrics[index]
    input_seq = torch.tensor(np.array(tokens[:-1]), dtype=torch.long)
    target_seq = torch.tensor(np.array(tokens[1:]), dtype=torch.long)
    midi_vector = torch.tensor(np.array(self.mid_vectors[index][1:]),
                               dtype=torch.float32)  # to match length of words vectors
    return input_seq, target_seq, midi_vector

  def divide_midi(self, midi_vectors: pd.Series):
    for song_idx, song in self.tokenized_lyrics.items():
      duration_of_word = 1 / len(song)
      midi = []
      for idx, word in enumerate(song):
        if (word != "pad") and (word != "eos"):
          midi.append(self.get_midi(parsed_midi=midi_vectors[song_idx], idx=idx, duration=duration_of_word))
        else:
          midi.append(np.zeros(12))
      self.mid_vectors.append(midi)

  def replace_midi(self, new_embeddings):
    self.mid_vectors = new_embeddings

  def get_midi(self, parsed_midi: pd.DataFrame, idx: int, duration: float) -> np.array:
    condition = np.logical_and(
      (parsed_midi[:, 0] >= float(idx * duration)),
      (parsed_midi[:, 1] <= float((idx + 1) * duration))
    )
    if self.auto_encoder is False:
      result = np.mean(parsed_midi[condition], axis=0)
      if (len(result) != 12) or (np.isnan(result).any()):
        result = np.zeros(12)
    else:
      result = parsed_midi[condition]
      if (len(result) == 0) or (np.isnan(result).any()):
        result = np.zeros(12)

    return result


train_dataset = LyricsDataset(
  tokenize_lyrics(train.dataset["lyrics"]),
  midi_vectors=train.dataset.parsed_mid,
  max_seq_length=600,
  word2vec=word2vec
)

validation_dataset = LyricsDataset(
  tokenize_lyrics(validation.dataset["lyrics"]),
  midi_vectors=validation.dataset.parsed_mid,
  max_seq_length=600,
  word2vec=word2vec
)
test_dataset = LyricsDataset(
  tokenize_lyrics(test.dataset["lyrics"]),
  midi_vectors=test.dataset.parsed_mid,
  max_seq_length=600,
  word2vec=word2vec
)

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"There are {len(train_dataset.model)} unique 'words' in the model\n"
      f"This includes 4 special tokens (end of line, end of song, padding and chorus)")

def print_length(dataset):
  print(f"Dataset length: {len(dataset.mid_vectors)}")
  print(f"Dataset songs: {len(dataset.mid_vectors[0])}")
  print(f"Dataset songs vector: {dataset.mid_vectors[0][0].shape}")
  print(f"Dataset songs vector: {dataset.mid_vectors[0][0][0]}")
print_length(train_dataset)
print_length(validation_dataset)
print_length(test_dataset)


# use multiple losses
"""
Losses implementation + CombinedLoss class to return a weighted average
"""


try:
  combined_loss = CombinedLyricsLoss(
      word2vec=word2vec,
      loss_func_list=[
          CosineSimilarityLoss(),
          SentimentLoss(),
          LineLenLoss(),
          SongLenLoss(),
          DiversityLoss(),
          EntropyRegularizationLoss(vocab_size=(len(word2vec))),
      ],
      loss_weights=[0.5, 0.1, 0, 0.2, 0.2, 0.1]
  )
except:
  print("loss error")

model = LyricsGAN(word2vec=word2vec, criterion=combined_loss)
model.train_model(
    train_dataloader,
    validation_dataloader,
    epochs=100,
    lr=0.001,
    patience=10
)


def get_song_structure(song):
  line_lengths = [len(line.split()) for line in get_song(song).split('\n')]
  return len(line_lengths), np.mean(line_lengths) if line_lengths else 0


def get_diversity_score(sequence):
  unique_words = len(set(sequence))
  total_words = len(sequence)
  diversity_ratio = unique_words / total_words
  return diversity_ratio


def evaluate_model(model, dataloader, num_generations=5):
  model.eval()
  cosine_sim = nn.CosineSimilarity(dim=0)
  sentiment_loss = SentimentLoss()

  total_sentiment_similarity = 0
  total_cosine_similarity = 0
  total_structure_similarity = 0
  total_diversity_similarity = 0
  total_songs = len(dataloader)

  with torch.no_grad():
    for inputs, _, midi_vector in dataloader:
      inputs = inputs.to(model.device)
      midi_vector = midi_vector.to(model.device)

      original_song = [model.word2vec_model.index_to_key[idx.item()] for idx in inputs[0]] + ["pad"]
      first_word = original_song[0]

      song_sentiment_similarity = 0.0
      song_cosine_similarity = 0.0
      song_structure_similarity = 0.0
      song_diversity_similarity = 0.0

      for _ in range(num_generations):
        generated_song = model.generate_song(first_word, midi_vector[0])
        generated_song += ["pad"] * (len(original_song) - len(generated_song))

        # Calculate sentiment similarity using SentimentLoss
        sentiment_similarity = 1 - sentiment_loss(
          np.array([generated_song]), np.array([original_song])
        )
        song_sentiment_similarity += sentiment_similarity

        # Calculate cosine similarity
        generated_indices = [
          model.word2vec_model.key_to_index[word]
          for word in generated_song if word in model.word2vec_model.key_to_index
        ]
        generated_vec = model.embedding(
          torch.tensor(generated_indices).to(model.device)
        )
        original_vec = model.embedding(torch.cat((inputs, inputs[:, -1].unsqueeze(1)), dim=-1))

        cosine_similarity = torch.mean(cosine_sim(generated_vec, original_vec.squeeze(0))).item()
        song_cosine_similarity += cosine_similarity

        # Calculate structure similarity
        gen_song_len, gen_avg_line_len = get_song_structure(generated_song)
        orig_song_len, orig_avg_line_len = get_song_structure(original_song)

        # Calculate diversity similarity
        gen_diversity = get_diversity_score(generated_song)
        orig_diversity = get_diversity_score(original_song)
        diversity_similarity = 1 - abs(gen_diversity - orig_diversity) / max(gen_diversity, orig_diversity)
        song_diversity_similarity += diversity_similarity

        song_len_similarity = 1 - abs(gen_song_len - orig_song_len) / max(gen_song_len, orig_song_len)
        line_len_similarity = 1 - abs(gen_avg_line_len - orig_avg_line_len) / max(gen_avg_line_len, orig_avg_line_len)
        structure_similarity = (song_len_similarity + line_len_similarity) / 2
        song_structure_similarity += structure_similarity

      # Calculate average similarities for this song and add to the total score
      avg_song_sentiment_similarity = song_sentiment_similarity / num_generations
      avg_song_cosine_similarity = song_cosine_similarity / num_generations
      avg_song_structure_similarity = song_structure_similarity / num_generations
      avg_song_diversity_similarity = song_diversity_similarity / num_generations

      total_sentiment_similarity += avg_song_sentiment_similarity
      total_cosine_similarity += avg_song_cosine_similarity
      total_structure_similarity += avg_song_structure_similarity
      total_diversity_similarity += avg_song_diversity_similarity

  # Calculate overall averages for the dataset
  avg_sentiment_similarity = total_sentiment_similarity / total_songs
  avg_cosine_similarity = total_cosine_similarity / total_songs
  avg_structure_similarity = total_structure_similarity / total_songs
  avg_diversity_similarity = total_diversity_similarity / total_songs

  print(f"Average Sentiment Similarity: {avg_sentiment_similarity:.4f}")
  print(f"Average Cosine Similarity: {avg_cosine_similarity:.4f}")
  print(f"Average Structure Similarity: {avg_structure_similarity:.4f}")
  print(f"Average Diversity Similarity: {avg_diversity_similarity:.4f}")

  return avg_sentiment_similarity, avg_cosine_similarity, avg_structure_similarity, avg_diversity_similarity


model = LyricsGAN(word2vec=word2vec, criterion=combined_loss)
checkpoint = torch.load("best_model.pth")
model.load_state_dict(checkpoint)

evaluate_model(model, test_dataloader)
song_idx = 0
with torch.no_grad():
    for inputs, _, midi_vector in test_dataloader:
        inputs = inputs.to(model.device)
        midi_vector = midi_vector.to(model.device)

        original_song = [model.word2vec_model.index_to_key[idx.item()] for idx in inputs[0]]
        first_word = original_song[0]
        print(f"\nLyrics #{song_idx + 1} - Title: {test.dataset.title[song_idx]}\nSeed: {first_word}\n")
        generated_song = model.generate_song(first_word, midi_vector[0])
        print(get_song(generated_song))
        song_idx += 1
