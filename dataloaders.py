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

# Training - lyrics + melody (.mid files with notes, instruments, etc.)
# Test - generate lyrics for melody


# # Get midi files folder and train/test .csv files
# def download_and_extract(repo_url, output_path):
#     # check if the output path exists
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#
#     # download ZIP file
#     response = requests.get(repo_url)
#     if response.status_code == 200:
#         # Extract from ZIP file
#         with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
#             zip_ref.extractall(output_path)
#         print(f"Files extracted to: {output_path}")
#     else:
#         print(f"Failed to download ZIP file: {response.status_code}")
#
# # Example usage
# repo_url = "https://github.com/Or-Gindes/DeepLearningBGU/raw/main/assignment3/Archive.zip?raw=true"  # link to the ZIP file
# output_path = "./"  # output directory
#
# download_and_extract(repo_url, output_path)


class MIDIDataset(Dataset):
  """
  Dataset object for MIDI files.
  Attributes:
    dataset: pandas dataframe with columns: artist, title, lyricsx
    path_to_mids: path to folder with MIDI files
  """
  def __init__(self, path_to_mids: str, dataset: pd.DataFrame):
    self.dataset = dataset.copy()
    self.parse_dataset(path_to_mids)

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset.iloc[idx, :]

  def __getattr__(self, name):
      if name in self.dataset.columns:
          return self.dataset[name]
      raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

  def get_item_from_artist_song_name(self, artist, song_name):
    """
    returns item from dataset based on artist and song name
    :param artist: artist name
    :param song_name: song name
    """
    return self.dataset.loc[self.dataset['artist'] == artist and self.dataset['title'] == song_name, :]

  def parse_dataset(self, path_to_mids):
    """
    Parses dataset and adds:
      directory column - which contains path to MIDI file.
      parsed_mid column - which contains parsed MIDI data.
    """
    self.dataset['directory'] = (path_to_mids + self.dataset.iloc[:, 0].str.strip() + "_-_" + self.dataset.iloc[:, 1].str.strip()).str.strip().str.replace(' ', '_') + ".mid"
    files = []
    for file in os.listdir(path_to_mids):
      if file.endswith('.mid'):
        files.append(os.path.join(path_to_mids, file))
    cleaned_file_names = [re.sub(r'_-_live.*|_-_extended.*|_-_violin.*|-2.mid', '.mid', file.lower()) for file in files]

    directory_dict = dict(zip(cleaned_file_names, files))
    self.dataset['directory'] = [directory_dict.get(song) for song in self.dataset['directory']]
    self.dataset['parsed_mid'] = self.dataset['directory'].map(MIDIDataset.get_midi_data)
    self.dataset = self.dataset.dropna()
    self.dataset.reset_index(drop=True, inplace=True)

  @staticmethod
  def get_midi_data(path_to_mid: str) -> pd.DataFrame:
    """
    Given a path extract features from MIDI file
    :param path_to_mid: path to MIDI file
    :param start_time: start time in seconds from the begining of the song
    :param end_time: end time in seconds from the begining of the song
    :return:
    """
    if not path_to_mid or not os.path.isfile(path_to_mid):
        print(f"File not found: {path_to_mid}")
        return None
    try:
        midi_data = pretty_midi.PrettyMIDI(path_to_mid)
    except Exception as e:
        print(f"Skipping file {path_to_mid}: {e}")
        return None
      # Load MIDI file, handling potential KeySignatureErrors
    # try:
    #   midi_data = pretty_midi.PrettyMIDI(path_to_mid)
    # except Exception as e:
    #   print(f"Skipping file {path_to_mid}: {e}")
    #   return None  # or handle the error differently as needed

    # remove noise
    midi_data.remove_invalid_notes()

    # parse midi to df
    instrument_col = []
    pitch_col = []
    velocity_col = []
    start_col = []
    end_col = []
    for i, instrument in enumerate(midi_data.instruments):
      for j, note in enumerate(instrument.notes):
        start_col.append(note.start)
        end_col.append(note.end)
        instrument_col.append(MIDIDataset.parse_instrument(instrument.name))
        pitch_col.append(note.pitch)
        velocity_col.append(note.velocity)

    end_col = np.array(end_col)
    start_col = np.array(start_col)/np.max(end_col)
    end_col /= np.max(end_col)
    pitch_col = np.array(pitch_col)/127
    velocity_col = np.array(velocity_col)/127

    df = pd.DataFrame({"start": start_col,
                      "end": end_col,
                      "instrument": instrument_col,
                      "duration": end_col - start_col,
                      "pitch": pitch_col,
                      "velocity": velocity_col
                      })

    df = df.sort_values(by=["start", "end"])
    df = df.reset_index(drop=True)

    ohe = OneHotEncoder(categories=[['guitar', 'strings', 'keyboard', 'horns', 'drums', 'melody', 'other']])
    ohe_columns = ohe.fit_transform(df[["instrument"]])
    df.drop(columns=['instrument'], inplace=True)
    df = np.concatenate((df.values, ohe_columns.toarray()), axis=1)
    return df

  @staticmethod
  def parse_instrument(instrument):
    if re.search(r'guitar|bass', instrument.lower()):
      return 'guitar'
    elif re.search(r'violin|cello', instrument.lower()):
      return 'strings'
    elif re.search(r'piano|keyboard|organ', instrument.lower()):
      return 'keyboard'
    elif re.search(r'sax|saxophone|trump.*|clarinet|flute', instrument.lower()):
      return 'horns'
    elif re.search(r'drum.*', instrument.lower()):
      return 'drums'
    elif re.search(r'melody', instrument.lower()):
      return 'melody'
    else:
      return 'other'