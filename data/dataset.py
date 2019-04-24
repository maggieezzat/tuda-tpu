#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
"""Generate tf.data.Dataset object for deep speech training/evaluation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import numpy as np
from six.moves import xrange 
import soundfile
import os
import tensorflow as tf
from absl import app as absl_app

import featurizer as featurizer


# Default vocabulary file
_VOCABULARY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocabulary.txt")


class AudioConfig(object):
    """Configs for spectrogram extraction from audio."""

    def __init__(self, sample_rate, window_ms, stride_ms, normalize=False):
        """Initialize the AudioConfig class.

    Args:
      sample_rate: an integer denoting the sample rate of the input waveform.
      window_ms: an integer for the length of a spectrogram frame, in ms.
      stride_ms: an integer for the frame stride, in ms.
      normalize: a boolean for whether apply normalization on the audio feature.
    """

        self.sample_rate = sample_rate
        self.window_ms = window_ms
        self.stride_ms = stride_ms
        self.normalize = normalize


class DatasetConfig(object):
    """Config class for generating the DeepSpeechDataset."""

    def __init__(self, audio_config, data_path, vocab_file_path, sortagrad):
        """Initialize the configs for deep speech dataset.

    Args:
      audio_config: AudioConfig object specifying the audio-related configs.
      data_path: a string denoting the full path of a manifest file.
      vocab_file_path: a string specifying the vocabulary file path.
      sortagrad: a boolean, if set to true, audio sequences will be fed by
                increasing length in the first training epoch, which will
                expedite network convergence.

    Raises:
      RuntimeError: file path not exist.
    """

        self.audio_config = audio_config
        assert tf.gfile.Exists(data_path)
        assert tf.gfile.Exists(vocab_file_path)
        self.data_path = data_path
        self.vocab_file_path = vocab_file_path
        self.sortagrad = sortagrad


def _normalize_audio_feature(audio_feature):
    """Perform mean and variance normalization on the spectrogram feature.

  Args:
    audio_feature: a numpy array for the spectrogram feature.

  Returns:
    a numpy array of the normalized spectrogram.
  """
    mean = np.mean(audio_feature, axis=0)
    var = np.var(audio_feature, axis=0)
    normalized = (audio_feature - mean) / (np.sqrt(var) + 1e-6)

    return normalized


def _preprocess_audio(audio_file_path, audio_featurizer, normalize):
    """Load the audio file and compute spectrogram feature."""
    data, _ = soundfile.read(audio_file_path)

    feature = featurizer.compute_spectrogram_feature(
        data,
        audio_featurizer.sample_rate,
        audio_featurizer.stride_ms,
        audio_featurizer.window_ms,
    )
    # Feature normalization
    if normalize:
        feature = _normalize_audio_feature(feature)

    # Adding Channel dimension for conv2D input.
    #feature = np.expand_dims(feature, axis=2)
    return feature


def _preprocess_data(file_path):
    """Generate a list of tuples (wav_filename, wav_filesize, transcript).

  Each dataset file contains three columns: "wav_filename", "wav_filesize",
  and "transcript". This function parses the csv file and stores each example
  by the increasing order of audio length (indicated by wav_filesize).
  AS the waveforms are ordered in increasing length, audio samples in a
  mini-batch have similar length.

  Args:
    file_path: a string specifying the csv file path for a dataset.

  Returns:
    A list of tuples (wav_filename, wav_filesize, transcript) sorted by
    file_size.
  """
    tf.logging.info("Loading data set {}".format(file_path))
    with tf.gfile.Open(file_path, "r") as f:
        lines = f.read().splitlines()
    # Skip the csv header in lines[0].
    lines = lines[1:]
    # The metadata file is tab separated.
    lines = [line.split("\t", 2) for line in lines]
    # Sort input data by the length of audio sequence.
    lines.sort(key=lambda item: int(item[1]))

    return [tuple(line) for line in lines]


class DeepSpeechDataset(object):
    """Dataset class for training/evaluation of DeepSpeech model."""

    def __init__(self, dataset_config):
        """Initialize the DeepSpeechDataset class.

    Args:
      dataset_config: DatasetConfig object.
    """
        self.config = dataset_config
        # Instantiate audio feature extractor.
        self.audio_featurizer = featurizer.AudioFeaturizer(
            sample_rate=self.config.audio_config.sample_rate,
            window_ms=self.config.audio_config.window_ms,
            stride_ms=self.config.audio_config.stride_ms,
        )
        # Instantiate text feature extractor.
        self.text_featurizer = featurizer.TextFeaturizer(
            vocab_file=self.config.vocab_file_path
        )

        self.speech_labels = self.text_featurizer.speech_labels
        self.entries = _preprocess_data(self.config.data_path)
        # The generated spectrogram will have 161 feature bins.
        self.num_feature_bins = 161


def batch_wise_dataset_shuffle(entries, epoch_index, sortagrad, batch_size):
    """Batch-wise shuffling of the data entries.

  Each data entry is in the format of (audio_file, file_size, transcript).
  If epoch_index is 0 and sortagrad is true, we don't perform shuffling and
  return entries in sorted file_size order. Otherwise, do batch_wise shuffling.

  Args:
    entries: a list of data entries.
    epoch_index: an integer of epoch index
    sortagrad: a boolean to control whether sorting the audio in the first
      training epoch.
    batch_size: an integer for the batch size.

  Returns:
    The shuffled data entries.
  """
    shuffled_entries = []
    if epoch_index == 0 and sortagrad:
        # No need to shuffle.
        shuffled_entries = entries
    else:
        # Shuffle entries batch-wise.
        max_buckets = int(math.floor(len(entries) / batch_size))
        total_buckets = [i for i in xrange(max_buckets)]
        random.shuffle(total_buckets)
        shuffled_entries = []
        for i in total_buckets:
            shuffled_entries.extend(entries[i * batch_size : (i + 1) * batch_size])
        # If the last batch doesn't contain enough batch_size examples,
        # just append it to the shuffled_entries.
        shuffled_entries.extend(entries[max_buckets * batch_size :])

    return shuffled_entries
          

def generate_dataset(data_dir, vocab_file=_VOCABULARY_FILE, sortagrad=True):
    print(vocab_file)
    """Generate a speech dataset."""
    audio_conf = AudioConfig(
        sample_rate=16000,
        window_ms=20,
        stride_ms=10,
        normalize=True,
    )

    train_data_conf = DatasetConfig(
        audio_conf, data_dir, vocab_file, sortagrad
    )
    speech_dataset = DeepSpeechDataset(train_data_conf)
    return speech_dataset


def pad_features(features, max_features_length, padding_values):  

  len_to_be_padded = max_features_length - len(features)
  exact = ( len_to_be_padded // len(padding_values) ) * 10
  extra = len_to_be_padded % len(padding_values)
  while exact > 0:
    features = np.concatenate((features, padding_values), axis=0)
    exact-=10
  features = np.concatenate((features, padding_values[:extra]), axis=0) 
  return features


def convert_to_TF(deep_speech_dataset, tf_records_path):
  data_entries = deep_speech_dataset.entries
  num_feature_bins = deep_speech_dataset.num_feature_bins
  audio_featurizer = deep_speech_dataset.audio_featurizer
  feature_normalize = deep_speech_dataset.config.audio_config.normalize
  text_featurizer = deep_speech_dataset.text_featurizer
  
  EOSindex = text_featurizer.token_to_index['$']
  print('Writing ', tf_records_path)
  max_features_length = -1
  max_labels_length = -1
  featuresA =[]
  labelsA = []

  for audio_file, _, transcript in data_entries:
    features = _preprocess_audio(
        audio_file, audio_featurizer, feature_normalize
    )
    labels = featurizer.compute_label_feature(
        transcript, text_featurizer.token_to_index
    )
    if(len(features) > max_features_length):
      max_features_length = len(features)
    if(len(labels) > max_labels_length):
      max_labels_length = len(labels)

    featuresA.append(features)
    labelsA.append(labels)

  #To make the  character '$' the end of sentence   
  max_labels_length +=1

  print("\n\n\n\n\n\n\n\n\n")
  print(max_features_length)
  print(max_labels_length)
  print("\n\n\n\n\n\n\n\n\n")

  with tf.python_io.TFRecordWriter(tf_records_path) as writer:
    for index in range(len(featuresA)):
      #pad features to max_features_length
      features = featuresA[index]
      features = pad_features(features, max_features_length, features[-10:])
      #pad labels to max_labels_length
      labels = labelsA[index]
      labels = labels +([EOSindex] * (max_labels_length - len(labels)))
      #flatten features array
      flattened_features = [item for sublist_20ms in features for item in sublist_20ms]
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'labels':_int64_feature(labels),
                  'features': create_float_feature(flattened_features),
                  'input_length': _int64_feature([max_features_length]),
                  'label_length': _int64_feature([max_labels_length]),
              }))
      print("Writing File: ", str(index), "/", str(len(data_entries)), end='\r')

    writer.write(example.SerializeToString())
    return (max_features_length, max_labels_length)


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def write_features_and_labels_lengths(max_features_length, max_labels_length, file_path):
    with open(file_path, 'w') as f:
        f.write(str(max_features_length) + '\n')
        f.write(str(max_labels_length) + '\n')


def input_fn(batch_size, tfrecord_input, max_features_length, max_labels_length, repeat=1):
    
    def decode_record(record, name_to_features):

        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, features=name_to_features)

        features_1d = tf.cast(example['features'], tf.float32)
        features = tf.reshape(features_1d,tf.stack([max_features_length,161,1]))
        labels = tf.cast(example['labels'], tf.int32)
        labels = tf.reshape(labels,tf.stack([max_labels_length]))
        
        input_length = tf.cast(example['input_length'], tf.int32)
        label_length = tf.cast(example['label_length'], tf.int32)

        return ({"features":features,
                "input_length":input_length,
                "label_length":label_length
                },labels)

    #TODO parallel batches
    dataset = tf.data.TFRecordDataset(tfrecord_input)

    name_to_features = { 
        "features":tf.FixedLenFeature([max_features_length*161], tf.float32), 
        "labels":tf.FixedLenFeature([max_labels_length], tf.int64),
        "input_length":tf.FixedLenFeature([1], tf.int64),
        "label_length":tf.FixedLenFeature([1], tf.int64)
 
        }

    dataset = dataset.repeat()
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        #decode_record,
        lambda record: decode_record(record, name_to_features),
        batch_size = batch_size,
        num_parallel_batches = 1,
        drop_remainder = True))
 
    return dataset


def export_speech_labels():
  text_featurizer = featurizer.TextFeaturizer(
            vocab_file=_VOCABULARY_FILE
        )
  speech_labels = text_featurizer.speech_labels
  return speech_labels


def main(_):

    #defining some paths
    root_dir = "E:/TUDA/german-speechdata-package-v2/"
    #train_csv = os.path.join(root_dir, "train.csv")
    train_csv = "/content/train.csv"
    test_csv = os.path.join(root_dir, "test.csv")
    dev_csv = os.path.join(root_dir, "dev.csv")
    #train_tfrecords = os.path.join(root_dir, "train.tfrecords")
    train_tfrecords = "/content/train.tfrecords"
    test_tfrecords = os.path.join(root_dir, "test.tfrecords")
    dev_tfrecords = os.path.join(root_dir, "dev.tfrecords")
    train_set_lengths = "/content/train_set_lengths.txt"
    test_set_lengths = os.path.join(root_dir, "test_set_lengths.txt")
    dev_set_lengths = os.path.join(root_dir, "dev_set_lengths.txt")


    #generating train tfrecords file and train lengths file
    train_ds = generate_dataset(train_csv)
    (max_features_train, max_labels_train) = convert_to_TF(train_ds, train_tfrecords)
    write_features_and_labels_lengths(max_features_train, max_labels_train, train_set_lengths)

    #generating test tfrecords file and test lengths file
    #test_ds = generate_dataset(test_csv)
    #(max_features_test, max_labels_test) = convert_to_TF(test_ds, test_tfrecords)
    #write_features_and_labels_lengths(max_features_test, max_labels_test, test_set_lengths)
    
    #generating dev tfrecords file and dev lengths file
    #dev_ds = generate_dataset(dev_csv)
    #(max_features_dev, max_labels_dev) = convert_to_TF(dev_ds, dev_tfrecords)
    #write_features_and_labels_lengths(max_features_dev, max_labels_dev, dev_set_lengths)



if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    absl_app.run(main)