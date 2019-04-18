from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
import tensorflow as tf

# pylint: enable=g-bad-import-order

import data.dataset as dataset
import decoder
import deep_speech_model
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers

import math
import random
import collections
# pylint: disable=g-bad-import-order
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import soundfile
import tensorflow as tf


# pylint: enable=g-bad-import-order

import data.featurizer as featurizer  # pylint: disable=g-bad-import-order


def generate_dataset(data_dir):
    """Generate a speech dataset."""
    audio_conf = dataset.AudioConfig(
        sample_rate=16000,
        window_ms=20,
        stride_ms=10,
        normalize=True,
    )
    train_data_conf = dataset.DatasetConfig(
        audio_conf, data_dir, "C:/Users/MariamDesouky/Desktop/tuda-tpu/data/vocabulary.txt", True
    )
    speech_dataset = dataset.DeepSpeechDataset(train_data_conf)
    return speech_dataset


def gen_TFRecord(deep_speech_dataset):

  data_entries = deep_speech_dataset.entries
  num_feature_bins = deep_speech_dataset.num_feature_bins
  audio_featurizer = deep_speech_dataset.audio_featurizer
  feature_normalize = deep_speech_dataset.config.audio_config.normalize
  text_featurizer = deep_speech_dataset.text_featurizer
  writers = []

  records_csv = []

  for data_entry in data_entries:
    #TODO BUCKETT
    writers.append(tf.python_io.TFRecordWriter(data_entry[0][:-4]+".tfrecord"))
    records_csv.append(data_entry[0][:-4]+".tfrecord")

  """Dataset generator function."""

  writer_index = 0
  for audio_file, _, transcript in data_entries:
      features = dataset._preprocess_audio(
          audio_file, audio_featurizer, feature_normalize
      )
      labels = featurizer.compute_label_feature(
          transcript, text_featurizer.token_to_index
      )

      flattened_features = [item for sublist_20ms in features for item in sublist_20ms]
      ff = tf.reshape(flattened_features,[161,1])
      feature_dict = collections.OrderedDict()
      feature_dict["features"] = _bytes_feature(tf.compat.as_bytes(np.asarray(flattened_features).tostring()))
      # create_float_feature(flattened_features)
      feature_dict["shape"] = _bytes_feature(tf.compat.as_bytes(np.asarray([len(features),num_feature_bins,1]).tostring()))
      #create_int_feature([len(features),num_feature_bins,1])
      feature_dict["labels"] = _bytes_feature(tf.compat.as_bytes(np.asarray(labels).tostring()))
      #create_int_feature(labels)

      tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
      writers[writer_index].write(tf_example.SerializeToString())

      writer_index += 1
      print("WRITING: " +str(writer_index + 1)+"/"+ str(len(writers)), end = '\r')
  
  for writer in writers:
    writer.close()

  with open("E:/TUDA/german-speechdata-package-v2/records_test.csv", 'w') as f:
      for record in records_csv:
          f.write(record + '\n')
    

def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature

def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature

def _bytes_feature(value):
     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def input_fn(batch_size, input_files_csv, repeat=1):
    input_files = []
    index = 0
    
    with open(input_files_csv, 'r') as f:
        input_files = f.readlines()
        input_files = [x.strip('\n') for x in input_files]
  
    
    def decode_record(record, name_to_features):

        """Decodes a record to a TensorFlow example."""
        name_to_features = { 
        "features":tf.FixedLenFeature([], tf.string), 
        "shape":tf.FixedLenFeature([], tf.string),
        "labels":tf.FixedLenFeature([], tf.string)
        }
        example = tf.parse_example([record], features=name_to_features)

        # Since the arrays were stored as strings, they are now 1d 
        features_1d = tf.decode_raw(example['features'], tf.float32)
        labels = tf.decode_raw(example['labels'], tf.int32)
        shape = tf.decode_raw(example['shape'], tf.int32)

        # In order to make the arrays in their original shape, they have to be reshaped.
        #label_restored = tf.reshape(label_1d, tf.stack([2, 3, -1]))
        #sample_restored = tf.reshape(sample_1d, tf.stack([2, 3, -1]))
        #print(shape)
        #TODO I have no idea if shape[0] is how it is supposed to be used
        #features_restored = tf.reshape(features_1d, shape[0])
        #print(tf.size(labels))
        return features_1d,labels
    #TODO parallel batches
    dataset = tf.data.TFRecordDataset(input_files_csv)

    #"E:/TUDA/german-speechdata-package-v2/test/2015-02-10-14-33-08_Realtek.tfrecord"

    dataset = dataset.repeat()
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
    decode_record,
    batch_size = batch_size,
    num_parallel_batches = 1,
    drop_remainder = True))
    #print(dataset)
    return dataset


def read_tfRecord():
    record_iterator = tf.python_io.tf_record_iterator(path="E:/TUDA/german-speechdata-package-v2/test/2015-02-10-14-33-08_Realtek.tfrecord")
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
  
  
        print(example)
        break
  
  # Exit after 1 iteration as this is purely demonstrative.

def check_reshape(deep_speech_dataset):
    

  data_entries = deep_speech_dataset.entries
  num_feature_bins = deep_speech_dataset.num_feature_bins
  audio_featurizer = deep_speech_dataset.audio_featurizer
  feature_normalize = deep_speech_dataset.config.audio_config.normalize
  text_featurizer = deep_speech_dataset.text_featurizer
 
  """Dataset generator function."""

 
  for audio_file, _, transcript in data_entries:
      features = dataset._preprocess_audio(
          audio_file, audio_featurizer, feature_normalize
      )
      labels = featurizer.compute_label_feature(
          transcript, text_featurizer.token_to_index
      )

      flattened_features = [item for sublist_20ms in features for item in sublist_20ms]
      #ff = tf.reshape(flattened_features,[features.shape[0],features.shape[1]])
      ff = np.reshape(flattened_features,[features.shape[0],features.shape[1],1])
      #for i in range(len(features)):
       #   for j in range(len(features[i])):
              #print(str(ff[i][j]) +"#############"+str( features[i][j]))
              


      #print(ff == features)
      print(ff[0])
      print("################")
      print(features[0])
      #print(features.shape)
      break


#ds = generate_dataset("E:/TUDA/german-speechdata-package-v2/test.csv")
#check_reshape(ds)
#gen_TFRecord(ds)
#input_fn(128, "E:/TUDA/german-speechdata-package-v2/records_test.csv", 1)
#input_fn(128, "E:/TUDA/german-speechdata-package-v2/records_test.csv", 1)
#read_tfRecord()