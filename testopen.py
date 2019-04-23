from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf
import logging
import cloudstorage as gcs
import webapp2

from google.appengine.api import app_identity



def read_file(self, filename):
  self.response.write('Reading the full file contents:\n')

  gcs_file = gcs.open(filename)
  contents = gcs_file.read()
  print(contents)
  gcs_file.close()

read_file("gs://deep_speech_bucket/german-speechdata-package-v2/test.csv")

