from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf


with open("gs://deep_speech_bucket/german-speechdata-package-v2/test.csv", 'r') as f:
    content = f.readlines()
    print(content[0])
    print(content[1])

