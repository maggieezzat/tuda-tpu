from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf


#Extracting transcripts from csv file::::
file_name = os.path.join(os.path.dirname(__file__), "data/dev.csv")
with open(file_name, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()
# Skip the csv header in lines[0].
lines = lines[1:]
# The metadata file is tab separated.
lines = [line.split("\t", 2) for line in lines]
lines = [tuple(line) for line in lines]
print(lines[0])
print(lines[1])

targets = [line[2] for line in lines]  # The ground truth transcript
print(targets[0])
print(targets[1])

