"""Download and preprocess Tuda dataset for DeepSpeech model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os import listdir, remove
from os.path import isfile, join
import soundfile

import codecs
import fnmatch
import os
import sys
import tarfile
import tempfile
import unicodedata

from xml.etree import cElementTree as ET
from xml.dom import minidom
import re

from absl import app as absl_app
from absl import flags as absl_flags
import pandas
from six.moves import urllib
import tensorflow as tf

directory = "C:/Users/MaggieEzzat/Desktop/german-speechdata-package-v2.tar/german-speechdata-package-v2"


def delete():
    filesCount = 0
    with open("C:/Users/MaggieEzzat/Desktop/deep_speech/data/corrupted.txt") as corfile:
        content = corfile.readlines()
    content = [x.strip() for x in content]
    cor = []
    for jk in range(len(content)):
        cor.append(content[jk][37:57])

    paths = ["test", "dev"]
    #,"train"]
    for path in paths:
        files = [f for f in listdir(join(directory, path)) if isfile(join(directory, path, f))]
        deleted = 0
        for file in files:
            if(".xml" in file):
                filesCount+=1
            if ("Kinect-Beam" in file) or ("Yamaha" in file) or ("Samson" in file) :
                deleted += 1
            else :
                for crptd in cor:
                    if( (".wav" in file) and (crptd in file)):
                        deleted += 1
                        break
        sofar = 1
        for file in files:
            if(".wav" in file):
                data, _ = soundfile.read(join(directory, path, file))
                if(len(data) <= 0):
                    remove(join(directory, path, file))
                    print(
                            "Deleting from "
                            + path
                            ,end="\r"
                        )
                    sofar += 1
            elif ("Kinect-Beam" in file) or ("Yamaha" in file) or ("Samson" in file) :
                remove(join(directory, path, file))
                print(
                            "Deleting from "
                            + path
                            + " : "
                            + str(int((sofar / deleted) * 100))
                            + "%",
                            end="\r",
                        )
                sofar += 1
            else :
                for crptd in cor:
                    if( (".wav" in file) and (crptd in file)):
                        remove(join(directory, path, file))
                        print(
                            "Deleting from "
                            + path
                            + " : "
                            + str(int((sofar / deleted) * 100))
                            + "%",
                            end="\r",
                        )
                        sofar += 1
                        break
        filesCount -= deleted
        print()
        print("=====================")
    return filesCount

def generate_csv(filesCount):
    filesSoFar = 1
    data_dir = directory
    sources = ["test", "dev"]
    #, "train"]
    for source_name in sources:
        csv = []
        dir_path=os.path.join(data_dir, source_name)
        for file in os.listdir(dir_path):
            if file.endswith(".xml"):
                tree = ET.parse(os.path.join(dir_path, file))
                recording = tree.getroot()
                if(recording.find('html')):
                    print("error")
                    continue
                sent = recording.find('cleaned_sentence')
                sent = sent.text.lower()
                sent = sent.replace("ä","ae")
                sent = sent.replace("ö","oe")
                sent = sent.replace("ü","ue")
                sent = sent.replace("ß","ss")
                transcript = unicodedata.normalize("NFKD", sent).encode(
                    "utf8", "ignore").decode("utf8", "ignore").strip().lower()
                file_xml,_ = file.split(".",1)
                found = 0
                for file_wav in os.listdir(dir_path):
                    if file_wav.startswith(file_xml) and file_wav.endswith(".wav"):
                        file_wav_dir = os.path.join(dir_path,file_wav)
                        wav_filesize = os.path.getsize(file_wav_dir)
                        csv.append((file_wav_dir, wav_filesize, transcript))
                        found += 1
                    if found >= 2 :
                        break
                #print(source_name + " : Proccessed : " +  file, end="\r")
                print(
                            "Processing : "
                            + str(int((filesSoFar / filesCount) * 100))
                            + "%",
                            end="\r",
                        )
                filesSoFar += 1
        print()
        df = pandas.DataFrame(
            data=csv, columns=["wav_filename", "wav_filesize", "transcript"])
        output_file=os.path.join(data_dir,source_name+".csv")
        df.to_csv(output_file, index=False, sep="\t")

        tf.logging.info("Successfully generated csv file {}".format(output_file))

def main(_):
    fileC = delete()
    generate_csv(fileC)

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  absl_app.run(main)    