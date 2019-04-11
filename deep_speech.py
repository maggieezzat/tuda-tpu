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

import string
import collections
import re
import num2words

directory = "E:/TUDA/german-speechdata-package-v2"

# "C:/Users/MaggieEzzat/Desktop/german-speechdata-package-v2.tar/german-speechdata-package-v2"
#
#

# =============================== Vocab ===================================

#
#   Number patterns
#
int_pattern = re.compile(r"[0-9]+")
float_pattern = re.compile(r"[0-9]+[,\.][0-9]+")

#
#   Allowed characters a-zA-Z'äüö
#
allowed = list(string.ascii_lowercase)
allowed.append("'")
allowed.append(" ")
allowed.extend(list("äöü"))

#
#   Replacement characters
#
replacer = {
    "àáâãåāăąǟǡǻȁȃȧ": "a",
    "æǣǽ": "ä",
    "çćĉċč": "c",
    "ďđ": "d",
    "èéêëēĕėęěȅȇȩε": "e",
    "ĝğġģǥǧǵ": "g",
    "ĥħȟ": "h",
    "ìíîïĩīĭįıȉȋ": "i",
    "ĵǰ": "j",
    "ķĸǩǩκ": "k",
    "ĺļľŀł": "l",
    "м": "m",
    "ñńņňŉŋǹ": "n",
    "òóôõøōŏőǫǭǿȍȏðο": "o",
    "œ": "ö",
    "ŕŗřȑȓ": "r",
    "śŝşšș": "s",
    "ţťŧț": "t",
    "ùúûũūŭůűųȕȗ": "u",
    "ŵ": "w",
    "ýÿŷ": "y",
    "źżžȥ": "z",
    "ß": "ss",
    "-­": " ",
}

#
#   Various replacement rules
#

special_replacers = {
    " $ ": "dollar",
    " £ ": "pfund",
    "m³": "kubikmeter",
    "km²": "quadratkilometer",
    "m²": "quadratmeter",
}

replacements = {}
replacements.update(special_replacers)

for all, replacement in replacer.items():
    for to_replace in all:
        replacements[to_replace] = replacement


#
#   Utils
#


def replace_symbols(word):
    """ Apply all replacement characters/rules to the given word. """
    result = word

    for to_replace, replacement in replacements.items():
        result = result.replace(to_replace, replacement)

    return result


def remove_symbols(word):
    """ Remove all symbols that are not allowed. """
    result = word
    bad_characters = []

    for c in result:
        if c not in allowed:
            bad_characters.append(c)

    for c in bad_characters:
        result = result.replace(c, "")

    return result


def word_to_num(word):
    """ Replace numbers with their written representation. """
    result = word

    match = float_pattern.search(result)

    while match is not None:
        num_word = num2words.num2words(
            float(match.group().replace(",", ".")), lang="de"
        ).lower()
        before = result[: match.start()]
        after = result[match.end() :]
        result = " ".join([before, num_word, after])
        match = float_pattern.search(result)

    match = int_pattern.search(result)

    while match is not None:
        num_word = num2words.num2words(int(match.group()), lang="de")
        before = result[: match.start()]
        after = result[match.end() :]
        result = " ".join([before, num_word, after])
        match = int_pattern.search(result)

    return result


def get_bad_character(text):
    """ Return all characters in the text that are not allowed. """
    bad_characters = set()

    for c in text:
        if c not in allowed:
            bad_characters.add(c)

    return bad_characters


def clean_word(word):
    """
    Clean the given word.
    1. numbers to words
    2. character/rule replacements
    3. delete disallowed symbols
    """
    word = word.lower()
    word = word_to_num(word)
    word = replace_symbols(word)
    word = remove_symbols(word)

    bad_chars = get_bad_character(word)

    if len(bad_chars) > 0:
        print('Bad characters in "{}"'.format(word))
        print("--> {}".format(", ".join(bad_chars)))

    return word


def clean_sentence(sentence):
    """
    Clean the given sentence.
    1. split into words by spaces
    2. numbers to words
    3. character/rule replacements
    4. delete disallowed symbols
    4. join with spaces
    """
    words = sentence.strip().split(" ")
    cleaned_words = []

    for word in words:
        cleaned_word = clean_word(word)
        cleaned_words.append(cleaned_word)

    return " ".join(cleaned_words)


# =============================== End Vocab ===================================


def delete():
    filesCount = 0
    txtCor = os.path.join(os.path.dirname(__file__), "corrupted.txt")
    with open(txtCor) as corfile:
        content = corfile.readlines()
    content = [x.strip() for x in content]
    cor = []
    for jk in range(len(content)):
        cor.append(content[jk][37:57])

    paths = ["test", "dev", "train"]
    for path in paths:
        files = [
            f
            for f in listdir(join(directory, path))
            if isfile(join(directory, path, f))
        ]
        deleted = 0
        for file in files:
            if ".xml" in file:
                filesCount += 1
            if ("Kinect-Beam" in file) or ("Yamaha" in file) or ("Samson" in file):
                deleted += 1
            else:
                for crptd in cor:
                    if (".wav" in file) and (crptd in file):
                        deleted += 1
                        break
        sofar = 1
        for file in files:
            if ".wav" in file:
                if os.path.getsize(join(directory, path, file)) <= 0:
                    remove(join(directory, path, file))
                    continue

                data, _ = soundfile.read(join(directory, path, file))
                if len(data) <= 0:
                    remove(join(directory, path, file))
                    print("Deleting from " + path, end="\r")
                    sofar += 1
                elif (
                    ("Kinect-Beam" in file) or ("Yamaha" in file) or ("Samson" in file)
                ):
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
                else:
                    for crptd in cor:
                        if (".wav" in file) and (crptd in file):
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
    sources = ["test", "dev", "train"]
    for source_name in sources:
        csv = []
        dir_path = os.path.join(data_dir, source_name)
        for file in os.listdir(dir_path):
            if file.endswith(".xml"):
                tree = ET.parse(os.path.join(dir_path, file))
                recording = tree.getroot()
                if recording.find("html"):
                    print("error")
                    continue
                sent = recording.find("cleaned_sentence")
                sent = sent.text.lower()
                # sent = sent.replace("ä","ae")
                # sent = sent.replace("ö","oe")
                # sent = sent.replace("ü","ue")
                # sent = sent.replace("ß","ss")
                # transcript = unicodedata.normalize("NFKD", sent).encode(
                #   "utf8", "ignore").decode("utf8", "ignore").strip().lower()
                transcript = clean_sentence(sent)
                file_xml, _ = file.split(".", 1)
                found = 0
                for file_wav in os.listdir(dir_path):
                    if file_wav.startswith(file_xml) and file_wav.endswith(".wav"):
                        file_wav_dir = os.path.join(dir_path, file_wav)
                        wav_filesize = os.path.getsize(file_wav_dir)
                        csv.append((file_wav_dir, wav_filesize, transcript))
                        found += 1
                    if found >= 2:
                        break
                # print(source_name + " : Proccessed : " +  file, end="\r")
                print(
                    "Processing : " + str(int((filesSoFar / filesCount) * 100)) + "%",
                    end="\r",
                )
                filesSoFar += 1
        print()
        df = pandas.DataFrame(
            data=csv, columns=["wav_filename", "wav_filesize", "transcript"]
        )
        output_file = os.path.join(data_dir, source_name + ".csv")
        df.to_csv(output_file, index=False, sep="\t")

        tf.logging.info("Successfully generated csv file {}".format(output_file))


def main(_):
    fileC = delete()
    generate_csv(fileC)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    absl_app.run(main)