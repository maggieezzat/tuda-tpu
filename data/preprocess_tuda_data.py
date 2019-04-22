# coding=utf-8

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


download_dir = "/home/maggieezzat9/TUDA/"
directory = "/home/maggieezzat9/TUDA/german-speechdata-package-v2"
#directory = "E:\Tuda-v2-unmodified.tar\Tuda-v2-unmodified\german-speechdata-package-v2"
tuda_url = "http://speech.tools/kaldi_tuda_de/german-speechdata-package-v2.tar.gz"

# =============================== Vocab ===================================

#   Number patterns
int_pattern = re.compile(r"[0-9]+")
float_pattern = re.compile(r"[0-9]+[,\.][0-9]+")

#   Allowed characters a-zA-Z'äüö
allowed = list(string.ascii_lowercase)
allowed.append("'")
allowed.append(" ")
allowed.extend(list("äöü"))

#   Replacement characters
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

#   Various replacement rules
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

def clean_sentence(sentence):
    """
    Clean the given sentence.
    1. split into words by spaces
    2. numbers to words
    3. character/rule replacements
    4. delete disallowed symbols
    4. join with spaces
    """

    def clean_word(word):
        """
        Clean the given word.
        1. numbers to words
        2. character/rule replacements
        3. delete disallowed symbols
        """


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


        word = word.lower()
        word = word_to_num(word)
        word = replace_symbols(word)
        word = remove_symbols(word)

        bad_chars = get_bad_character(word)

        if len(bad_chars) > 0:
            print('Bad characters in "{}"'.format(word))
            print("--> {}".format(", ".join(bad_chars)))

        return word



    words = sentence.strip().split(" ")
    cleaned_words = []

    for word in words:
        cleaned_word = clean_word(word)
        cleaned_words.append(cleaned_word)

    return " ".join(cleaned_words)

# =============================== End Vocab ===================================

def download_and_extract(directory, url):
    """Download and extract tuda-de dataset.

  Args:
    directory: the directory where to extract the downloaded folder.
    url: the url to download the data file.
  """

    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)

    _, tar_filepath = tempfile.mkstemp(suffix=".tar.gz")

    try:
        tf.logging.info("Downloading %s to %s" % (url, tar_filepath))

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                "\r>> Downloading {} {:.1f}%".format(
                    tar_filepath, 100.0 * count * block_size / total_size
                )
            )
            sys.stdout.flush()

        urllib.request.urlretrieve(url, tar_filepath, _progress)
        print()
        statinfo = os.stat(tar_filepath)
        tf.logging.info(
            "Successfully downloaded %s, size(bytes): %d" % (url, statinfo.st_size)
        )
        with tarfile.open(tar_filepath, "r") as tar:
            tar.extractall(directory)
    finally:
        tf.gfile.Remove(tar_filepath)


def generate_second_list_corrupted_files(directory):
    """Generate corrupted2.txt from Tuda Data
        corrupted.txt i.e. first corrupted list was taken from here: 
        https://github.com/uhh-lt/kaldi-tuda-de/blob/master/s5_r2/local/cleanup/problematic_wavs.txt 
    """
    
    paths = ["test", "dev", "train"]
    corrupted_files = []

    for path in paths:
        files = [
            f
            for f in listdir(join(directory, path))
            if isfile(join(directory, path, f))
        ]

        total_files=len(files)
        processed_files = 0
        
        for file in files:
            processed_files+=1
            if ".wav" in file: 
                print("Checking files from " + path + " set " + str(processed_files) + "/" + str(total_files), end="\r")
                if os.path.getsize(join(directory, path, file)) <= 0:
                    corrupted_files.append(file)
                    continue
                data, _ = soundfile.read(join(directory, path, file))
                if len(data) <= 0:
                    corrupted_files.append(file)

        print()
        print("Done checking " + path + " set")
        print("=====================")

    with open('corrupted2.txt', 'w') as f:
        for file in corrupted_files:
            f.write("%s\n" % file)
    
    print("Done writing corrupted2.txt" +
    "Together with corrupted.txt they contain all corrupted files in Tuda-De")
    print("=====================")


def delete():

    cor = []
    corrupted_lists = ["corrupted.txt", "corrupted2.txt"]
    for corrupted_list in corrupted_lists:
        txtCor = os.path.join(os.path.dirname(__file__), corrupted_list)
        with open(txtCor) as corfile:
            content = corfile.readlines()
        content = [x.strip() for x in content]
        for jk in range(len(content)):
            if corrupted_list == "corrupted.txt":
                cor.append(content[jk][37:57])
            else:
                cor.append(content[jk])


    paths = ["test", "dev", "train"]
    for path in paths:
        files = [
            f
            for f in listdir(join(directory, path))
            if isfile(join(directory, path, f))
        ]
        total_files = len(files)
        processed_files = 0
    
        for file in files:
            processed_files+=1
            print("Deleting from " + path + " " + str(processed_files) + "/" + str(total_files), end="\r")
            if ".wav" in file:
                #remove 3 microphones out of 5 
                #remove the mic condition if you want to keep all mics
                if (("Kinect-Beam" in file) or ("Yamaha" in file) or ("Samson" in file)):
                    remove(join(directory, path, file))
                else:
                    for crptd in cor:
                        if (crptd in file):
                            remove(join(directory, path, file))
                            break

            #fix a corrupted xml file in dev set =)
            if path == "dev" and ".xml" in file:
                fix_xml = False
                xml_name = ""
                newContent = ""
                with open(join(directory, path, file), 'r+', encoding="utf8") as f:
                    content = f.readlines()
                    content = [x.strip() for x in content]
                    for con in content:
                        if con.find("<html><body>") != -1:
                            newContent = con[12:]
                            newContent = newContent[:-14]
                            fix_xml = True
                            xml_name = file
                if fix_xml:
                    os.remove(join(directory, path, xml_name))
                    with open(join(directory, path, xml_name), 'w', encoding="utf8") as f:
                        f.write("%s" % newContent)
                    fix_xml = False
                    xml_name = ""
                    newContent = ""


        print()
        print("Done deleting from " + path + " " + str(processed_files) + "/" + str(total_files))
        print("=====================")


def generate_csv():
 
    paths = ["test", "dev", "train"]
    
    for path in paths:
        csv = []
        files = [
            f
            for f in listdir(join(directory, path))
            if isfile(join(directory, path, f))
        ]
        dir_path = os.path.join(directory, path)
        processed_files = 0
        total_files = len(files)

        for file in files:
            file_path = os.path.join(dir_path, file)
            processed_files+=1
            print("Processing " + path + " " + str(processed_files) + "/" + str(total_files), end="\r")
            if file.endswith(".xml"):
                tree = ET.parse(file_path)
                recording = tree.getroot()
                sent = recording.find("cleaned_sentence")
                sent = sent.text.lower()
                transcript = clean_sentence(sent)
                file_xml, _ = file.split(".", 1)
                found = 0
                for wav_file in files:
                    if wav_file.startswith(file_xml) and wav_file.endswith(".wav"):
                        wav_file_dir = os.path.join(dir_path, wav_file)
                        wav_file_size = os.path.getsize(wav_file_dir)
                        csv.append((wav_file_dir, wav_file_size, transcript))
                        found += 1
                    #remove that check if you keep more than 2 microphones    
                    if found >= 2:
                        break

        print()
        df = pandas.DataFrame(
            data=csv, columns=["wav_filename", "wav_filesize", "transcript"]
        )
        output_file = os.path.join(directory, path + ".csv")
        df.to_csv(output_file, index=False, sep="\t")
        
        print("Successfully generated csv file {}.csv".format(path))
        print("=====================")


def main(_):

    #download_and_extract(download_dir,tuda_url)

    if tf.gfile.Exists(os.path.join(os.path.dirname(__file__), "corrupted2.txt")):
        print("corrupted list 2 already found")
    else:
        generate_second_list_corrupted_files(directory)
    
    delete()
    generate_csv()
   


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    absl_app.run(main)
