from os import listdir, remove
from os.path import isfile, join
import time
directory = "ScriptData"
paths = ["train", "test", "dev"]
for path in paths:
    files = [f for f in listdir(join(directory,path)) if isfile(join(directory,path, f))]
    deleted = 0
    for file in files:
        if ("Kinect-Beam" in file) or ("Yamaha" in file):
            deleted += 1
    sofar = 1
    for file in files:
        if ("Kinect-Beam" in file) or ("Yamaha" in file):
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
    print()
