# 9/23/18

# Pre-processing on dataset
# remove all URLs
# Text generation using Trump's tweeets


# Notes:


################################################################################
# IMPORTs
import os
import re

current_dir = os.getcwd()
filename = "tweets_og.txt"
new_filename = "tweets_clean_noURLs.txt"
file = os.path.join(current_dir, filename)
new_file = os.path.join(current_dir, new_filename)

rgx_pattern = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

# read in file
text = [line.rstrip("\n") for line in open(file, encoding="utf8")]

# replace all URLs with empty string
new_text = []
for i in text:
    new_text.append(re.sub(rgx_pattern, "", i))

# write to file
with open(new_file, "w+", encoding="utf8") as f:
    for line in new_text:
        f.write(line + "\n")
