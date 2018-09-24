import os

current_dir = os.getcwd()
filename = "tweets.txt"

file = os.path.join(current_dir, filename)
text = [line.rstrip("\n") for line in open(file, encoding="utf8")]
print(text[:10])
