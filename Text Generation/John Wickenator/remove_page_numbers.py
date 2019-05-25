import os
import re


# read input from file
file = os.path.join(os.getcwd(), "data\\johnwick.txt")
text = [line for line in open(file, encoding="utf-8")]
text = "".join(text)

print(text)
pattern = "Page.*[0-9].|[0-9].*"  # remove page numbers at bottom of page and at top right corner of page
x = re.sub(
    pattern,
    "",
    text
)
print(x)

"""
# write output to file
new_file = os.path.join(os.getcwd(), "data\\new.txt")
with open(new_file, "w+") as f:
    f.write(x)
"""
