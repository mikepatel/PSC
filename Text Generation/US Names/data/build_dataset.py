"""
John Jefferson III and Michael Patel
June 2019
Python 3.6.5
TF 1.12.0

Project Description:
    - Generate our own list of US names using RNNs!

Datasets:
    csv file of US names

Notes:

Things to examine:

"""
################################################################################
# Imports
import os
import re
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup


################################################################################
# 1990s, 2000s, 2010s
datasets = {
    "1990s": {
        "url": "",
        "csv": ""
    },
    "2000s": {
        "url": "",
        "csv": ""
    },
    "2010s": {
        "url": "",
        "csv": ""
    }
}
generic_url = "https://www.ssa.gov/oact/babynames/decades/namesX.html"
generic_csv = os.path.join(os.getcwd(), "X.csv")

for decade in datasets:
    datasets[decade]["url"] = re.sub("X", decade, generic_url)
    datasets[decade]["csv"] = re.sub("X", decade, generic_csv)

print(datasets)
quit()

url = "https://www.ssa.gov/oact/babynames/decades/names2000s.html"

with urllib.request.urlopen(url) as response:
    page = response.read()

soup = BeautifulSoup(page, "html.parser")

tbody = soup.find("tbody")
trs = tbody.find_all("tr")

rows = []

for tr in trs:
    tds = tr.find_all("td")
    row = [tds[i].text for i in range(len(tds))]

    rows.append(row)

rows = rows[:-1]
for r in rows:
    print(r)
