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
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup


################################################################################
# 1990s, 2000s, 2010s
datasets = {
    "1990s": {
        "url": "https://www.ssa.gov/oact/babynames/decades/names1990s.html",
        "csv": ""
    },
    "2000s": {
        "url": "https://www.ssa.gov/oact/babynames/decades/names2000s.html",
        "csv": ""
    },
    "2010s": {
        "url": "https://www.ssa.gov/oact/babynames/decades/names2010s.html",
        "csv": ""
    }
}

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
