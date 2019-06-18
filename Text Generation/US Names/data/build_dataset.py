"""
John Jefferson III and Michael Patel
June 2019
Python 3.6.5
TF 1.12.0

Project Description:
    - Generate our own list of US names using RNNs!

Datasets:
    csv files of US names for decades: 1990s, 2000s, 2010s

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
class Dataset:
    def __init__(self):
        self.datasets = {
            "1990s": {
                "url": "",
                "csv": "",
                "rows": [],
                "df": pd.DataFrame()
            },
            "2000s": {
                "url": "",
                "csv": "",
                "rows": [],
                "df": pd.DataFrame()
            },
            "2010s": {
                "url": "",
                "csv": "",
                "rows": [],
                "df": pd.DataFrame()
            }
        }

    # initialize url for each decade
    def init_url(self):
        generic_url = "https://www.ssa.gov/oact/babynames/decades/namesX.html"

        for decade in self.datasets:
            self.datasets[decade]["url"] = re.sub("X", decade, generic_url)

    # initialize csv for each decade
    def init_csv(self):
        generic_csv = os.path.join(os.getcwd(), "X.csv")

        for decade in self.datasets:
            self.datasets[decade]["csv"] = re.sub("X", decade, generic_csv)

    # parse html
    def parse_html(self):
        for decade in self.datasets:
            try:
                url = self.datasets[decade]["url"]

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
                self.datasets[decade]["rows"] = rows

            except Exception as e:
                print(e)

    def build_df(self):
        for decade in self.datasets:
            df = pd.DataFrame()

            rows = self.datasets[decade]["rows"]
            for r in rows:
                rank_df = pd.DataFrame({"Rank": pd.Series(r[0])})
                boys_df = pd.DataFrame({"Boy Name": pd.Series(r[1])})
                boys_freq_df = pd.DataFrame({"Boy Name Frequency": pd.Series(r[2])})
                girls_df = pd.DataFrame({"Girl Name": pd.Series(r[3])})
                girls_freq_df = pd.DataFrame({"Girl Name Frequency": pd.Series(r[4])})

                df_row = pd.concat([rank_df, boys_df, boys_freq_df, girls_df, girls_freq_df], axis=1)
                df = pd.concat([df, df_row])

            self.datasets[decade]["df"] = df

    # write to csv
    def write2csv(self):
        for decade in self.datasets:
            self.datasets[decade]["df"].to_csv(self.datasets[decade]["csv"], index=None)


################################################################################
# Main
if __name__ == "__main__":
    d = Dataset()

    # set up url, csv
    d.init_url()
    d.init_csv()

    # parse html
    d.parse_html()

    # build dataframes
    d.build_df()

    # write to csv
    d.write2csv()
