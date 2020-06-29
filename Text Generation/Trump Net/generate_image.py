"""
Michael Patel
June 2020

Project description:
    Trump tweet generator

File description:
    Use Tweetgen.com to generate a tweet image
"""
################################################################################
# Imports
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


################################################################################
# Main
if __name__ == "__main__":
    # chromedriver
    chromedriver_filepath = os.path.join(os.getcwd(), "chromedriver.exe")
    #chrome_options = Options
    driver = webdriver.Chrome(
        chromedriver_filepath
        #options=chrome_options
    )

    # open url
    url = "https://www.tweetgen.com/create/tweet.html"
    driver.get(url=url)

    # Theme (light)

    # Profile Picture
    profile_pic_filepath = os.path.join(os.getcwd(), "trump_profile.jpg")
    profile_pic_upload = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "pfpInput"))
    )
    profile_pic_upload.send_keys(profile_pic_filepath)

    # Name

    # Username

    # Verified User

    # Tweet Content

    # Image (skip)

    # Time

    # Date

    # Retweets

    # Likes

    # Client (skip)

    # Generate Image and Download

    #driver.close()
