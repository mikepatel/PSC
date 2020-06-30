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
from datetime import datetime
import urllib.request
from time import sleep

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


################################################################################
# Main
if __name__ == "__main__":
    GENERATED_DIR = os.path.join(os.getcwd(), "generated")

    # chromedriver
    chromedriver_filepath = os.path.join(GENERATED_DIR, "chromedriver.exe")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(
        chromedriver_filepath,
        options=chrome_options
    )

    # open url
    url = "https://www.tweetgen.com/create/tweet.html"
    driver.get(url=url)

    # Theme (light)

    # Profile Picture
    profile_pic_filepath = os.path.join(GENERATED_DIR, "trump_profile.jpg")
    profile_pic_upload = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "pfpInput"))
    )
    profile_pic_upload.send_keys(profile_pic_filepath)

    # Name
    name = "Donald J. Trump"
    name_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "nameInput"))
    )
    name_element.send_keys(name)

    # Username
    username = "realDonaldTrump"
    username_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "usernameInput"))
    )
    username_element.send_keys(username)

    # Verified User
    verify_checkbox = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/div/div[1]/form/div[6]/div/label"))
    )
    verify_checkbox.click()

    # Tweet Content
    tweet_content = "MAGA"
    tweet_box = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "tweetTextInput"))
    )
    tweet_box.send_keys(tweet_content)

    # Image (skip)

    # Time
    time = datetime.now().strftime("%H:%M")
    time_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "time"))
    )
    time_element.send_keys(time)

    # Date
    # Day
    day = datetime.now().strftime("%d")
    day_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "dayInput"))
    )
    day_element.send_keys(day)

    # Month
    month = datetime.now().strftime("%m")
    month_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "monthInput"))
    )
    month_element.send_keys(month)

    # Year
    year = datetime.now().strftime("%Y")
    year_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "yearInput"))
    )
    year_element.send_keys(year)

    # Retweets
    randomize_button = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/div/div[1]/form/div[11]/div[2]/button"))
    )
    randomize_button.click()

    # Likes

    # Client (skip)

    # Generate Image and Download
    generate_button = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="downloadButton"]'))
    )
    generate_button.click()

    # download generated image
    sleep(1)
    gen_image = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "imageOutput"))
    )
    src = gen_image.get_attribute("src")
    urllib.request.urlretrieve(src, os.path.join(GENERATED_DIR, "generated.png"))

    driver.close()
