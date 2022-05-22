import argparse
import pandas as pd
import os
from tqdm import tqdm as tqdm
import urllib.request
from urllib.error import HTTPError
from urllib.error import URLError
import numpy as np
import sys

parser = argparse.ArgumentParser(description='r/Fakeddit image downloader')

parser.add_argument('--source', type=str, help='train, validate, or test')
parser.add_argument('--rowcount', type=int, default=200,help='How many imgs you want to download, max 57000')

args = parser.parse_args()

df = pd.read_csv(args.source, sep="\t")
df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)

pbar = tqdm(total=len(df))

img_path = "images"

if not os.path.exists(img_path):
  os.makedirs(img_path)
no_images, no_images_urls, count = 0, [], 0
for index, row in df.iterrows():
  if count > args.rowcount:
    break
  if os.path.exists(row["id"] + ".jpg"):
    continue
  if row["hasImage"] == True and row["image_url"] != "" and row["image_url"] != "nan":
    image_url = row["image_url"]
    try:
      urllib.request.urlretrieve(image_url, filename= f"./{img_path}/{row['id']}.jpg")
      count+=1
    except HTTPError:
      no_images_urls.append(row['image_url'])
      no_images += 1
    except URLError:
      no_images_urls.append(row['image_url'])
      no_images += 1
  pbar.update(1)
print("done")
print('No urls = ', no_images_urls)
print('Count links without images = ', no_images)
print(f'{count-1} images downloaded')
