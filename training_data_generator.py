import numpy as np
import sys
import pandas as pd
import os
import cv2
import argparse

#
parser = argparse.ArgumentParser(description='r/Fakeddit image downloader')

parser.add_argument('--source', type=str, help='train, validate, or test')
parser.add_argument('--train', type=float,default=0.8,help='training data percent in total')
parser.add_argument('--test', type=float,default=0.2,help='test data percent in total')
parser.add_argument('--vaildation', type=float,default=0.2,help='vaildation data percent in total')

args = parser.parse_args()

df = pd.read_csv(args.source, sep="\t")
df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)
lst = []
fail = []
for index, row in df.iterrows():
    if os.path.exists("./images/" + row["id"] + ".jpg"):
        try:
            image = cv2.imread("./images/" + row["id"] + ".jpg")
        except:
            fail.append( row["id"])
            print('----------------------------------------------------------')
            continue
        if image is None:
            fail.append(row['id'])
            print('----------------------------------------------------------')
            continue
        lst.append([row['clean_title'], row['id'] + ".jpg", row['2_way_label']])
        print(index)
        
print(fail)
new_df = pd.DataFrame(lst, columns=['text','image','label'])
train = new_df.sample(frac=args.train, random_state=100)
test = new_df.sample(frac=args.test, random_state=20)
vaildation = new_df.sample(frac=args.vaildation, random_state=15)

train.to_csv('training_data.csv', index = None, header=True)

test.to_csv('test_data.csv', index = None, header=True)

vaildation.to_csv('vaildation_data.csv', index = None, header=True)
