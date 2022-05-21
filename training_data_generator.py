import numpy as np
import sys
import pandas as pd
import os
import cv2


df = pd.read_csv('E:/UB2022Spring/CSE676/Projcet/multimodal_test_public.tsv', sep="\t")
df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)
lst = []
fail = []
for index, row in df.iterrows():
    if os.path.exists("./data/" + row["id"] + ".jpg"):
        try:
            image = cv2.imread("./data/" + row["id"] + ".jpg")
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
train = new_df.sample(frac=0.1, random_state=100)
test = new_df.sample(frac=0.02, random_state=20)
vaildation = new_df.sample(frac=0.02, random_state=15)

train.to_csv('training_data.csv', index = None, header=True)

test.to_csv('test_data.csv', index = None, header=True)

vaildation.to_csv('vaildation_data.csv', index = None, header=True)
