import os
import time
import pathlib
import argparse
import operator

import pandas as pd

from functools import reduce
from setting import setting

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["sentiment", "qqp", "nli"])
parser.add_argument("--description", type=int)
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

def get_latest_file(folder):
    current_time = time.time()
    filename_dict = {
        filename: current_time - float(filename.stem) for filename in folder.glob("*.json")
        if filename.stem.isdigit() and not os.path.isdir(filename)
    }

    latest_file = min(filename_dict, key=filename_dict.get) if filename_dict != dict() else None

    return latest_file

filename_path = setting.data_base_path / pathlib.Path("labeling/{}/{}".format(args.task, str(args.description).zfill(3)))
train_df = pd.read_json(get_latest_file(filename_path / "train"), lines=True, orient="records")
test_df = pd.read_json(get_latest_file(filename_path), lines=True, orient="records")

if args.task in ["sentiment"]:
    test_texts = test_df.text.tolist()

if args.task in ["qqp", "nli"]:
    train_df.text = train_df.text.apply(lambda x: tuple(x.split("\n\t\t")))
    test_df.text = test_df.text.apply(lambda x: tuple(x.split("\n\t\t")))

    test_texts = reduce(operator.concat, [[tup, tup[::-1]] for tup in test_df.text.tolist()])

# prevent train data leak into the test data
train_df = train_df[~train_df.text.isin(test_texts)]

stat_df = pd.concat((
        train_df.validity.value_counts().to_frame("train").T,
        test_df.validity.value_counts().to_frame("test").T
))

print("Dataset Statistics before Balancing Training Set")
print(stat_df)

train_df = train_df.groupby("validity").sample(n=train_df.validity.value_counts().min()).reset_index(drop=True)

if args.save:
    save_path = setting.data_base_path / pathlib.Path("gpt/{}/{}".format(args.task, str(args.description).zfill(3)))
    save_path.mkdir(parents=True, exist_ok=True)
    train_df.to_pickle(save_path / "train.pkl")
    test_df.to_pickle(save_path / "test.pkl")



