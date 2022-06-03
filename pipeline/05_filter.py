import pathlib
import argparse

import pandas as pd

from termcolor import cprint
from simpletransformers.classification import (
    ClassificationArgs,
    ClassificationModel
)
from utils.common import (
    seed_everything,
    load_pickle_file
)
from setting import setting

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["sentiment", "qqp", "nli"])
args = parser.parse_args()

task = args.task

classifier_path = setting.base_path / "classifier"
test_path = setting.data_base_path / pathlib.Path(f"test/{args.task}")
df = pd.concat([pd.read_pickle(filename) for filename in test_path.glob("*.pkl")])
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

cprint("NUMBER OF SENTENCES={}".format(len(df)), "red")

desc_dict = load_pickle_file(setting.data_base_path / pathlib.Path(f"config/{task}.pkl"))

model_args = ClassificationArgs()

model_args.manual_seed = 42
model_args.max_seq_length = 32
model_args.eval_batch_size = 16

records = list()
for idx, desc in desc_dict.items():
    model_name, model_type = str(classifier_path / f"{task}/{idx}/best_model"), "roberta"
    sents = df[df.description == desc].text.tolist()

    # if there is no need to use a classifier, then we automatically assign 1
    if not pathlib.Path(model_name).exists():
        records.extend(
            {
                "text": sent,
                "validity": 1
            } for sent in sents
        )
        continue

    model = ClassificationModel(
        model_type,  # model type
        model_name,  # model name
        num_labels=2,
        args=model_args,
    )

    preds, _ = model.predict(sents)

    records.extend(
        {
            "text": sent,
            "validity": pred
        } for sent, pred in zip(sents, preds)
    )


df = pd.merge(
    left=df,
    right=pd.DataFrame(records),
    on="text"
)
df.to_pickle(setting.data_base_path / f"test/{task}.pkl")



