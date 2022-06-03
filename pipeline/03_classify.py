import pathlib
import argparse

import pandas as pd

from simpletransformers.classification import (
    ClassificationArgs,
    ClassificationModel
)
from sklearn.metrics import (
    f1_score,
    accuracy_score
)

from utils.common import (
    seed_everything,
    save_pickle_file
)

from setting import setting

seed_everything(42)

parser = argparse.ArgumentParser()

parser.add_argument("--train", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--train_size", type=int)
parser.add_argument("--task", choices=["sentiment", "qqp", "nli"])
parser.add_argument("--description", type=int)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

task = args.task
setup = "-".join(f"{k}={v}" for k, v in vars(args).items())


def process_sentence_pair_dataset(df):
    df = pd.DataFrame(
        [
            {
                "text_a": tup.text[0],
                "text_b": tup.text[1],
                "labels": tup.labels
            }
            for tup in df.itertuples()
        ]
    )
    return df

####################################################################################################
# loading data

idx = str(args.description).zfill(3)
base_path = setting.data_base_path / pathlib.Path("gpt/{}/{}".format(task, idx))

train_file = "train.pkl" if args.train_size is None else "train={}.pkl".format(str(args.train_size).zfill(3))
train_df = pd.read_pickle(base_path / train_file).rename(columns={"validity": "labels"})
eval_df = pd.read_pickle(base_path / "test.pkl").rename(columns={"validity": "labels"})

if args.task == "qqp":
    train_df = process_sentence_pair_dataset(train_df)
    eval_df = process_sentence_pair_dataset(eval_df)

####################################################################################################
# train and test

classifier_path = setting.base_path / "classifier"
model_args = ClassificationArgs()

model_args.manual_seed = 42

model_args.max_seq_length = 32
model_args.train_batch_size = 16
model_args.eval_batch_size = 16
model_args.num_train_epochs = 20

model_args.save_model_every_epoch = False
model_args.save_steps = -1

model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 1000
model_args.evaluate_during_training_verbose = True
model_args.output_dir = str(classifier_path / f"{task}/{idx}")
model_args.cache_dir = str(classifier_path / "cache_dir")
model_args.best_model_dir = str(classifier_path / f"{task}/{idx}/best_model")
model_args.tensorboard_dir = str(classifier_path / f"runs/{task}/{setup}")
model_args.overwrite_output_dir = True

model_name, model_type = "roberta-base" if args.train else f"{task}/{idx}/best_model", "roberta"

model = ClassificationModel(
    model_type,  # model type
    model_name,  # model name
    num_labels=2,
    args=model_args,
)

if args.train:
    model.train_model(
        train_df,
        eval_df=eval_df,
        accuracy=accuracy_score,
        f1=f1_score
    )

if args.test:
    outputs = model.eval_model(
        eval_df,
        accuracy=accuracy_score,
        f1=f1_score
    )

    result_path = classifier_path / pathlib.Path(f"results/{task}")
    result_path.mkdir(exist_ok=True, parents=True)

    save_pickle_file(outputs, result_path / f"{idx}.pkl")