import pathlib
import argparse

import pandas as pd

from testaug.annotate import get_annotation

from testaug.testaug import (
    generate_gpt3_test_suite
)

from utils.common import (
    get_system_time,
    seed_everything,
    load_pickle_file
)
from utils.data import (
    load_checklist
)

from termcolor import cprint
from setting import setting

parser = argparse.ArgumentParser()
parser.add_argument("--query", action="store_true")
parser.add_argument("--annotate", action="store_true")
parser.add_argument("--size", type=int, default=300)
parser.add_argument("--phase2", action="store_true")
parser.add_argument("--task", choices=["sentiment", "qqp", "nli"])
parser.add_argument("--description", type=int)
args = parser.parse_args()

task = args.task
desc_dict = load_pickle_file(setting.data_base_path / pathlib.Path(f"config/{task}.pkl"))
idx = str(args.description).zfill(3)
desc = desc_dict[idx]

cprint("Query={}".format(desc), "red")

unlabeled_path = setting.data_base_path / pathlib.Path("unlabeled/{}/{}".format(args.task, idx))
if args.phase2: unlabeled_path = unlabeled_path / "train"
unlabeled_path.mkdir(exist_ok=True, parents=True)

labeling_path = setting.data_base_path / pathlib.Path("labeling/{}/{}".format(args.task, idx))
if args.phase2: labeling_path = labeling_path / "train"
labeling_path.mkdir(exist_ok=True, parents=True)

####################################################################################################
# step 1: query

seed_everything(42)

T = load_checklist(task=task, keep_pool=True)
T = T[T.description == desc]

n_unique_template = T.template.nunique()

if args.query:
    df = generate_gpt3_test_suite(
            T,
            task=task,
            model="text-davinci-001",
            n_demonstration=3,
            n_per_query=3,
            n_test_case_per_template=int(args.size / n_unique_template)
            )
    # append the name with system time to store multiple queries
    df.to_pickle(unlabeled_path / "unlabeled@{}.pkl".format(get_system_time()))

####################################################################################################
# step 2: annotate

if args.annotate:
    df = pd.concat([pd.read_pickle(filename) for filename in unlabeled_path.glob("unlabeled*.pkl")])\
           .drop_duplicates(subset=["text"])\
           .reset_index(drop=True)
    get_annotation(df, labeling_path=labeling_path)
