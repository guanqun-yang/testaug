import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

import math
import pathlib
import argparse
import operator
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from functools import reduce
from itertools import (
    chain,
    combinations
)

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score
)

from testaug.diversity import (
    self_bleu_n,
    distinct_dependency_paths
)

from testaug.testaug import (
    prepare_testing,
    generate_template_test_suite,
    generate_new_templates,
    balance_sample_by_description,
    test_model
)

from utils.data import (
    load_checklist
)

from utils.common import (
    seed_everything,
    load_pickle_file,
    save_pickle_file,
    set_pandas_display,
    get_latest_file,
    make_cell,
    escape_chars,
)
from collections import defaultdict
from setting import setting

# pandas display
set_pandas_display()

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["sentiment", "qqp", "nli"])
parser.add_argument("--rotation", type=int, default=15)
parser.add_argument("--font_size", type=int, default=15)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--table", action="store_true")
args = parser.parse_args()

# global settings for matplotlib
font = {
    "weight": "bold",
    'size': args.font_size}  # use for title
plt.rc('font', **font)  # sets the default font
plt.rcParams['text.usetex'] = True  # using LaTeX
plt.rcParams['lines.linewidth'] = 4

names = ["template", "gpt3", "expansion"]
seeds = [42, 74, 25, 14, 11]

if args.task == "sentiment":
    caps = ['Negation', 'Temporal', 'SRL', 'Vocabulary']
    models = [
        ("01", "textattack/distilbert-base-cased-SST-2", "distilbert"),
        ("02", "textattack/albert-base-v2-SST-2", "albert"),
        ("03", "textattack/bert-base-uncased-SST-2", "bert"),
        ("04", "textattack/roberta-base-SST-2", "roberta")
    ]
    model_names = [
        "textattack/distilbert-base-cased-SST-2",
        "textattack/albert-base-v2-SST-2",
        "textattack/bert-base-uncased-SST-2",
        "textattack/roberta-base-SST-2"
    ]
    num_labels = 2

if args.task == "qqp":
    caps = ['Temporal', 'Negation', 'Taxonomy', 'SRL', 'Vocabulary', 'Coref']

    models = [
        ("05", "textattack/distilbert-base-cased-QQP", "distillbert"),
        ("06", "textattack/albert-base-v2-QQP", "albert"),
        ("07", "textattack/bert-base-uncased-QQP", "bert")
    ]
    model_names = [
        "textattack/distilbert-base-cased-QQP",
        "textattack/albert-base-v2-QQP",
        "textattack/bert-base-uncased-QQP"
    ]
    num_labels = 3

if args.task == "nli":
    caps = ['QUANTIFIER', 'PRESUPPOSITION', 'WORLD', 'CONDITIONAL', 'SYNTACTIC', 'LEXICAL', 'CAUSAL']
    models = [
        ("08", "textattack/distilbert-base-cased-snli", "distillbert"),
        ("09", "textattack/albert-base-v2-snli", "albert")
    ]
    model_names = [
        "textattack/distilbert-base-cased-snli",
        "textattack/albert-base-v2-snli",
    ]
    num_labels = 3

mode_names = ["template_gpt3_expansion", "template_expansion", "template_gpt3", "template"]

####################################################################################################
record_dict = defaultdict(dict)
records = list()

for idx, model, _ in models:
    for filename in (setting.base_path / pathlib.Path(f"results/{idx}")).rglob("*.pkl"):
        seed = int(filename.stem.split("=")[-1])
        result_dict = load_pickle_file(filename)

        for mode, test_df in result_dict.items():
            y_true = test_df.label.tolist()
            y_before = test_df.before.tolist()
            y_after = test_df.after.tolist()

            records.append(
                {
                    "idx": idx,
                    "model": model,
                    "mode": mode,
                    "seed": seed,
                    "before": 1 - accuracy_score(y_true, y_before),
                    "after": 1 - accuracy_score(y_true, y_after),
                    "reduction": accuracy_score(y_true, y_after) - accuracy_score(y_true, y_before)
                }
            )


df = pd.DataFrame(records)
if args.plot and args.task == "nli":
    mode_names = ["template_gpt3", "template"]

df = df[df["mode"].isin(mode_names)]

if args.table:
    df = df.drop(columns=["idx", "seed"]).groupby(["model", "mode"]).agg(list).reset_index()
    df = df.groupby("model").agg(list)

    records = list()
    for model in model_names:
        before = df.loc[model, "before"][0]

        modes = df.loc[model, "mode"]
        afters = df.loc[model, "after"]
        reductions = df.loc[model, "reduction"]

        records.append(
            {
                "model": model,
                "before": before,
                **{(mode, "after"): after for mode, after in zip(modes, afters)},
                **{(mode, "reduction"): reduction for mode, reduction in zip(modes, reductions)}
            }
        )

    df = pd.DataFrame(records)[
        ["model", "before"] +
        [(mode, col) for mode, col in itertools.product(mode_names, ["after", "reduction"])]
        ]
    df = df.set_index("model")


    heldout_df = pd.read_pickle(setting.base_path / pathlib.Path("dataset/heldout.pkl"))

    heldout_df = heldout_df[heldout_df.task.isin(["sst2", "qqp", "snli"])]
    heldout_df.task = heldout_df.task.map({"sst2": "sentiment", "snli": "nli", "qqp": "qqp"})

    heldout_df = heldout_df[["model", "accuracy"]].set_index("model").rename_axis(index=None)
    heldout_df = 1 - heldout_df
    heldout_df = heldout_df.rename(columns={"accuracy": "error"})

    df = pd.merge(
        left=heldout_df,
        right=df,
        left_index=True,
        right_index=True
    )

    df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: "{:.3f} ({:.3f})".format(np.mean(x), np.std(x)))

    if args.task == "nli":
        df.iloc[:, 4:8] = df.iloc[:, 4:8].applymap(lambda x: "/")

    print(df)
