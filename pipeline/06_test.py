import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

import math
import pathlib
import argparse

import pandas as pd

from itertools import (
    chain,
    combinations
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
    save_pickle_file,
    remove_directories
)
from setting import setting

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["sentiment", "qqp", "nli"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--reproduce", action="store_true")
args = parser.parse_args()

seed_everything(args.seed)

if args.task == "sentiment":
    caps = ['Negation', 'Temporal', 'SRL', 'Vocabulary']
    models = [
        ("01", "textattack/distilbert-base-cased-SST-2", "distilbert"),
        ("02", "textattack/albert-base-v2-SST-2", "albert"),
        ("03", "textattack/bert-base-uncased-SST-2", "bert"),
        ("04", "textattack/roberta-base-SST-2", "roberta")
    ]
    num_labels = 2
if args.task == "qqp":
    caps = ['Temporal', 'Negation', 'Taxonomy', 'SRL', 'Vocabulary', 'Coref']
    models = [
        ("05", "textattack/distilbert-base-cased-QQP", "distilbert"),
        ("06", "textattack/albert-base-v2-QQP", "albert"),
        ("07", "textattack/bert-base-uncased-QQP", "bert")
    ]
    num_labels = 2
if args.task == "nli":
    caps = ['QUANTIFIER', 'PRESUPPOSITION', 'WORLD', 'CONDITIONAL', 'SYNTACTIC', 'LEXICAL', 'CAUSAL']
    models = [
        ("08", "textattack/distilbert-base-cased-snli", "distilbert"),
        ("09", "textattack/albert-base-v2-snli", "albert")
    ]
    num_labels = 3

result_path = setting.base_path / "results"
template_df = load_checklist(task=args.task, do_add_neutral=False, keep_pool=True)

for idx, model_name, model_type in models:
    n_unique_template = template_df.template.nunique()
    n = n_unique_template / 2 - 1 if n_unique_template % 2 == 0 else math.floor(n_unique_template / 2)
    T1, T2 = balance_sample_by_description(template_df, n=n)

    if args.task != "nli":
        df1 = generate_template_test_suite(T1, n_test_case_per_template=100).sample(n=100)
    else:
        df1 = load_checklist(task="nli")
        df1 = df1[df1.template.isin(T1.template.tolist())].sample(n=100)

    if args.reproduce:
        df2 = pd.read_pickle(setting.data_base_path / pathlib.Path(f"test_suites/{args.task}.pkl"))
    else:
        df2 = pd.read_pickle(setting.data_base_path / pathlib.Path(f"test/{args.task}.pkl"))
    df2 = df2.sample(n=100)

    df3 = None
    if args.task != "nli":
        T3 = generate_new_templates(df2)
        df3 = generate_template_test_suite(
            T3,
            n_test_case_per_template=100,
            template_column="new_template"
        )
        if len(df3) >= 100: df3 = df3.sample(n=100)

    names = ["template", "gpt3", "expansion"]
    df_dict = {
        "template": df1,
        "gpt3": df2,
        "expansion": df3
    }

    dfs = list()
    result_dict = dict()
    full_patch_df, test_df = prepare_testing(df_dict)
    for subset in chain(*map(lambda x: combinations(names, x), range(1, len(names)+1))):
        remove_directories(["testing/"])

        patch_df = full_patch_df[full_patch_df.source.isin(subset)]
        if patch_df.empty:
            continue

        test_df = test_model(model_name, model_type, args.task, num_labels, patch_df, test_df)

        mode = "_".join(subset)
        result_dict[mode] = test_df

    (result_path / idx).mkdir(parents=True, exist_ok=True)
    save_pickle_file(result_dict, result_path / pathlib.Path(f"{idx}/seed={args.seed}.pkl"))