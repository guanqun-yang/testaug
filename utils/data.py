import ast
import pathlib

import pandas as pd

from termcolor import colored
from checklist.editor import Editor

from setting import setting


def load_checklist(n_test_case_per_template=10, 
                   do_add_neutral=True,
                   task="sentiment",
                   usecols=None,
                   keep_pool=False):

    assert task in ["sentiment", "qqp", "nli"], colored("[ERROR] task could be either 'sentiment', 'qqp' and 'nli'", "red")

    # nli
    if task == "nli":
        data_df = pd.read_pickle(setting.data_base_path / pathlib.Path("checklist/nli.pkl"))
        data_df = data_df.groupby("template").sample(n=n_test_case_per_template).reset_index(drop=True)
        data_df = data_df.assign(description=data_df.apply(lambda x: "{}{}".format(x["capability"], x["label"]), axis="columns"))
        data_df = data_df[['capability', 'description', 'template', 'label', 'text']]

        if usecols:
            usecols = data_df.columns.intersection(pd.Index(usecols))
            data_df = data_df[usecols]

        return data_df
    
    # sentiment and QQP
    template_path = setting.data_base_path / pathlib.Path(f"checklist/{task}.json")
    template_df = pd.read_json(template_path, lines=True, orient="records")

    editor = Editor()

    records = list()
    for record in template_df.to_dict("records"):
        if (
            task == "sentiment"
            and record["label"] == 1
            and not do_add_neutral
        ):
            continue

        config = {
            "templates": record["template"], 
            "product": True,
            "remove_duplicates": True,
            "mask_only": False,
            "unroll": False,
            "meta": False,
            "save": True,
            "labels": None,
            "nsamples": n_test_case_per_template
            }
        pool = record["pool"]

        try:
            test_cases = editor.template(**config, **pool).data
        except Exception:
            print("[EXCEPTION] {} could not generate test cases".format(record["template"]))
            continue
        
        for test_case in test_cases:
            row = {
                "capability": record["capability"],
                "description": record["description"],
                # "split": record["split"],
                "template": record["template"],
                "label": record["label"] if do_add_neutral else (1 if record["label"] >= 1 else 0),
                "text": test_case
            }
            if keep_pool:
                row["pool"] = record["pool"]

            records.append(row)

    data_df = pd.DataFrame(records)

    # drop duplicates
    data_df = data_df.assign(temp=data_df.text.astype(str))\
                     .drop_duplicates(subset=["temp"], ignore_index=True)\
                     .drop("temp", axis="columns")
    
    # make sure the number of samples is exactly n_test_case_per_template
    data_df = data_df.assign(temp=data_df.template.astype(str))\
                     .groupby("temp")\
                     .sample(n=n_test_case_per_template, replace=True)\
                     .drop("temp", axis="columns")\
                     .reset_index(drop=True)
    
    # convert string representation of list back to list
    if task == "qqp":
        data_df[["text", "template"]] = data_df[["text", "template"]].applymap(lambda x: tuple(ast.literal_eval(x)) if isinstance(x, str) else tuple(x))

    if usecols is not None:
        data_df = data_df[usecols]

    return data_df







