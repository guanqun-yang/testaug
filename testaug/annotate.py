import os
import time
import random

import pandas as pd

from tqdm import tqdm
from termcolor import colored, cprint


def get_current_sys_time():
    return str(time.time()).split(".")[0]


def get_latest_file(folder):
    current_time = time.time()
    filename_dict = {
        filename: current_time - float(filename.stem) for filename in folder.glob("*.json")
        if filename.stem.isdigit() and not os.path.isdir(filename)
    }

    latest_file = min(filename_dict, key=filename_dict.get) if filename_dict != dict() else None

    return latest_file


def save_labeling(records, latest_df, labeling_path):
    new_df = pd.DataFrame(records)
    df = pd.concat([latest_df, new_df], ignore_index=True)
    
    df.to_json(labeling_path / "{}.json".format(get_current_sys_time()), lines=True, orient="records")


####################################################################################################

def parse_capture(capture):
    if capture == "": return 1
    elif capture == "\t": return 0
    else: return None


def check_and_convert_to_string(samples):
    # handle singleton case
    if not isinstance(samples, list): samples = [samples]

    if all(isinstance(sample, tuple) for sample in samples):
        example = "\n\t\t".join(["{}\n\t\t{}".format(s[0], s[1]) for s in samples])
    elif all(isinstance(sample, str) for sample in samples):
        example = "\n\t\t".join(samples)

    else:
        raise NotImplementedError("Unsupported data type")

    return example


def get_annotation(
    df, 
    attributes=["validity"], 
    labeling_path="labeling/",
    save_steps=50,
    log_steps=50
    ):
    # annotation instruction 
    base_instruction = "\n\tEnter - satisfied, Tab - NOT satisfied, Others - relabel current sentence"

    # labeling
    labeling_path.mkdir(parents=True, exist_ok=True)

    df = df[["capability", "description", "template", "label", "demonstration", "text", "pool"]]
    df = df.groupby(["capability", "description", "template", "label"]).agg(list).reset_index()

    already_labeled_df = None
    latest_file = get_latest_file(labeling_path)
    if latest_file:
        already_labeled_df = pd.read_json(latest_file, lines=True, orient="records")


    records = list()
    for _, row in df.iterrows():
        capability = row["capability"]
        desc = row["description"]
        template = row["template"]
        label = row["label"]
        pool = row["pool"][0]

        example = check_and_convert_to_string(row["demonstration"][0])

        total = set(row["text"])

        # proceed with previous labeling efforts
        if latest_file:
            labeled = set(already_labeled_df[already_labeled_df.description == desc].text.tolist())

            if any([isinstance(t, tuple) for t in total]):
                gpts = list(set("\n\t\t".join(t) for t in total) - labeled)
            else:
                gpts = list(total - labeled)
        else:
            gpts = list(total)

        if len(gpts) == 0: 
            cprint(f"[SKIPPING] {desc} had been labeled", "red")
            continue
        else: 
            # permutate the list of texts
            gpts = random.sample(gpts, k=len(gpts))    

        ind = 0
        with tqdm(total=len(gpts)) as pbar:
            while ind < len(gpts):
                gpt = check_and_convert_to_string(gpts[ind])

                instruction = "\n\tLabel the generated sentence based on its validity:"
                instruction += "\n\tA invalid sentence either"
                instruction += "\n\t(1) Does not show the required linguistic capability in description, or"
                instruction += "\n\t(2) Does not have correct label, or"
                instruction += "\n\t(3) Includes private information or offensive contents."
                instruction += colored(f"\n\tDescription:\n\t\t{desc}", "cyan")
                instruction += f"\n\tLabel:\n\t\t{label}"
                instruction += f"\n\tExamples:\n\t\t{example}"
                instruction += colored(f"\n\tGPT:\n\t\t{gpt}", "green")

                if ind < 0: ind = 0
                if ind < len(gpts):
                    print(instruction)
                    
                    label_dict = dict()
                    for attr in attributes:
                        capture_attr = input(base_instruction + "\n\t>> {}: ".format(attr.capitalize()))
                        label_dict[attr] = parse_capture(capture_attr)

                    if any(v is None for v in label_dict.values()):
                        ind -= 1
                        continue
                    else: 
                        ind += 1
                        pbar.update(1)

                records.append(
                    {
                        "capability": capability,
                        "description": desc,
                        "template": template,
                        "text": gpt,
                        "label": label,
                        "pool": pool,
                        **label_dict
                    } 
                )

                # save labeling efforts every 50 sentences
                if len(records) % save_steps == 0: 
                    cprint("[SAVING] saving newly labeled data", "red")
                    save_labeling(records, already_labeled_df, labeling_path)
                
                # show progress every 100 sentences
                if len(records) % log_steps == 0:
                    cprint("[PROGRESS] completed labeling of {} sentences!".format(len(records)), "red")

    save_labeling(records, already_labeled_df, labeling_path)
    