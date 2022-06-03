import re
import os
import json
import time
import math
import torch
import pickle
import shutil
import random
import string
import hashlib
import pathlib

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from PIL import Image, ImageFont, ImageDraw, ImageColor
from nltk import corpus
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models
from torchvision import datasets
from torchvision import transforms 

from sklearn.model_selection import train_test_split

from datetime import datetime, timedelta
from collections import defaultdict


def get_system_time():
    return str(time.time()).split(".")[0]


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def save_text_file(record_list, filename):
    result = ""
    for record in record_list:
        result += f"{record}\n"

    with open(filename, "w") as fp: fp.write(result)


def load_text_file(filename):
    if not pathlib.Path(filename).exists(): 
        print("FILE NOT EXISTENT")
        return None

    with open(filename, "r", errors="ignore") as fp:
        return fp.readlines()


def save_json_file(record_list, filename):
    result = ""
    for record in record_list:
        result += "%s\n" % json.dumps(record)
    
    with open(filename, "w") as fp: fp.write(result)


def load_json_file(filename):
    record_list = list()
    with open(filename, "r", errors="ignore") as fp:
        for line in fp.readlines():
            record_list.append(json.loads(line))
    
    return record_list


def save_pickle_file(data, filename):
    with open(filename, "wb") as fp:
        pickle.dump(data, fp)


def load_pickle_file(filename):
    data = None
    with open(filename, "rb") as fp:
        data = pickle.load(fp)

    return data


def get_line_cnt(path):
    cnt = None
    with open(path, "r", errors="ignore") as fp:
        cnt = len(fp.readlines())
    return cnt


def create_directories(path_list):
    # path_list: a list of paths in the current directory in string format
    for path in path_list:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def remove_directories(path_list):
    # path_list: a list of paths in the current directory in string format
    for path in path_list:
        if os.path.exists(path): shutil.rmtree(path)



def print_long_text(text, max_char=50):
    cnt = 0
    result = ""
    for token in re.split(r"\s+", text):
        cnt += len(token)
        result += token + " "
        if cnt >= max_char:
            cnt = 0
            result += "\n"
    
    print(result)


def print_k_per_line(lst, k=5):
    result = ""
    for idx, item in enumerate(lst):
        result += f"{item}, "
        if (idx + 1) % k == 0:
            print(result)
            result = ""
    print(result)


def convert_text_to_shake(text):
    return hashlib.shake_128(bytes(text, encoding="raw_unicode_escape")).hexdigest(5)


def generate_random_string(k):
    return "".join(random.sample(string.punctuation + string.ascii_letters + " ", k))


def generate_random_sentence():
    lst = [" ".join(tokens) for tokens in corpus.gutenberg.sents('shakespeare-macbeth.txt') if len(tokens) >= 10]
    return random.choice(lst)


def print_tuple(tuple_list, max_char_length=50):
    # extended version of print_triplet() that accepts tuple of any length
    # tuple_list: [(a1, b1, c1, d1,...), (a2, b2, c2, d2,...), ...]

    # if any of the tuple has more than n_token tokens, ignore extra tokens
    n_token = min(map(len, tuple_list))
    token_list_dict = defaultdict(list)

    length = 0
    full_string = ""
    string_format = ""
    for tup in tuple_list:    
        # length    
        max_len = max(map(len, tup))
        length += max_len

        # print format
        string_format += "{:<%d" % (max_len + 2) + "}"

        for i in range(n_token): token_list_dict[i].append(tup[i])

        if length >= max_char_length:
            # append
            for token_list in token_list_dict.values():
                full_string += "%s\n" % string_format.format(*token_list)
            full_string += "\n"

            # reset
            length = 0
            string_format = ""
            token_list_dict = defaultdict(list)
    
    # when remaining tokens is shorter than max_char_length, append remaining tokens
    for token_list in token_list_dict.values():
        full_string += "%s\n" % string_format.format(*token_list)

    print(full_string)


# in case the transformers.trainer_utils do not have this function
def get_last_checkpoint(folder):
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")

    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


def get_folder_name(path, n_digit=3):
    pattern ="\d{" + f"{n_digit}" + "}"
    folders = [folder.name for folder in filter(lambda folder: folder.is_dir() and re.match(pattern, folder.name),
                                                list(pathlib.Path(path).glob("*")))]
    
    max_folder_number = 0
    if folders: max_folder_number = max([int(folder.lstrip("0")) for folder in folders])

    return str(max_folder_number + 1).zfill(3)

def get_latest_file(folder):
    current_time = time.time()
    filename_dict = {
        filename: current_time - float(filename.stem) for filename in folder.glob("*.json")
        if filename.stem.isdigit() and not os.path.isdir(filename)
    }

    latest_file = min(filename_dict, key=filename_dict.get) if filename_dict != dict() else None

    return latest_file

####################################################################################################

def parse_termcolor_color(text):
    # termcolor only supports limited number of colors (https://pypi.org/project/termcolor/)
    # the colors are specified by syntax: \x1b[3<idx>m<text>\x1b[0m, where idx refers to a color

    prefix_pattern = "\\x1b\[3(\d)m"
    suffix_pattern = "\\x1b\[0m"

    colors = ['grey', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']

    try:
        search_result = re.search(prefix_pattern, text)
        color = ImageColor.getrgb(colors[int(search_result[1])])
    except Exception:
        color = (0, 0, 0)

    for pattern in [prefix_pattern, suffix_pattern]: text = re.sub(pattern, "", text)
    
    return text, color


def convert_text_to_image(textfile_path):
    # adapated from: https://stackoverflow.com/a/29775654/7784797
    # usage:
    #   path = random.choice(list(pathlib.Path("data/").glob("*.txt")))
    #   image = convert_text_to_image(path)
    #   image.show()
    #   image.save('content.png')

    PIL_GRAYSCALE = 'L'
    PIL_RGB = "RGB"
    PIL_WIDTH_INDEX = 0
    PIL_HEIGHT_INDEX = 1

    with open(textfile_path) as f:
        lines = tuple(line.rstrip() for line in f.readlines())

    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", size=20)
    except IOError:
        font = ImageFont.load_default()

    # make a sufficiently sized background image based on the combination of font and lines
    font_points_to_pixels = lambda pt: round(pt * 96.0 / 72)
    margin_pixels = 20

    # height of the background image
    tallest_line = max(lines, key=lambda line: font.getsize(line)[PIL_HEIGHT_INDEX])
    max_line_height = font_points_to_pixels(font.getsize(tallest_line)[PIL_HEIGHT_INDEX])
    realistic_line_height = max_line_height * 0.8  # apparently it measures a lot of space above visible content
    image_height = int(math.ceil(realistic_line_height * len(lines) + 2 * margin_pixels))

    # width of the background image
    widest_line = max(lines, key=lambda s: font.getsize(s)[PIL_WIDTH_INDEX])
    max_line_width = font_points_to_pixels(font.getsize(widest_line)[PIL_WIDTH_INDEX])
    image_width = int(math.ceil(max_line_width + (2 * margin_pixels)))

    # draw the background
    # background_color = 255  # white
    # image = Image.new(PIL_GRAYSCALE, (image_width, image_height), color=background_color)
    
    background_color = (255, 255, 255)
    image = Image.new(PIL_RGB, (image_width, image_height), color=background_color)
    draw = ImageDraw.Draw(image)

    # draw each line of text
    horizontal_position = margin_pixels

    for i, line in enumerate(lines):
        vertical_position = int(round(margin_pixels + (i * realistic_line_height)))

        # NOTE: 15 is a magic number here
        line = line.replace("\t", " " * 8)
        start_position = horizontal_position + 15 * (len(line) - len(line.lstrip()))

        text, font_color = parse_termcolor_color(line)

        # specify color: https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html#colors
        draw.text((start_position, vertical_position), text.strip(), fill=font_color, font=font)

    return image

####################################################################################################

def set_pandas_display(max_colwidth=100):
    pd.options.display.max_rows = None
    pd.options.display.max_colwidth = max_colwidth
    pd.options.display.max_columns = None


####################################################################################################

def preprocess_text(text):
    # remove links
    pattern = re.compile(r'https?://\S+|www\.\S+')
    text = pattern.sub(r'', text)

    # remove stop words
    pattern = re.compile(r'\b(' + r'|'.join(corpus.stopwords.words('english')) + r')\b\s*')
    text = pattern.sub(r'', text)

    # remove emojis
    pattern = re.compile("["
                         u"\U0001F600-\U0001F64F"
                         u"\U0001F300-\U0001F5FF"
                         u"\U0001F680-\U0001F6FF"
                         u"\U0001F1E0-\U0001F1FF"
                         u"\U00002702-\U000027B0"
                         u"\U000024C2-\U0001F251"
                         "]+", flags=re.UNICODE)
    text = pattern.sub(r'', text)

    # remove HTML tags
    text = BeautifulSoup(text, "lxml").get_text()

    # remove speacial characters (for example, punctuations)
    text = re.sub(r"[^a-zA-Z\d]", " ", text)

    # remove extra spaces
    text = re.sub(' +', ' ', text)

    # remove space at the beginning and the end
    text = text.strip()

    return text


def update_json_options(filename, option_dict):
    with open(filename, "r") as fp:
        data = json.load(fp)

    for key in option_dict.keys():
        if key not in data.keys(): continue
        data[key] = option_dict[key]
    
    with open(filename, "w") as fp:
        json.dump(data, fp, indent=4, sort_keys=True)


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_image_dataloader(name="mnist", size=100, batch_size=8):
    transform = transforms.Compose([transforms.ToTensor()])

    if name == "mnist":
        dataset = datasets.MNIST(".", download=True, train=True, transform=transform)
    if name == "cifar10":
        dataset = datasets.CIFAR10(".", download=True, train=True, transform=transform)

    train_idx, val_idx = train_test_split(
        np.random.choice(np.arange(len(dataset)), size),
        test_size=0.2, 
        shuffle=True
        )
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler)
    

    return train_dataloader, val_dataloader


def load_dummy_network(num_input=784, num_class=10):
    return nn.Sequential(
        nn.Linear(num_input, 100),
        nn.ReLU(),
        nn.Linear(100, num_class)
    )


def show_file_change(filename):
    delta = timedelta(seconds=time.time() - os.stat(filename).st_mtime)
    print("The file {} was updated {} seconds ago".format(filename, delta.seconds))

####################################################################################################
# latex related

def limit_length(text, max_len=30):
    output = ""
    for i in range(0, len(text), max_len):
        output = output + "\\\\ " + text[i:i+max_len]
    
    output = "\makecell[l]{" + output.strip("\\\\") + "}"
    return output


def make_cell(text1, text2, max_len=None):
    if max_len:
        text1 = limit_length(text1, max_len)
        text2 = limit_length(text2, max_len)

    return r"\begin{tabular}{@{}l@{}}" + f"{text1}" + r"\\" + f"{text2}" + r"\end{tabular}"


def escape_chars(text):
    chars = ["$", "_", "{", "}", "%"]
    for c in chars: text = text.replace(c, f"{c}")

    return text