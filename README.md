# TestAug - A Framework for Augmenting Capability-based NLP Tests


## Pipeline

The following will demonstrate how TestAug is applied to testing a sentiment classifier's ability to handle negation in a sentence. Before going through each step, make sure that

1. The [computation environment](#environment) is properly set up.

2. You have a valid access key to GPT-3 and it could be accessed from `~/.bashrc` as something look like below. You could still experience our system **without** this key by going to Step 5 directly as a sample of queried dataset is provided.

   ```bash
   export OPENAI_API_KEY="XYZ"
   ```

Now we are ready to use the TestAug system:

- Step 0: Clone the repository and make sure the paths in the `setting/setting.py` are set correctly.

- Step 1: Query and annotate the sentences based on seed sentences from the CheckList test suite. 

  - The `--description 3` specifies one concrete case of negation in the CheckList test suite:

  	> A negative sentiment sentence with negated positive sentiment word and neutral contents in the middle.

  - The queried and annotated sentences will appear in the following directories:
    - Unlabeled Data: `dataset/unlabeled/003/unlabeled@<timestamp>.pkl`
    - Labeled Data: `dataset/labeling/003/<timestamp>.json`.

  ```bash
  python pipeline/01_annotate.py --query --annotate --task sentiment --description 3
  ```

- Step 2: Inspect the labeled data in `data/labeling/003/` to see if the ratio of valid sentences $\tau$ is above the predefined threshold (in our project, this threshold is set to 90%). 

  - If $\tau \geq 90\%$, directly proceed to Step 4 without stepping into Step 3.
  - if $\tau < 90\%$, repeat Step 1 as following, where `--phase2` will query a training set to train a `roberta-base` classifier to filter out the invalid sentences. Proceed to Step 3 after this step.

  In our case, we do need to go through this additional step as the ratio is below 90%.

  ```bash
  python pipeline/01_annotate.py --query --annotate --task sentiment --description 3 --phase2
  ```

- Step 3: Train a `roberta-base` classifier to automatically filter out the invalid sentences. 

  ```bash
  # prepare training data
  python pipeline/02_prepare_data.py --task sentiment --description 3 --save
  
  # train the classifier
  python pipeline/03_classify.py --task sentiment --description 3 --train --test
  ```

- Step 4: Query a set of test cases and filter invalid ones out if necessary

  ```bash
  python pipeline/04_query.py --task sentiment --description 3
  ```
  
- Step 5: Filter out the invalid cases if the ratio of valid cases exceeds predefined threshold (Step 2).
	
  ```bash
  python pipeline/05_filter.py --task sentiment
  ```
  
- Step 6: Test the classifiers in question with the augmented test suite provided by TestAug. Note that `--seed 42` here is related to the test suites evaluation mentioned in the paper; it could be set to a different value for a more complete comparison.
	
  ```bash
  python pipeline/06_test.py --task sentiment --seed 42
  ```
  
  By default, the TestAug tests the following 4 models; all of them have shown [decent accuracies](https://textattack.readthedocs.io/en/latest/3recipes/models.html) on the original validation set. Additional models could be easily incorporated if they appear on the [HuggingFace model hub](https://huggingface.co/models).
  
  | Model                                    | Validation (Test) Accuracy |
  | ---------------------------------------- | -------------------------- |
  | `textattack/distilbert-base-cased-SST-2` | 90.02%                     |
  | `textattack/albert-base-v2-SST-2`        | 92.55%                     |
  | `textattack/bert-base-uncased-SST-2`     | 92.43%                     |
  | `textattack/roberta-base-SST-2`          | 94.04%                     |

- Step 6: Compare the model's error rates on different test suites.

  ```bash
  python pipeline/07_report.py --setting aggregate --task sentiment
  ```

## Reproducing Experiments

With all three tasks' test suites available, the Table 3 could be reproduced following the steps below.

- Step 1: Run the `reproduce.sh` script below:

  ```bash
  bash reproduce.sh
  ```

- Step 2: Reproduce Table 3 of the paper:

  ```bash
  python pipeline/07_report.py --task sentiment --table
  ```
  

## Environment

It is recommended to set up our system using a computing platform with GPU support.

```bash
conda create --name testaug python==3.8.0
conda activate testaug

# generic libraries
conda install numpy pandas matplotlib seaborn scikit-learn ipython tqdm termcolor
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch 

# libraries related to NLP models
pip install transformers datasets
pip install simpletransformers

pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm

# libraries related to capability-based testing
pip install checklist

# library related to querying GPT-3
pip install openai
pip install retelimit

# library related to evaluation metrics
pip install networkx
pip install fast_bleu

# current project
pip install -e .
```

