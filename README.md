# Natural Language Generation for Tweets

## Environment Setup

To setup the conda environment run the following commands:

```
conda env create -f env.yml
conda activate nlp-4650-project
```

To download the necessary python packages, run the following command:

```
pip install -r requirements.txt
```

## Data

The data for our project is located under the `data` directory. We have included the scraped CSV files along with the train, test, and validation splits with cleaned data for each account.

## Models

We do not provide the trained models in the code in our GitHub repo. These models are very large and could not fit. As such, we have included multiple releases with the different models attached as a downloadable artifact. To see these, please navigate to the releases section of our repo and download the models into the `models` directory.

## Training

We have provided scripts to train our models on our dataset. The results from training can be found the following notebooks:

- GPT_2.ipynb
- bert2bert.ipynb
- lstm.pynb

If you would prefer to train the models yourself, please keep reading. All credit goes to the team over at [HuggingFace](https://huggingface.co/) for the source code for `run_language_modeling.py`, `run_mlm.py` and `run_generation.py`. You can find those [here](https://github.com/huggingface/transformers/tree/master/examples).

### GPT-2

```
!python run_language_modeling.py \
  --output_dir=models/<username> \
  --train_data_file=data/<username>/train.txt \
  --eval_data_file=data/<username>/valid.txt \
  --line_by_line \
  --model_type=gpt2 \
  --model_name_or_path=gpt2 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size=2 \
  --learning_rate=5e-5 \
  --num_train_epochs=5
```

We have also provided a hyper-parameter search script.

```
!python run_hp_search.py \
  --output_dir=hp_search_results/<username> \
  --train_data_file=data/<username>/train.txt \
  --eval_data_file=data/<username>/valid.txt \
  --line_by_line
  --model_type=gpt2 \
  --model_name_or_path=gpt2 \
  --do_train \
  --do_eval \
  --disable_tqdm=True \
```

### BERT

```
!python run_mlm.py \
  --output_dir=models/<username> \
  --train_file=data/<username>/train.txt \
  --validation_file=data/<username>/valid.txt \
  --line_by_line \
  --model_type=bert \
  --model_name_or_path=bert-base-cased \
  --do_train \
  --do_eval \
  --per_device_train_batch_size=2 \
  --per_device_eval_batch_size=2 \
  --learning_rate=5e-5 \
  --num_train_epochs=5
```

Similarly, the hyper-parameter search script will also work with this model.

```
!python run_hp_search.py \
  --output_dir=hp_search_results/<username> \
  --train_data_file=data/<username>/train.txt \
  --eval_data_file=data/<username>/valid.txt \
  --line_by_line
  --model_type=bert \
  --model_name_or_path=bert-base-cased \
  --do_train \
  --do_eval \
  --disable_tqdm=True \
```

### LSTM

```
!python run_lstm.py \
  --username <username> \
  --num_epochs 7 \
  --eval_every 2000 \
  --hidden_size 512 \
  --embedding_dim 300 \
  --lr 0.0004
```

### LSTM with GloVe Embeddings

```
!python run_lstm.py \
  --username <username> \
  --use_pretrained_embeddings True \
  --num_epochs 7 \
  --eval_every 2000 \
  --hidden_size 512 \
  --lr 0.0004
```

## Generation and Metrics

We have also provided scripts to generate and compute metrics for each model. To generate and compute metrics for a model you must first have a saved version in the `models` directory.

### GPT-2 and BERT

To generate sequences for a given account, run the following script.

```
!python run_generation.py \
  --model_type gpt2 \
  --model_name_or_path "models/<username>" \
  --length 50 \
  --prompt "<BOS>" \
  --stop_token "<EOS>" \
  --k 50 \
  --num_return_sequences 5
```

To compute the metrics, run the following script.

```
!python run_metrics.py \
  --model_type gpt2 \
  --model_name_or_path "models/<username>" \
  --test_data_file "data/<username>/test.txt" \
  --length 50 \
  --stop_token "<EOS>" \
  --k 50 \
  --num_ref 50
```

### LSTM

To generate sequences for a given account, run the following script.

```
!python run_lstm_generation.py \
  --username <username> \
  --num_sequences 10 \
  --temperature 0.8 \
  --k 50 \
  --p 0.9 \
  --seed 52
```

To compute the metrics, run the following script. Adding `with_glove` will change the name of output file in the `outputs` directory.

```
!python run_lstm_metrics.py \
  --username <username> \
  --test_data_file data/<username>/test.txt \
  --temperature 0.8 \
  --k 50 \
  --p 0.9 \
  --with_glove True
```
