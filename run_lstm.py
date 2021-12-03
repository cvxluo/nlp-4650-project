import argparse
import logging
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchtext.data.metrics import bleu_score
from torchtext.vocab import GloVe

from dataset import *

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--username",
    default=None,
    type=str,
    required=True,
    help="Account name to train model for",
)
parser.add_argument("--length", type=int, default=50)
parser.add_argument(
    "--hidden_size", type=int, default=256, help="Size of hidden layer for LSTM"
)
parser.add_argument(
    "--embedding_dim",
    type=int,
    default=300,
    help="Size of embedding dimension. NOTE: This is overwritten when using pretrained embeddings.",
)
parser.add_argument("--n_layers", type=int, default=1, help="Number of LSTM layers")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument(
    "--num_epochs", type=int, default=5, help="Total number of train epochs"
)
parser.add_argument("--dropout", type=float, default=0.2, help="LSTM dropout rate")
parser.add_argument(
    "--use_pretrained_embeddings",
    type=bool,
    default=False,
    help="Whether to use twitter GloVe embeddings or not",
)
parser.add_argument(
    "--eval_every",
    type=int,
    default=1000,
    help="Number of iterations between eval metric outputs for LSTM model",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="outputs",
    help="Output dir of metrics results",
)
parser.add_argument(
    "--seed", type=int, default=42, help="random seed for initialization"
)
parser.add_argument(
    "--no_cuda", action="store_true", help="Avoid using CUDA when available"
)
parser.add_argument(
    "--bos_token", type=str, default="<BOS>", help="Beginning of sentence token"
)
parser.add_argument(
    "--eos_token", type=str, default="<EOS>", help="End of sentence token"
)
parser.add_argument(
    "--train_split",
    type=float,
    default=0.8,
    help="Train split for dataset",
)
args = parser.parse_args()
args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

bos_token = args.bos_token
eos_token = args.eos_token


class TextGenerator(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        embedding_dim,
        hidden_size,
        dropout=0.2,
        n_layers=1,
    ):
        super(TextGenerator, self).__init__()
        self.num_layers = n_layers
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.encoder = nn.Embedding(self.input_size, self.embedding_dim)

        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_size,
            self.num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        input = self.encoder(input)
        output, hidden = self.lstm(input)
        output = self.decoder(output)
        return output, hidden


class TwitterDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Tokenizer,
        add_special_tokens=True,
    ) -> None:
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.bos_token = bos_token
        self.eos_token = eos_token

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self.df.iloc[index]
        tweet = sample["text"]
        if self.add_special_tokens:
            tweet = bos_token + " " + tweet + " " + eos_token
        tweet_indices = self.tokenizer.encode(tweet).ids
        tweet_tensor = torch.tensor(tweet_indices, dtype=torch.long)
        return tweet_tensor


@dataclass
class DataCollatorWithPadding:
    tokenizer: Tokenizer

    def __call__(self, batch):
        padding_value = self.tokenizer.token_to_id(eos_token)
        padded_batch = pad_sequence(batch, padding_value=padding_value)
        padded_batch = torch.transpose(padded_batch, 0, 1)
        return padded_batch


def train_tokenizer(username: str, output_dir="model", unk_token="[UNK]") -> Tokenizer:
    tokenizer = Tokenizer(models.WordLevel(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    special_tokens = [bos_token, eos_token, unk_token]
    trainer = trainers.WordLevelTrainer(
        special_tokens=special_tokens, show_progress=True
    )
    texts = load_dataset(username)["text"].tolist()
    tokenizer.train_from_iterator(texts, trainer=trainer)
    output_file = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(output_file)
    return tokenizer


def train(
    model,
    tokenizer,
    train_iterator,
    val_iterator,
    num_epochs=1,
    lr=0.001,
    eval_every=1000,
):
    vocab_size = tokenizer.get_vocab_size()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    model.zero_grad()

    candidates = []
    references = []

    losses = []
    bleu_scores = []
    perplexities = []

    total_iter = 0

    for e in range(num_epochs):
        curr_iter = 0
        total_loss = 0

        for tweet in train_iterator:
            tweet = tweet.to(device)
            input = tweet[:, :-1]  # exclude EOS token
            target = tweet[:, 1:]  # exclude BOS token

            # cleanup
            optimizer.zero_grad()
            # forward pass
            output, hidden = model(input)
            loss = criterion(output.reshape(-1, vocab_size), target.reshape(-1))

            # backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            curr_iter += 1
            total_iter += 1

            if total_iter % eval_every == 0:
                """
                # BLEU score calculation
                candidates = [sample_sequence(model, vocab, temperature=0.6).split()] # for i in range(3)]
                reference_corp = list(map(lambda ids : index_to_sentence(ids.squeeze(), vocab_itos), iterator))
                references = [reference_corp] # for i in range(3)]
                bleu = bleu_score(candidate_corpus=candidates, references_corpus=references, max_n=4)
                """
                bleu = 0
                train_loss = total_loss / curr_iter
                eval_loss, eval_perplexity = evaluate(model, tokenizer, val_iterator)
                logger.info(
                    "[Iter %d] Loss %.2f  Val Loss: %.2f  Val Perplexity: %.2f"
                    % (
                        total_iter,
                        train_loss,
                        eval_loss,
                        eval_perplexity,
                    )
                )
                losses.append(train_loss)
                perplexities.append(eval_perplexity)
                model.train()

    return losses, perplexities, bleu_scores


def evaluate(
    model,
    tokenizer,
    val_iterator,
):
    vocab_size = tokenizer.get_vocab_size()
    criterion = nn.CrossEntropyLoss()
    model.eval()

    candidates = []
    references = []

    total_loss = 0

    with torch.no_grad():
        for tweet in val_iterator:
            tweet = tweet.to(device)
            input = tweet[:, :-1]
            target = tweet[:, 1:]

            # cleanup
            # forward pass
            output, hidden = model(input)
            loss = criterion(output.reshape(-1, vocab_size), target.reshape(-1))

            total_loss += loss.item()

    avg_loss = total_loss / len(val_iterator)
    avg_perplexity = math.exp(avg_loss)

    return avg_loss, avg_perplexity


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


device = torch.device(
    "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
)

set_seed(args)

output_dir = os.path.join("models", args.username)
df = load_dataset(args.username)
tokenizer = train_tokenizer(args.username, output_dir=output_dir)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
vocab_size = tokenizer.get_vocab_size()
train_df, val_df = split_train_val(df, split=[args.train_split, 1.0 - args.train_split])
train_dataset = TwitterDataset(train_df, tokenizer)
val_dataset = TwitterDataset(val_df, tokenizer)

hidden_size = args.hidden_size
embedding_dim = args.embedding_dim
n_layers = args.n_layers
batch_size = 1  # this is fixed
lr = args.lr
num_epochs = args.num_epochs
dropout = args.dropout
use_pretrained_embeddings = args.use_pretrained_embeddings
# Note that if we use pretrained embeddings, our embedding_dim parameter will be ignored

train_iterator = DataLoader(
    train_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True
)
val_iterator = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)
model = TextGenerator(
    vocab_size,
    vocab_size,
    embedding_dim,
    hidden_size,
    n_layers=n_layers,
    dropout=dropout,
).to(device)

print(args)

if use_pretrained_embeddings:
    logger.info("Using pretrained twitter GloVe embeddings...")
    embedding = GloVe(name="twitter.27B", dim=200)
    pretrained_embeddings = embedding.vectors
    input_size, embedding_dim = pretrained_embeddings.shape
    model = TextGenerator(
        input_size,
        vocab_size,
        embedding_dim,
        hidden_size,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)
    model.encoder.weight.data.copy_(pretrained_embeddings)
    model.encoder.weight.requires_grad = False
else:
    model.encoder.weight.requires_grad = True

avg_losses, bleu_scores, perplexities = train(
    model,
    tokenizer,
    train_iterator,
    val_iterator,
    num_epochs=num_epochs,
    lr=lr,
    eval_every=args.eval_every,
)
model_file = os.path.join(output_dir, "pytorch_model.bin")
torch.save(model, model_file)

plt.plot(avg_losses)
plt.plot(perplexities)

eval_loss, eval_perplexity = evaluate(model, tokenizer, val_iterator)
result = {"eval_loss": eval_loss, "eval_perplexity": eval_perplexity}

output_eval_file = os.path.join(output_dir, "eval_results_lstm.txt")
with open(output_eval_file, "w+") as writer:
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
