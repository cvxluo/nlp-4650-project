import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer

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
    "--num_sequences", type=int, default=10, help="Number of sequences to generate"
)
parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
)
parser.add_argument(
    "--repetition_penalty",
    type=float,
    default=1.0,
    help="puts a penalty on repeating tokens",
)
parser.add_argument("--k", type=int, default=50)
parser.add_argument("--p", type=float, default=0.9)
parser.add_argument(
    "--model_dir",
    type=str,
    default="models",
    help="Output dir of model",
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


def load_tokenizer(username: str, model_dir="models"):
    output_file = os.path.join(model_dir, username, "tokenizer.json")
    tokenizer = Tokenizer.from_file(output_file)
    return tokenizer


def clean_up_tokenization(out_string: str) -> str:
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" '", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string


def sample_sequence(
    model,
    tokenizer: Tokenizer,
    input_ids: torch.LongTensor,
    max_length=50,
    temperature=1.0,
    repetition_penalty=1.0,
    top_k=50,
    top_p=0.9,
):
    eos_token_id = tokenizer.token_to_id(eos_token)
    min_tokens_to_keep = 1
    filter_value = -float("Inf")

    while True:
        output, hidden = model(input_ids)
        next_token_logits = output[:, -1, :]

        scores = next_token_logits

        if repetition_penalty is not None and repetition_penalty != 1.0:
            score = torch.gather(scores, 1, input_ids)
            score = torch.where(
                score < 0, score * repetition_penalty, score / repetition_penalty
            )
            scores = scores.scatter_(1, input_ids, score)

        if temperature is not None and temperature != 1.0:
            scores = scores / temperature

        if top_k is not None and top_k != 0:
            top_k = min(max(top_k, min_tokens_to_keep), scores.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
            scores = scores.masked_fill(indices_to_remove, filter_value)

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scores, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            scores = scores.masked_fill(indices_to_remove, filter_value)

        probs = nn.functional.softmax(scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        next_token = next_tokens.squeeze().item()

        if next_token == eos_token_id or input_ids.shape[-1] >= max_length:
            break

    return input_ids


def generate_sequences(
    model,
    tokenizer: Tokenizer,
    prompt=None,
    max_length=50,
    num_sequences=5,
    temperature=1.0,
    repetition_penalty=1.0,
    k=50,
    p=0.9,
):
    if prompt is None:
        prompt = input("Prompt: ")
        prompt = bos_token + " " + prompt

    generated_tweets = []

    input_ids = torch.tensor(
        tokenizer.encode(prompt).ids, dtype=torch.long, device=device
    ).unsqueeze(0)

    for _ in range(num_sequences):
        sequence = sample_sequence(
            model,
            tokenizer,
            input_ids,
            max_length=max_length,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_k=k,
            top_p=p,
        ).squeeze(0)
        sequence = sequence.tolist()
        tweet = tokenizer.decode(sequence)
        tweet = clean_up_tokenization(tweet)
        generated_tweets.append(tweet)

    if len(generated_tweets) == 1:
        return generated_tweets[0]

    return generated_tweets


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


device = torch.device(
    "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
)

set_seed(args)

tokenizer = load_tokenizer(args.username, model_dir=args.model_dir)
model_file = os.path.join(args.model_dir, args.username, "pytorch_model.bin")
model = torch.load(model_file)

tweets = generate_sequences(
    model,
    tokenizer,
    num_sequences=args.num_sequences,
    prompt=bos_token,
    temperature=args.temperature,
    k=args.k,
    p=args.p,
)
for i, tweet in enumerate(tweets):
    logger.info(f"Sample {i}: {tweet}")
