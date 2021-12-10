import argparse
import logging
import os
import re

import numpy as np
import torch
from bert_score import score as BERTscore
from nltk.translate.bleu_score import sentence_bleu
from transformers import (BertLMHeadModel, BertTokenizerFast, GPT2LMHeadModel,
                          GPT2Tokenizer)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "bert": (BertLMHeadModel, BertTokenizerFast),
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--test_data_file",
        type=str,
        default=None,
        required=True,
        help="Location of test data file",
    )
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument(
        "--stop_token",
        type=str,
        default=None,
        help="Token at which text generation is stopped",
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
        help="primarily useful for CTRL model; in that case, use 1.2",
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--num_ref", type=int, default=50)
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
    args = parser.parse_args()

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError(
            "the model {} you specified is not supported. You are welcome to add it and open a PR :)"
        )

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    args.length = adjust_length_to_model(
        args.length, max_sequence_length=model.config.max_position_embeddings
    )
    logger.info(args)

    with open(args.test_data_file, encoding="utf-8") as f:
        test_dataset = [
            line
            for line in f.read().splitlines()
            if (len(line) > 0 and not line.isspace())
        ]

    num_ref = args.num_ref
    references = []
    tokenized_references = []
    candidates = []

    for i in range(num_ref):
        tokenized_reference = test_dataset[i].split()[1:-1]  # remove bos and eos tokens
        tokenized_references.append(tokenized_reference)
        reference = " ".join(tokenized_reference)
        references.append(reference)
        prompt = "<BOS> " + tokenized_reference[0]  # include first token of reference
        encoded_prompt = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).to(args.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args.length,
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        for generated_sequence in output_sequences:
            generated_sequence = generated_sequence.tolist()
            # Decode text
            tweet = tokenizer.decode(
                generated_sequence,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            # Remove all text after the stop token
            tweet = tweet[: tweet.find(args.stop_token) if args.stop_token else None]
            candidates.append(tweet)

    bleu_scores = []
    for i in range(args.num_ref):
        tokenized_candidate = candidates[i].split()
        bleu = sentence_bleu(
            tokenized_references, tokenized_candidate
        )
        bleu_scores.append(bleu)

    bertscore_precision, bertscore_recall, bertscore_f1 = BERTscore(
        candidates, references, lang="en"
    )
    bleu_score = np.mean(np.array(bleu_scores))

    results = {
        "bleu_score": bleu_score,
        "bertscore_precision": bertscore_precision.mean().item(),
        "bertscore_recall": bertscore_recall.mean().item(),
        "bertscore_f1": bertscore_f1.mean().item(),
    }

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(candidates)
    print(references)

    model_name = re.sub(r"/", "_", args.model_name_or_path).lower()
    output_metrics_file = os.path.join(
        args.output_dir, f"metrics_{args.model_type}_{model_name}.txt"
    )
    with open(output_metrics_file, "w") as writer:
        logger.info("***** Metrics results *****")
        for key in sorted(results.keys()):
            logger.info("    %s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))


if __name__ == "__main__":
    main()
