import logging
import math
import os

import torch
from transformers import (DataCollatorForLanguageModeling, GPT2Config,
                          GPT2LMHeadModel, GPT2TokenizerFast,
                          LineByLineTextDataset, Trainer, TrainingArguments,
                          set_seed)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    train_data_file,
    eval_data_file,
    epochs=5,
    learning_rate=5e-5,
    seed=42,
    batch_size=2,
    output_dir="model",
    model_path=None,
    **kwargs,
):
    """Train the GPT-2 by fine-tuning the pretrained LM weights"""
    set_seed(seed)

    training_args = TrainingArguments(
        num_train_epochs=epochs,
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        **kwargs,
    )

    config = GPT2Config.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = config.eos_token_id

    block_size = tokenizer.model_max_length

    train_dataset = LineByLineTextDataset(tokenizer, train_data_file, block_size)
    eval_dataset = LineByLineTextDataset(tokenizer, eval_data_file, block_size)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Training
    special_tokens_dict = {
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>",
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    train_output = trainer.train(model_path=model_path)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print(train_output)

    # Evaluation
    results = {}
    eval_output = trainer.evaluate()

    perplexity = math.exp(eval_output["eval_loss"])
    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    results.update(result)
    return model, results


def hp_search(
    train_data_file,
    eval_data_file,
    n_trials=10,
    output_dir="model",
    hp_space=None,
):
    config = GPT2Config.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    train_dataset = LineByLineTextDataset(
        tokenizer, train_data_file, tokenizer.model_max_length
    )
    eval_dataset = LineByLineTextDataset(
        tokenizer, eval_data_file, tokenizer.model_max_length
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = config.eos_token_id

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        disable_tqdm=True,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
    )

    def model_init():
        return GPT2LMHeadModel.from_pretrained(
            "gpt2",
            config=config,
        )

    trainer = Trainer(
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_init=model_init,
    )

    def hp_space_init(trial):
        from ray import tune

        return (
            {
                "learning_rate": tune.loguniform(5e-6, 5e-4),
                "per_device_train_batch_size": tune.grid_search([2, 4]),
                "gradient_accumulation_steps": tune.choice([1, 2]),
            }
            if hp_space is None
            else hp_space
        )

    return trainer.hyperparameter_search(
        hp_space=hp_space_init, n_trials=n_trials, backend="ray"
    )


def generate(
    username: str,
    prompt="",
    stop_token="<EOS>",
    temperature=1.0,
    top_k=50,
    log_output=False,
    top_p=0.95,
    min_length=10,
    max_length=80,
    n_sequences=1,
    repetition_penalty=1.0,
):
    """Generate sequences from the fine-tuned GPT-2 model"""
    model_path = f"models/{username}"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)

    if isinstance(prompt, torch.Tensor):
        input_ids = prompt.to(device)
    elif isinstance(prompt, str) and tokenizer is not None:
        input_ids = tokenizer.encode(
            prompt, add_special_tokens=True, return_tensors="pt"
        )
        input_ids = input_ids.to(device)
    else:
        raise ValueError("Invalid prompt input")

    outputs = model.generate(
        input_ids,
        top_k=top_k,
        min_length=min_length,
        max_length=max_length,
        top_p=top_p,
        num_return_sequences=n_sequences,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        do_sample=True,
    )

    if len(outputs.shape) > 2:
        outputs.squeeze_()

    generated_tweets = []
    for i, generated_sequence in enumerate(outputs):
        generated_sequence = generated_sequence.tolist()
        tweet = tokenizer.decode(
            generated_sequence,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        tweet = tweet[: tweet.find(stop_token) if stop_token else None]
        generated_tweets.append(tweet)
        if log_output:
            logger.info("Sample {}: {}".format(i, tweet))

    if len(generated_tweets) == 1:
        generated_tweets = generated_tweets[0]

    return generated_tweets


def load_model(username: str, models_dir="models", verbose=False):
    """Loads a pretrained model from local models"""
    model_dir = os.path.join(models_dir, username)

    if verbose:
        logger.info(f"Loading model from {models_dir}...")

    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)

    if verbose:
        logger.info(f"Loaded fine-tuned GPT-2 model for {username}")

    return model, tokenizer
