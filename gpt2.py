import logging
import os
import time
import datetime
from typing import Dict

import torch
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer, GPT2LMHeadModel,  GPT2Tokenizer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

    
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


class TweetDataset(Dataset):
    """Dataset to hold tweets from an account"""

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int = 768):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        self.tokenizer = tokenizer
        self.examples = []
        self.attention_masks = []

        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            tweets = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        for tweet in tweets:
            batch_encoding = tokenizer(tweet, truncation=True, max_length=block_size, padding="max_length")
            self.attention_masks.append(torch.tensor(batch_encoding['attention_mask']))
            self.examples.append(torch.tensor(batch_encoding["input_ids"]))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i], self.attention_masks[i]


def train(model, device, optimizer, train_iterator, val_iterator, tokenizer,
          epochs=5, sample_every=100, n_samples=1):
    """Train the GPT-2 by fine-tuning the pretrained LM weights"""

    total_t0 = time.time()
    training_stats = []

    for epoch_i in range(0, epochs):
        # ========================================
        #               Training
        # ========================================

        logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        logger.info('Training...')

        t0 = time.time()
        total_train_loss = 0
        generated_samples = []
        model.train()

        for step, batch in enumerate(train_iterator):
            input_ids = batch[0].to(device)
            labels = batch[0].to(device)
            attention_mask = batch[1].to(device)

            model.zero_grad()        

            outputs = model(
                input_ids,
                labels=labels, 
                attention_mask=attention_mask,
                token_type_ids=None
            )

            loss = outputs[0]
            batch_loss = loss.item()
            total_train_loss += batch_loss

            # Get sample every x batches.
            if step % sample_every == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                logger.info("Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.".format(step, len(train_iterator), batch_loss, elapsed))

                model.eval()
                samples = generate(model, device, tokenizer, n_sequences=n_samples)
                generated_samples.append(samples)
                model.train()

            loss.backward()
            optimizer.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_iterator)       
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        logger.info("Average training loss: {0:.2f}".format(avg_train_loss))
        logger.info("Training epoch took: {:}".format(training_time))
            
        # ========================================
        #               Validation
        # ========================================

        logger.info("Running Validation...")

        t0 = time.time()
        model.eval()
        total_eval_loss = 0

        # Evaluate data for one epoch
        for batch in val_iterator:
            
            input_ids = batch[0].to(device)
            labels = batch[0].to(device)
            attention_mask = batch[1].to(device)
            
            with torch.no_grad():        
                outputs  = model(input_ids, 
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs[0]  
                
            batch_loss = loss.item()
            total_eval_loss += batch_loss        

        avg_val_loss = total_eval_loss / len(val_iterator)
        validation_time = format_time(time.time() - t0)    

        logger.info("Validation Loss: {0:.2f}".format(avg_val_loss))
        logger.info("Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time,
                'Samples': generated_samples,
            }
        )

    logger.info("Training complete!")
    logger.info("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
    return training_stats


def generate(model, device, tokenizer, prompt="<BOS>", temperature=.85, top_k=50, log_output=True,
             top_p=0.95, max_length=20, n_sequences=1, repetition_penalty=1.5):
    """Generate sequences from the fine-tuned GPT-2 model"""
    model.eval()

    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    input_ids = input_ids.to(device)

    outputs = model.generate(
        input_ids,
        top_k=top_k, 
        max_length=max_length,
        top_p=top_p, 
        num_return_sequences=n_sequences,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        early_stopping=True,
        do_sample=True,
    )
    decoded_outputs = []

    for i, output in enumerate(outputs):
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        decoded_outputs.append(decoded_output)
        if log_output:
            logger.info("Sample {}: {}".format(i, decoded_output))
    
    return decoded_outputs


def print_model_details(model):
    """Print out the details of a model"""
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    logger.info('The GPT-2 model has {:} different named parameters.'.format(len(params)))
    logger.info('==== Embedding Layer ====')

    for p in params[0:2]:
        logger.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    logger.info('==== First Transformer ====')

    for p in params[2:14]:
        logger.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    logger.info('==== Output Layer ====')

    for p in params[-2:]:
        logger.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


def save_model(username: str, model, tokenizer, output_dir="models"):
    """Saves the model to models/<username>. The model can then be loaded using from_pretrained method."""
    output_dir = os.path.join(output_dir, username)

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f"Saving model to {output_dir}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Done saving. Model is located at {output_dir}.")


def load_model(username: str, models_dir="models", print_details=True):
    """Loads a pretrained model from local models"""
    model_dir = os.path.join(models_dir, username)

    logger.info(f"Loading model from {models_dir}...")

    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

    logger.info(f"Loaded fine-tuned GPT-2 model for {username}")

    if print_details:
        print_model_details(model)

    return model, tokenizer
