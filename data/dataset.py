import os
import pickle
import re
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2Tokenizer
import torch
from google.colab import drive

# Paths for caching and output on Drive
CACHE_FILE = "/content/drive/My Drive/pubmed_qa_cache.pkl"
OUTPUT_FILE = "/content/drive/My Drive/pubmedqa.txt"
dataset = load_dataset("pubmed_qa", "pqa_artificial")
with open(CACHE_FILE, "wb") as f:
    pickle.dump(dataset, f)

# Initialize T5 model and tokenizer for paraphrasing and GPT-2 for tokenization
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
device = "cuda" if torch.cuda.is_available() else "cpu"
t5_model.to(device)

def paraphrase_batch(examples, max_length=128):
    """
    Processes a batch of examples:
      - For each question in the batch, run T5 in batch mode to paraphrase it.
      - Then format the QA pair (original + paraphrased version) into a single string.
    """
    # Extract questions and long answers from the batch
    questions = examples["question"]
    long_answers = examples["long_answer"]
    
    # Create input texts for T5
    inputs = [f"paraphrase: {q}" for q in questions]
    encoded_inputs = t5_tokenizer.batch_encode_plus(
        inputs,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
    
    # Generate paraphrases in batch
    outputs = t5_model.generate(
        input_ids=encoded_inputs["input_ids"],
        attention_mask=encoded_inputs["attention_mask"],
        max_length=max_length,
        num_return_sequences=1,
        early_stopping=True
    )
    # Decode the outputs
    paraphrased_questions = [t5_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    # Build the formatted text for each QA pair
    formatted = []
    for orig_q, para_q, ans in zip(questions, paraphrased_questions, long_answers):
        original_text = f"Question: {orig_q}\nAnswer: {ans}\n<|endoftext|>\n"
        paraphrased_text = f"Question: {para_q}\nAnswer: {ans}\n<|endoftext|>\n"
        formatted.append(original_text + paraphrased_text)
    
    return {"formatted": formatted}
# Use batched mapping to process the dataset
# Note: set num_proc=1 when using GPU to avoid CUDA multiprocessing issues.
formatted_dataset = dataset["train"].map(
    paraphrase_batch, 
    batched=True, 
    batch_size=64, 
    num_proc=1
)
# Save directly to a text file on Drive
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(formatted_dataset["formatted"])
