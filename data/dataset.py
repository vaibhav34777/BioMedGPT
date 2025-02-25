import pickle
import os
from datasets import load_dataset

CACHE_FILE = "pubmed_qa_cache.pkl"

# Load dataset from cache or fetch it if not cached
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        dataset = pickle.load(f)
else:
    dataset = load_dataset("pubmed_qa", "pqa_artificial")  # loading the pubmedqa dataset
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(dataset, f)

# Formatting the dataset
def format_qa(example):  
    return {
        "formatted": f"Question: {example['question']}\nAnswer: {example['long_answer']}\n<|endoftext|>\n"
    }

formatted_dataset = dataset["train"].map(format_qa)

# Save directly to a text file
with open("pubmedqa.txt", "w", encoding="utf-8") as f:
    f.writelines(formatted_dataset["formatted"])


