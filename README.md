# BioMedGPT
### Project Overview

This project involves fine-tuning the GPT-2 (124M) model on the PubMedQA dataset. The model was implemented from scratch in PyTorch, and pre-trained GPT-2 weights were loaded before fine-tuning. The dataset was formatted into a question-answer format and contains approximately 50 million characters. The fine-tuning process was optimized using various strategies to improve performance and prevent overfitting.

### Dataset

 Source: PubMedQA dataset from Hugging Face.

 Format: Question-Answer pairs.

 Size: ~50 million characters.

 Train/Validation Split: 98:2 ratio.

### Model Implementation

 Base Model: GPT-2 (124M) written from scratch in PyTorch.

 Preprocessing: Used tiktoken GPT-2 tokenizer.

 Pretrained Weights: Loaded GPT-2 (124M) weights before fine-tuning.

### Training Configuration

Batch Size: 16

Block Size: 128

Gradient Accumulation Steps: 32

Epochs: 5

Steps per Epoch: 274

Learning Rate Scheduler:

10 warm-up steps.

Cosine decay to 1% of the initial learning rate.

Maximum learning rate: 1e-5 for slow training.

### Optimization Techniques:

Mixed precision training for memory efficiency on GPU.

Dropout: 0.6.

Weight Decay: 0.3.

Prevented overfitting by tuning dropout and weight decay values.

Used Fused AdamW optimizer for better efficiency.

#### Training Results

Train Loss per Epoch:

Epoch 1: <4.21>
Epoch 2: <4.16>
Epoch 3: <4.15>
Epoch 4: <4.11>
Epoch 5: <4.08>

#### Validation Loss per Epoch:

Epoch 1: <5.51>
Epoch 2: <5.50>
Epoch 3: <5.47>
Epoch 4: <5.51>
Epoch 5: <5.49>

### Model Performance & Observations

The model is not generating perfect outputs and often hallucinates.

Due to the small model size, it struggles with reasoning in biomedical questions.

The available computational resources limited the model's capacity for more complex fine-tuning.

The learning rate scheduler helped in stable convergence.

Mixed precision reduced memory usage without affecting performance.

Proper dropout and weight decay settings prevented overfitting observed in earlier experiments.

Fused AdamW optimizer improved training efficiency.

### Sample Outputs

#### Input:

Question: Is myocardial infarct-sparing effect of adenosine A2A receptor activation due to its action on CD4+ T lymphocytes
?

#### Generated Output:

Answer: In humans, the presence of AD2A2A overexpression in peripheral cells may contribute to the pathogenesis of the disease pathogenesis, suggesting that the absence of elevated levels, characterized by systemic immune cells, may significantly contribute to the pathogenesis of AD2A2A activity. Our results provide the support for an association between AD2A-induced hyperkinocell death of patients with elevated mortality, elevated CD4 expression, and increased hepatocell activation.

#### Expected Output:

 Answer: In humans, the presence of AD2A2A overexpression in peripheral cells may contribute to the myocardial infarct-sparing effect of adenosine A2A receptor activation, suggesting that modulation of CD4+ T lymphocytes plays a critical role. Our observations indicate that elevated AD2A2A activity is associated with a reduction in hyperkinetic inflammatory responses mediated by CD4+ cells, which may lead to improved myocardial preservation. This association supports the notion that the cardioprotective effects observed with adenosine A2A receptor activation are at least partly due to its action on CD4+ T lymphocytes, highlighting a potential therapeutic pathway for reducing infarct-related damage.

### Requirements

To run this project, install the required dependencies:

pip install torch transformers datasets tiktoken

### Running the Training Script

Execute the training script with:

python train.py

### Future Improvements

Experiment with different batch sizes and accumulation steps.

Test additional learning rate schedules.

Evaluate model performance using different metrics.

Explore further fine-tuning on domain-specific medical datasets.

Consider using a larger model for improved reasoning capabilities.

### Acknowledgments

OpenAI for the GPT-2 model.

Hugging Face for providing easy access to datasets and pre-trained models.

PyTorch for enabling efficient model implementation and training.


