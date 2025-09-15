# BioMedGPT
![Alt text](demo-biomed.png)

BioMedGPT is a fine-tuned GPT-2 Medium model built from scratch in PyTorch and tailored for biomedical question answering. The project leverages a large-scale, domain-specific dataset and advanced data augmentation techniques, including paraphrasing of questions using T5. The resulting model is capable of answering complex biomedical queries with improved accuracy and context-awareness.

### Overview
BioMedGPT is designed to address the unique challenges of biomedical question answering by fine-tuning a GPT-2 Medium model on a dataset of approximately 200,000 QA pairs containing around 30 million tokens (as measured by the GPT-2 tokenizer). By incorporating paraphrasing-based data augmentation, the model learns to handle various formulations of biomedical questions while preserving the integrity of the answers.

### Features
- #### Custom Model Architecture:
  The model is implemented from scratch in PyTorch, with weights later loaded from the Hugging Face GPT-2 model, allowing for extensive 
  customization and transparency.

- #### Domain-Specific Fine-Tuning:
  Fine-tuned on a biomedical QA dataset (99:1 train-validation split) containing ~200K pairs and ~30 million tokens.

- #### Data Augmentation with Paraphrasing:
  Utilizes T5 to paraphrase each question once, thereby increasing input variability while maintaining answer accuracy.

- #### Robust Regularization:
  A dropout rate of 0.3 and a weight decay of 0.3 are used during training to reduce overfitting.

- #### Optimized Training Regime:
  The learning rate starts at 1e-6 and is scheduled to decay cosinely to 1e-7. For the second epoch, the minimum learning rate is 
  maintained throughout, as the model is trained for a total of 2 epochs.



### Installation and Setup

- #### Clone the Repository:
  Clone the BioMedGPT repository from GitHub to your local machine.

- #### Environment Setup:
  Create a Python virtual environment and install the required dependencies as listed in the below.
  ```bash
   pip install torch transformers datasets tiktoken huggingface-hub tqdm

- #### Data Access:
  You can run the dataset.py script to get the preprocessed qa dataset from hugging face.

### Data Preprocessing
The data preprocessing pipeline:

- Loads the biomedical QA dataset.

- Applies T5-based paraphrasing to each question (keeping the answers intact).

- Generates both the original and paraphrased QA pairs.

- Saves the formatted output to a text file for efficient loading during fine-tuning.

This step ensures that the model is exposed to multiple ways of asking the same question, improving its robustness and generalization.



### Model Architecture and Fine-Tuning

- #### Model Construction:
  The GPT-2 Medium model is written from scratch in PyTorch. After building the architecture, weights from a pretrained Hugging Face model are loaded into the custom model.

- #### Fine-Tuning Strategy:
  Only the last 12 transformer blocks (out of 24) are trained, while the remaining blocks remain frozen. This strategy accelerates   
  training and helps mitigate overfitting.

- #### Regularization:
  A dropout rate of 0.3 is applied, and weight decay is set to 0.3 to further prevent overfitting.



### Training Details
- #### Batch Size
  I have used a batch size of 16 and a block size of 128 while trainig but i was processing 65536 tokens in a step so i am using 32 gradient accumulation steps.
- #### Learning Rate Schedule:
  The training starts with a learning rate of 1e-6, which decays using a cosine schedule to 1e-7. For the second epoch, the model 
  continues training at the minimum learning rate.

- #### Epochs:
  The model is fine-tuned for 2 epochs.

- #### Performance:
  The training process achieved a training loss of 2.26 and a validation loss of 3.24, indicating good learning and generalization on the     biomedical QA task.



### Evaluation
Model evaluation involves both quantitative and qualitative analysis:

- #### Loss Curve:
  Detailed training loss curves are available at output/loss.png.

- #### Example QA Comparisons:
  Side-by-side comparisons of the model’s answers against ideal answers for various biomedical questions in output/demo.ipynb

- #### Error Analysis:
A discussion on observed hallucinations or inaccuracies.
| **Question** | **Model Answer** | **Ideal Answer** | **Errors in Model Answer** |
|--------------|------------------|------------------|-----------------------------|
| **How does the renin-angiotensin-aldosterone system (RAAS) regulate blood pressure, and what are the implications of ACE inhibitors in its modulation?** | The RAAS increases systolic and diastolic pressures during arterial occlusion. Increased RAAS activity contributes to vascular resistance in hypertensive patients. Vasodilation may decrease RAAS activity, with nitric oxide influencing cytokine release. | RAAS regulates blood pressure by controlling fluid balance and blood vessel constriction. Renin converts angiotensinogen to angiotensin I, which becomes angiotensin II—a vasoconstrictor. ACE inhibitors block this conversion, lowering blood pressure and reducing heart strain. | ✅ Correct: RAAS activity affects blood pressure and is elevated in hypertension.<br>❌ Incorrect: No mention of key components like renin, angiotensin I/II, or aldosterone; introduces unclear concept of "plasma endothelial nitric oxide (VENO)"; doesn't explain how ACE inhibitors work. |
| **What role do cytochrome P450 enzymes play in drug metabolism?** | CYP2D6 contributes to P450 activity in drug metabolism, affecting enzyme activation and degradation. It is linked to p53 depletion and elevated P450s activity in liver and lymph. | Cytochrome P450 enzymes metabolize drugs by oxidizing them, affecting drug efficacy and toxicity. CYP2D6, CYP3A4, and others vary across individuals, influencing drug clearance and interactions. | ✅ Correct: Mentions CYP2D6’s role in drug metabolism.<br>❌ Incorrect: p53 is not related to CYP2D6 function; lymph system involvement is misleading; omits core function of oxidation and individual variation. |
| **How does CRISPR-Cas9 technology enable genome editing?** | CRISPR-Cas9 targets genome regions in immune cells to deliver DNA outside the body. It creates functional immune cells that recognize pathogens, leading to novel therapies. | CRISPR-Cas9 uses a guide RNA to locate a DNA sequence and the Cas9 enzyme to cut it. This enables precise gene edits, allowing corrections, deletions, or insertions for research or therapy. | ✅ Correct: Suggests genome targeting is involved.<br>❌ Incorrect: Misrepresents core mechanism; immune cells and DNA delivery are unrelated to CRISPR-Cas9 function; lacks explanation of guide RNA and Cas9's role in cutting DNA. |



### Kaggle Notebook

For detailed insights into the training process, including loss curves and training details, please refer to the Kaggle Notebook:
[https://www.kaggle.com/code/vaibhav1908/biomedgpt]

This notebook includes comprehensive details on data preprocessing, model training, and evaluation metrics.

### Future Improvements

- #### Enhanced Augmentation:
  Explore paraphrasing both questions and answers or integrating additional biomedical data sources.

- #### Retrieval-Augmented Generation:
  Integrate domain-specific retrieval mechanisms to further reduce hallucinations.

- #### Model Scaling:
  Investigate the impact of fine-tuning more transformer blocks or employing multi-GPU training setups.

- #### Advanced Evaluation Metrics:
  Incorporate metrics like BLEU, ROUGE, or factuality scoring to quantitatively assess answer quality.

### Acknowledgments

OpenAI for the GPT-2 model.

Hugging Face for providing easy access to datasets and pre-trained models.

PyTorch for enabling efficient model implementation and training.

### License
This project is licensed under the MIT License. See the LICENSE file for details.
