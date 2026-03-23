# LLM Product Classification

This project investigates the use of **Large Language Models (LLMs)** for multi-class product classification in a real-world retail scenario. It focuses on comparing two modern approaches in Natural Language Processing:

- **Fine-tuning** of pre-trained Transformer models  
- **Zero-shot inference** using instruction-based LLMs  

The goal is to evaluate their performance under realistic constraints such as noisy data, class imbalance, and limited computational resources.

---

## Overview

Automatic product classification is a fundamental task in domains such as e-commerce, inventory management, and retail analytics. Accurate categorization enables better search, recommendation systems, and business intelligence.

Recent advances in NLP, particularly Transformer-based architectures, have significantly improved performance in text classification tasks. This project explores how these models behave in a **realistic and challenging environment**, rather than in idealized benchmark datasets.

Two main approaches are studied:

- **Fine-tuning**: adapting pre-trained models (e.g., BERT, RoBERTa, BETO) to a specific classification task using labeled data  
- **Zero-shot learning**: leveraging general-purpose LLMs (e.g., Mistral, Qwen, Codestral) without additional training, using prompt-based inference  

---

## Dataset

The dataset consists of **real-world product descriptions** collected from Spanish supermarket catalogs. It presents several challenges typical of production environments:

- Noisy and unstructured text (abbreviations, inconsistencies, formatting issues)  
- Hierarchical and domain-specific labels  
- Strong class imbalance (frequent vs rare categories)  
- Semantic ambiguity between similar products  

These characteristics make it a realistic benchmark for evaluating model robustness and generalization.

---

## Methodology

### Fine-tuning

Fine-tuning experiments were conducted using the Hugging Face Transformers ecosystem.

- Models: BERT, BETO, RoBERTa, DeBERTa, DistilBERT, among others  
- Training strategy: supervised learning with labeled data  
- Evaluation: 4-fold cross-validation  
- Metrics:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score (macro and weighted)  

This approach allows models to specialize in the domain and learn task-specific patterns.

---

## Computational Constraints

This project was developed entirely in local environments without access to high-performance servers or advanced GPUs.

Model selection and training configuration were adapted to these constraints:

- Medium-sized models (e.g., RoBERTa, DeBERTa, XLM-R)
- Limited number of training epochs to ensure convergence and avoid overfitting
- Small batch sizes and efficient optimization strategies

For zero-shot evaluation, quantized LLMs (GGUF format) were executed locally using LM Studio, enabling inference on CPU without requiring advanced infrastructure.

This setup reflects realistic scenarios such as small companies, academic environments, or individual projects, demonstrating that competitive results can be achieved with modest computational resources.

---

### Zero-shot Learning

Zero-shot evaluation was performed using locally deployed LLMs through LM Studio.

- Models: Mistral, Qwen, Codestral, and other GGUF-based models  
- No additional training required  
- Prompt engineering used to guide classification  
- Label space divided into subsets to reduce context overload  

This approach evaluates the model's ability to generalize without task-specific training.

---

## Results

| Approach     | Performance |
|--------------|------------|
| Fine-tuning  | >90% Accuracy and F1-score |
| Zero-shot    | Significantly lower performance |

Key observations:

- Fine-tuned models consistently outperform zero-shot approaches  
- Zero-shot performance degrades with:
  - increasing number of classes  
  - rare or ambiguous labels  
- Prompt design has a strong impact on zero-shot results  

---

## Key Insights

- Fine-tuning is highly effective for domain adaptation  
- Zero-shot approaches are flexible but less reliable in specialized tasks  
- Class imbalance significantly affects model performance  
- Larger models do not always guarantee better results  
- Running LLMs locally is feasible, even with limited resources  

---

## Repository Structure

llm-product-classification/
│
├── data/ # Dataset or instructions to obtain it
├── notebooks/ # Exploratory analysis and experiments
├── src/ # Core code (training, evaluation, inference)
├── results/ # Tables, figures, and evaluation outputs
├── configs/ # Experiment configurations
├── docs/ # Project documentation and report
└── README.md


---

## Environment

This project was developed using Python and common NLP libraries such as:

- transformers  
- torch  
- pandas  
- scikit-learn  
- numpy  

A `requirements.txt` file is included for reference.

---

## Notes

- This repository is intended for **research and demonstration purposes**.  
- The code may require adaptation depending on the dataset, model, and environment.  
- Models and datasets are not included due to size and availability constraints.  

---

## Author

Paul Santos Ramos  

This project was developed as part of a Master's Degree in Cybersecurity and Data Intelligence.
