# AI Tinkerers Kuala Lumpur October 2024 Hackathon (LLM-as-a-Judge): Fine-Tuning Malaysian DeBERTaV2 & Mistral 7B for Logical Consistency Classification and Reasoning

### Overview
This repo details code written as part of the **1st place solution** for the [AI Tinkerer's Hackathon in Kuala Lumpur](https://www.linkedin.com/posts/supa-ai_llms-techinnovation-llm-activity-7256832143694192640-INSI?utm_source=share&utm_medium=member_desktop)
for an LLM-as-a-Judge use case.

It involves fine-tuning **Malaysian DeBERTaV2** and **Mistral 7B** models for yes/no classification and reasoning tasks, focusing on a Natural language inference (NLI) task.

In our case, NLI is the task of determining whether a "hypothesis" is true (*entailment*) or false (*contradiction*) given a `statement`-`question`/`paragraph`-`statement` pair. By leveraging translated datasets and Chain-of-Thought reasoning techniques, this project demonstrates the potential for finetuned smaller models to act as scalable judges, in line with the [JudgeLM paper](https://arxiv.org/abs/2310.17631).

Crucially, **we leverage open source models and datasets in the Malay language** to enable adaptation to the local Malaysian context.

### Methodology
A comprehensive presentation we prepared for the Hackathon can be found [here](/miscellaneous/werecooked_LLM_JUDGE_v20241024.pdf).

1. [Dataset Translation](/notebooks-data-preparation/): Translated English datasets into Malay to enable focused fine-tuning for Malay language understanding using **OpenAI's 4o-mini**.
2. [Chain-of-Thought Reasoning](/notebooks-data-preparation/): Augmented datasets with CoT reasoning using **OpenAI's 4o-mini** to enhance logical reasoning capabilities.
3. [Fine-Tuning](/notebooks-finetuning-models/): Utilized **Google Colab's A100 GPU** (40 GB VRAM) to fine-tune models on the curated datasets using **QLoRA** and **Huggingface's SFTTrainer**.
4. [Benchmarking](/notebooks-benchmarking-exercises/): Benchmarking and training runs was done/monitored using **weave** (**Weights & Biases**) 

### Models
Original models:
- https://huggingface.co/mesolitica/malaysian-mistral-7b-32k-instructions-v4
- https://huggingface.co/mesolitica/malaysian-debertav2-base

Fine-tuned models:
- NLI only: https://huggingface.co/wanadzhar913/malaysian-debertav2-finetune-on-boolq
- NLI only: https://huggingface.co/wanadzhar913/malaysian-mistral-llmasajudge-v2
- NLI & Reasoning: https://huggingface.co/wanadzhar913/malaysian-mistral-llmasajudge-v3

### Datasets
Original datasets:
- https://huggingface.co/datasets/google/boolq
- https://huggingface.co/datasets/r-three/fib

Translated datasets:
- https://huggingface.co/datasets/wanadzhar913/fib-malay
- https://huggingface.co/datasets/wanadzhar913/boolq-malay

Translated & Reasoning column generated datasets:
- https://huggingface.co/datasets/wanadzhar913/fib-malay-with-chain-of-thought
- https://huggingface.co/datasets/wanadzhar913/boolq-malay-with-chain-of-thought

### Results
Our approach yielded significant improvements in logical reasoning tasks for Malay and English language , validated by metrics including accuracy, F1-score. These results secured **1st place** at the **AI Tinkerer's Hackathon in Kuala Lumpur**.*

| **Model**            | **Accuracy (%)** | **F1-Score (%)** |
|----------------------|------------------|------------------|
| OpenAI 4o-mini       | 78               | 80               |
| Malaysian DeBERTaV2  | 51               | 48               |
| Malaysian Mistral V2 | 65               | 74               |
| Malaysian Mistral V2 | 61               | 69               |

**Due to time/compute constraints, we didn't evaluate on the entire test set. You can check how we sampled the testing set [here](/notebooks-benchmarking-exercises/generate_validation_dataset_for_presentation.ipynb).*

### Acknowledgments
Special thanks to:
- [Mesoltica](https://github.com/mesolitica) for their open-source models we used for fine-tuning.
- [AI Tinkerer's Kuala Lumpur](https://kuala-lumpur.aitinkerers.org/) for organizing the hackathon.
- [Joseph](https://www.linkedin.com/in/joseph-jlyc-chin/) from [DocuAsk](https://www.linkedin.com/company/docuask/) for providing OpenAI credits enabling us to access **4o-mini**.
- Team members and collaborators for their contributions.

### Improvements
- **Due to time/compute constraints, we didn't evaluate on the entire test set**. A more accurate result can be obtained by evaluating on the entire dataset(s).
- Set `bf16` parameter to `True` to optimize compute efficiency without significantly sacrificing model accuracy.
- Increase the `gradient_accumulation_steps` to deal with the small GPU constraints or increase the `batch_size` if we've access to a larger GPU. The reasoning is mainly to avoid [Out of Memory Errors (OOM)](https://discuss.huggingface.co/t/batch-size-vs-gradient-accumulation/5260).
- Given more compute resources, we can also increase our `patience` variable and train for more than 10 epochs.
- **Limiting the reasoning portion (in the training dataset) to only be in Malay**. Since the model has been instruction finetuned to mainly reply in Malay, it'd be confusing to have it reason back in English.
