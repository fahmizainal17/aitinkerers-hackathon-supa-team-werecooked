# **🤖 AI Tinkerers Hackathon - Supa Team WeRecooked 🤖**

<!-- Badge to Visit Project -->
<div align="center"> 
    <a href="https://your-streamlit-app-url.com">
        <img src="https://img.shields.io/badge/Visit%20AI%20Tinkerers%20Hackathon%20Project-brightgreen?style=for-the-badge&logo=streamlit" alt="Visit AI Tinkerers Hackathon Project"/>
    </a>
</div>

---

## **📋 Overview**

The **AI Tinkerers Hackathon Project - Supa Team WeRecooked** is an initiative aimed at building and benchmarking AI models, particularly focusing on developing Large Language Model (LLM) Judges. The project covers dataset preparation, benchmarking, finetuning models, and creating a user-friendly interface to showcase our results using Streamlit.

---

## **Table of Contents**

1. [🎯 Objectives](#-objectives)
2. [🔧 Technologies Used](#-technologies-used)
3. [🗂️ Directory Structure](#-directory-structure)
4. [📁 Key Components](#-key-components)
5. [📊 Visual Elements and Data](#-visual-elements-and-data)
6. [🔄 Project Workflow](#-project-workflow)
7. [🎉 Conclusion](#-conclusion)
8. [🔮 Future Enhancements](#-future-enhancements)
9. [📚 References](#-references)
10. [📜 License](#-license)

---

## **🎯 Objectives**

- **📊 Develop benchmark models**: Create and evaluate LLMs like OpenAI Mini 4.0 and Mistral for specific use cases.
- **📁 Dataset preparation**: Preprocess datasets for benchmarking AI models, including language translations and task-specific data.
- **💻 Finetune models**: Explore and fine-tune models to improve accuracy on human preference datasets.
- **🚀 Showcase results**: Deploy results using Streamlit to create an interactive platform.

---

## **🔧 Technologies Used**

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)

---

## **🗂️ Directory Structure**

The project structure is as follows:

```plaintext
.
├── README.md
├── datasets
│   ├── boolq-english-train.jsonl
│   ├── fib-malay-openai.jsonl
│   └── for_presentation/
├── miscellaneous
│   └── AIT_Problemstatement2_SUPA.pdf
├── notebooks-benchmarking-exercises
│   ├── 03_benchmark_openaimini4_0_llmasajudge_v1_v2.ipynb
│   └── 03_benchmark_malaysian_mistral_llmasajudge_v2.ipynb
├── notebooks-data-preparation
│   ├── 01_dataset_prep_boolq_openai.ipynb
│   └── archive_01_dataset_prep_fib_t5.ipynb
├── notebooks-finetuning-models
│   ├── 02_finetune_v1_malaysian_debertav2_base.ipynb
│   └── 02_finetune_v2_malaysian_mistral_7b_32k_instructions_v4.ipynb
└── requirements.txt
```

---

## **📁 Key Components**

- **🔍 Data Preparation**: Includes notebooks and scripts for preparing datasets such as BoolQ and FIB for different models (e.g., OpenAI and T5).
- **📊 Benchmarking**: Focuses on evaluating the performance of various AI models such as OpenAI Mini 4.0 and Mistral LLM as Judges.
- **🔧 Model Finetuning**: Contains code to finetune models like Malaysian DeBERTaV2 and Mistral LLM.

---

## **📊 Visual Elements and Data**

- **📁 Datasets**: Processed and raw datasets are available for English and Malay human preferences.
- **📉 Benchmarking Notebooks**: Jupyter notebooks that contain the evaluation of models using different datasets.
- **🖥️ Interactive Interface**: Results will be visualized and shared using Streamlit.

---

## **🔄 Project Workflow**

1. **📂 Environment Setup**:
   - Set up a virtual environment and install required dependencies using `requirements.txt`.

2. **🔨 Data Processing**:
   - Prepare datasets for specific tasks, including translation and task-specific formatting.

3. **🚀 Model Finetuning**:
   - Fine-tune models on preprocessed datasets and evaluate performance.

4. **📊 Benchmarking**:
   - Benchmark models using evaluation notebooks to assess effectiveness in judging tasks.

5. **🌐 Deployment**:
   - Deploy results to an interactive Streamlit app for presentation.

---

## **🎉 Conclusion**

This project showcases the effectiveness of AI models like OpenAI Mini 4.0 and Mistral in the task of LLM Judges, providing a platform for performance evaluation and model comparison. By fine-tuning models on human-preference datasets, we aim to develop more accurate AI models.

---

## **🔮 Future Enhancements**

- **📈 Advanced Benchmarking**: Expand model benchmarking to include more datasets and AI models.
- **🤖 Further Finetuning**: Apply more advanced techniques for finetuning models to improve performance.
- **🌐 Enhanced UI**: Improve the Streamlit UI for better interaction and visualization.

---

## **📚 References**

- [OpenAI API Documentation](https://beta.openai.com/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

---

## **📜 License**

**Supa Team WeRecooked License**

All rights reserved. Unauthorized use or reproduction of any part of this project is prohibited. For usage and license inquiries, contact the project maintainers.

