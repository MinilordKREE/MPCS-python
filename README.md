# Project Report: Moral Foundations Classification on Reddit Discussions

# Introduction

The vast and ever-expanding textual records of contemporary social issues, often debated online with moral undertones, present a rich yet intricate landscape for exploring how moral concerns manifest and evolve in real-world discussions. Central to this exploration is Moral Foundations Theory, a comprehensive framework categorizing moral standards into five core dimensions: Care/harm, Fairness/cheating, Loyalty/betrayal, Authority/subversion, and Sanctity/degradation. This framework has become instrumental in data-driven analyses of online discourse, offering valuable insights into the moral underpinnings of public debates.
To deepen our understanding of this interplay, I aim to assess the performance of various machine learning models in classifying moral foundations within discussions on the Reddit platform—a vibrant and diverse arena for contemporary online dialogue. This investigation will shed light on the effectiveness of these models in capturing the nuanced moral dimensions of user-generated content.
Moreover, my goal is to become an AI engineer specialized in LLM. Thus I will also explore the potential using LLM to clarify the human subjectivity hiding in these text. 

## Objectives
The dataset is from the paper: The Moral Foundations Reddit Corpus https://arxiv.org/abs/2208.05545 and you can check the detail of the dataset here in huggindface: https://huggingface.co/datasets/USC-MOLA-Lab/MFRC

Implement and assess baseline classification models, specifically focusing initially on Random Forest and Logistic Regression.
To improve model performance, I used a better encoding method via RoBERTa embedding layer instead of using simple Tf-id to vectorize 
the text. I used one A100 GPU with 100 GB memory space to run the embedding process.

Evaluate and compare the accuracy and interpretability of different models in classifying moral foundations.

# Progress and Methodology

## Week 4: Data Preparation and Preprocessing

During this phase, Reddit discussion data was collected and preprocessed. The preprocessing steps included:

Text cleaning, target column deciding.

Tokenization and vectorization to convert text into numerical features.

Label encoding for the five moral dimensions: Care, Fairness, Loyalty, Authority, and Sanctity.

## Week 5: Exploratory Data Analysis (EDA) and Initial Modeling

The exploratory data analysis revealed the distribution and frequency of different moral dimensions across the dataset. Preliminary insights indicated certain moral dimensions (e.g., Care and Fairness) appeared more frequently in user discussions than others (e.g., Sanctity).

Initial modeling commenced with baseline models:

Random Forest: Employed due to its robustness to noise and capability to capture non-linear relationships.

Logistic Regression: Selected for interpretability and computational efficiency.

Week 6: Baseline Model Implementation

Baseline models were fully implemented and evaluated. Performance metrics used mainly is AUC scores. At this stage:

Logistic Regression showed clear interpretability and quick convergence.
Random Forest provided better accuracy, though with less transparency regarding feature importance.
However, the performance AUC is between 0.5 to 0.6, which means it is slightly better than random guessing.

## Week 7: Model Improvement

Further improvements were achieved by optimizing text encoding method. Instead of using TF-ID to simplely convert the text in into vectors, RoBERTa model first tokenize the text into token id arrays. Then parse the array into encoder-decoder neural network transformer structure to convert the array into embedding matrix to better capture the semantic meaning.
This phase significantly enhanced the accuracy and reliability of predictions for the less frequently occurring moral foundations (e.g., Sanctity).
The best result can even result in 0.66 AUC scores (from 0.53).

## Week 8: Exploration of using LLM

The trandition machine learning model seems performed poorly. As an AI engineer, I also explore the potential of using LLM to clarify the emotions hidden behind these text. Findings: the LLM can perform quite well on these tasks.

# Results and Findings

Although the task seems too difficult to the traditional simple machine learning models, using advance text encoding can optmize the performance in a promising improvement. The LLMs show its potential on not only objective tasks, but also these subjective tasks.