# Data Science Project Report: Synthetic DNA Analysis

## 1. Introduction
This report details a data science project focused on analyzing a synthetic DNA dataset. The primary objective of this project is to demonstrate a typical data science workflow, including data collection, exploration, cleaning, preprocessing, exploratory data analysis (EDA), visualization, statistical modeling, and interpretation of results. This project aims to provide a comprehensive example suitable for a resume portfolio, showcasing skills in Python programming, data manipulation with Pandas, data visualization with Matplotlib and Seaborn, and machine learning with Scikit-learn.

## 2. Dataset Overview
The dataset used in this project is `synthetic_dna_dataset.csv`. It contains synthetic DNA sequence information and associated attributes. The dataset comprises 3000 entries and 13 columns, including:

*   `Sample_ID`: Unique identifier for each DNA sample.
*   `Sequence`: The synthetic DNA sequence itself.
*   `GC_Content`: Percentage of Guanine (G) and Cytosine (C) bases in the sequence.
*   `AT_Content`: Percentage of Adenine (A) and Thymine (T) bases in the sequence.
*   `Sequence_Length`: Length of the DNA sequence.
*   `Num_A`, `Num_T`, `Num_C`, `Num_G`: Counts of Adenine, Thymine, Cytosine, and Guanine bases, respectively.
*   `kmer_3_freq`: Frequency of 3-mer sequences (sequences of 3 nucleotides).
*   `Mutation_Flag`: A binary flag indicating the presence of a mutation.
*   `Class_Label`: Categorical label indicating the origin of the DNA (e.g., Human, Plant, Virus).
*   `Disease_Risk`: Categorical label indicating the associated disease risk (High, Medium, Low).

Initial exploration revealed no missing values in any of the columns, indicating a clean dataset for analysis. The data types are appropriate for the respective columns, with numerical data for content and counts, and object types for IDs, sequences, and categorical labels.

## 3. Data Cleaning and Preprocessing
Before proceeding with analysis and modeling, the dataset underwent several preprocessing steps:

*   **Categorical Feature Encoding**: The `Class_Label` and `Disease_Risk` columns, being categorical, were converted into numerical format using one-hot encoding. This transformation creates new binary columns for each category, which is necessary for most machine learning algorithms. For `Disease_Risk`, `Disease_Risk_Low` and `Disease_Risk_Medium` were created, with `High` being implied when both are false.
*   **Feature Removal**: The `Sample_ID` and `Sequence` columns were dropped from the dataset. `Sample_ID` is a unique identifier and does not contribute to predictive modeling, while `Sequence` itself requires more advanced bioinformatics techniques for direct use in this type of tabular modeling approach.

These steps ensured that the data was in a suitable format for subsequent exploratory data analysis and machine learning tasks.

## 4. Exploratory Data Analysis (EDA) and Visualization
Exploratory Data Analysis was performed to understand the underlying patterns, relationships, and distributions within the dataset. Several visualizations were generated to provide insights:

### 4.1. Distribution of GC Content
This histogram illustrates the distribution of GC content across the synthetic DNA samples. It helps in understanding the variability and central tendency of GC content in the dataset.

![GC Content Distribution](https://private-us-east-1.manuscdn.com/sessionFile/4MhSbUvLxiGt5EY2ckzpJN/sandbox/rmQPlfBPgvmiFgcvX6t51N-images_1756237218198_na1fn_L2hvbWUvdWJ1bnR1L3N5bnRoZXRpY19kbmFfcHJvamVjdC9nY19jb250ZW50X2Rpc3RyaWJ1dGlvbg.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvNE1oU2JVdkx4aUd0NUVZMmNrenBKTi9zYW5kYm94L3JtUVBsZkJQZ3ZtaUZnY3ZYNnQ1MU4taW1hZ2VzXzE3NTYyMzcyMTgxOThfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzTjViblJvWlhScFkxOWtibUZmY0hKdmFtVmpkQzluWTE5amIyNTBaVzUwWDJScGMzUnlhV0oxZEdsdmJnLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=SbqVBqHVUkLSmLZ5eisz9ioXsZMNcBGL5kwi6FynyWeDSwGqNolvstGtbTnY0TR4SxSGF1w4u57lgf8whlrjLIARlU2DyQVMtGGRSpV8yZiHqrBv~wCVEtC3hxIKG40e51XNdkFK~lKcS9xJ~qHeV7qSlTP8AQ9y3izQm20h9sqmt6GKk44XUbEKdvTdr8SMII9IJuUCIn~HQAdCKRcBUnlET8xblIQUmhtqkPk463v0EUuskfP229iGJl-M7Z2fRYWlQxUbQSIih0jIcsuMuP3p1nsLANK8mdtdep~OVa0RQomm1xjEAO3SowE9ioG-tiCi1U-sskFCcL5~bkTBDQ__)

### 4.2. GC Content vs. Sequence Length by Mutation Flag
This scatter plot visualizes the relationship between GC content and sequence length, with points colored according to the `Mutation_Flag`. This helps in identifying if there are any visible clusters or patterns related to mutations based on these two features.

![GC Content vs. Sequence Length Scatter Plot](https://private-us-east-1.manuscdn.com/sessionFile/4MhSbUvLxiGt5EY2ckzpJN/sandbox/rmQPlfBPgvmiFgcvX6t51N-images_1756237218198_na1fn_L2hvbWUvdWJ1bnR1L3N5bnRoZXRpY19kbmFfcHJvamVjdC9nY19zZXF1ZW5jZV9sZW5ndGhfc2NhdHRlcg.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvNE1oU2JVdkx4aUd0NUVZMmNrenBKTi9zYW5kYm94L3JtUVBsZkJQZ3ZtaUZnY3ZYNnQ1MU4taW1hZ2VzXzE3NTYyMzcyMTgxOThfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzTjViblJvWlhScFkxOWtibUZmY0hKdmFtVmpkQzluWTE5elpYRjFaVzVqWlY5c1pXNW5kR2hmYzJOaGRIUmxjZy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=IRAcA7vY1QZBeHcFrZHq1hn5GGinn8LWYNOX5TuvTdmb8e2MaJO6vLbBst66uvt-8Bd8TDosBk4TBgyCkhpB7aYC-kDYA1tdizBTPCQMUirNgZH6WysOb2KHih-F4PurKepHhwUPp~KnrM7Mua~iKchFfyF5K6NZfq63RjfXQmcX5KO6-5MqOkIkdsXtmU3-erU-zW7ouAvSRpOTGJbyWcGGQIRevTjxlYsOF1u6X1FVIuBV0egPzfs5mG3Xer6GJEwMZnBQ2UtqR7FrlJbV6PyTkIBP7OccDpOj7RH5zaFzz~gaBofcOCPCcMbiO-FVk1piJueZ7I5EbWaxV2fe1A__)

### 4.3. Count of Each Class Label
This bar plot shows the frequency of each `Class_Label` (Human, Plant, Virus) in the dataset. It provides an overview of the distribution of DNA origins.

![Class Label Count](https://private-us-east-1.manuscdn.com/sessionFile/4MhSbUvLxiGt5EY2ckzpJN/sandbox/rmQPlfBPgvmiFgcvX6t51N-images_1756237218199_na1fn_L2hvbWUvdWJ1bnR1L3N5bnRoZXRpY19kbmFfcHJvamVjdC9jbGFzc19sYWJlbF9jb3VudA.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvNE1oU2JVdkx4aUd0NUVZMmNrenBKTi9zYW5kYm94L3JtUVBsZkJQZ3ZtaUZnY3ZYNnQ1MU4taW1hZ2VzXzE3NTYyMzcyMTgxOTlfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzTjViblJvWlhScFkxOWtibUZmY0hKdmFtVmpkQzlqYkdGemMxOXNZV0psYkY5amIzVnVkQS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=ZC-ut~lCSR8VIsR0vxrJocNm7Hd4MXYeZpHREV1Sn~v-tfTNpfP9ff0ectDWrFWBpszcorMRfJgfY29UkQwURVOYaRLzln0xNw-teWgaw8LMR3lzBbCc1UuXM1V78ckrYYKIRJwcNLhauYwrQdiSHiZNK6lRlS9kl0gjPeE92Wa4TcK9Gthr5ZHbbPX5OhS0D~Vem-K2eGoOW3~zMeE6vDeEhmuL-YeYRmUPGb94NI32b~0KCzzaszK8AnyIlwcZC5~t8HyECClgTGS~DNUpRXC7YZyGDDLAR-xUYR1Ix6u7uQPfCVbQGjXC0WzeVNIRw1njyYNBo2Zllgw~0CZC5Q__)

### 4.4. Distribution of Disease Risk
This bar plot displays the distribution of `Disease_Risk` categories (High, Medium, Low). It helps in understanding the prevalence of each risk level in the dataset.

![Disease Risk Distribution](https://private-us-east-1.manuscdn.com/sessionFile/4MhSbUvLxiGt5EY2ckzpJN/sandbox/rmQPlfBPgvmiFgcvX6t51N-images_1756237218200_na1fn_L2hvbWUvdWJ1bnR1L3N5bnRoZXRpY19kbmFfcHJvamVjdC9kaXNlYXNlX3Jpc2tfZGlzdHJpYnV0aW9u.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvNE1oU2JVdkx4aUd0NUVZMmNrenBKTi9zYW5kYm94L3JtUVBsZkJQZ3ZtaUZnY3ZYNnQ1MU4taW1hZ2VzXzE3NTYyMzcyMTgyMDBfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzTjViblJvWlhScFkxOWtibUZmY0hKdmFtVmpkQzlrYVhObFlYTmxYM0pwYzJ0ZlpHbHpkSEpwWW5WMGFXOXUucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=l6rpdGV9x-jK--eYndk20RdKL2ayygD-LGrCu-Qhnm8pbOlNDRmsMRM9I3cK9eiwJz5SspDNJWKMxAetlX8J6QFBklL~dxsylZwcs3bprKCgKeby-Blz8NWVWVqGwbmWxbj9LSTkUw1bAZEsF4WX~T12O7tzreOIgpB49HpO85~6ub50DJeoAHdsVcTXdiOR~K6Y1sD4~wtBjRRO~Nxl8j~uG8KCSFubxEVGFHklHDx~FBUfwjXlIkZRlxbfCGg4BYaRuEvCJxMhko6OhYLgjFXn96MV7KXnqN~HfY8SJQFRmBVCbP1oGpUd347GHIHsEp1xdKJuxC247Dz03DdlJQ__)

These visualizations provide a foundational understanding of the dataset's characteristics and potential relationships between variables, guiding further analysis and modeling efforts.

## 5. Statistical Analysis and Modeling
For statistical analysis and predictive modeling, a Random Forest Classifier was chosen due to its robustness and ability to handle various data types. The goal was to predict the `Disease_Risk` based on the other features.

### 5.1. Model Training and Evaluation
The preprocessed data was split into training and testing sets (70% training, 30% testing). A Random Forest Classifier with 100 estimators was trained on the training data. The model's performance was then evaluated on the unseen test data.

**Model Accuracy**: 0.3511

**Classification Report**:
```
              precision    recall  f1-score   support

        High       0.32      0.32      0.32       319
         Low       0.34      0.38      0.36       282
      Medium       0.40      0.35      0.38       299

    accuracy                           0.35       900
   macro avg       0.35      0.35      0.35       900
weighted avg       0.35      0.35      0.35       900
```

The model achieved an accuracy of approximately 35.11%. The classification report indicates that the model's performance is relatively low across all classes, with precision, recall, and F1-scores hovering around 0.32-0.40. This suggests that the model struggles to accurately classify the disease risk categories given the current features.

### 5.2. Feature Importance
Feature importance analysis from the Random Forest model provides insights into which features were most influential in the prediction process:

```
kmer_3_freq          0.221083
Num_C                0.124260
Num_G                0.124153
Num_A                0.121955
Num_T                0.121746
GC_Content           0.085794
AT_Content           0.085777
Mutation_Flag        0.035626
Class_Label_Human    0.026949
Class_Label_Plant    0.026442
Class_Label_Virus    0.026215
Sequence_Length      0.000000
dtype: float64
```

## 6. Results Interpretation and Insights

### 6.1. Model Performance
The Random Forest Classifier model achieved an accuracy of approximately 35.11% in predicting disease risk. The classification report further indicates low precision, recall, and F1-scores across all three disease risk categories (High, Low, Medium). This suggests that the current features and model are not highly effective in accurately predicting disease risk from the synthetic DNA dataset.

### 6.2. Feature Importance
Analysis of feature importance reveals the following:

*   **kmer_3_freq** is the most significant predictor, indicating that the frequency of 3-mer sequences plays a crucial role in determining disease risk.
*   **Nucleotide counts (Num_A, Num_T, Num_C, Num_G)** also show considerable importance, suggesting that the individual composition of nucleotides within the DNA sequence is relevant.
*   **GC_Content and AT_Content** have moderate importance, which is expected given their direct relationship with nucleotide counts.
*   **Mutation_Flag** has low importance, implying that the presence of a general mutation flag, as defined in this dataset, does not strongly correlate with disease risk.
*   **Class_Label (Human, Plant, Virus)** and **Sequence_Length** show very low to no importance. This suggests that the origin of the DNA sequence (human, plant, or virus) and its overall length are not significant factors in predicting disease risk based on this model.

### 6.3. Key Insights
1.  **Sequence Composition is Key:** The most influential features are related to the detailed composition of the DNA sequence, particularly the frequency of 3-mer sequences and the counts of individual nucleotides. This highlights the importance of granular sequence information for disease risk prediction.
2.  **Limited Predictive Power of High-Level Features:** Broad features like `Mutation_Flag`, `Class_Label`, and `Sequence_Length` contribute minimally to the model's predictive power. This could indicate that these features, as represented in this dataset, do not capture sufficient information related to disease risk, or that their relationship with disease risk is more complex than what a Random Forest model can capture directly.
3.  **Model Limitations:** The low accuracy suggests that a more sophisticated model or additional, more relevant features might be necessary to achieve higher predictive performance for disease risk. Further research into biological significance of k-mer frequencies and specific mutation types could yield better results.

## 7. Conclusion and Future Work
This project successfully demonstrated a complete data science workflow using a synthetic DNA dataset. While the Random Forest model provided some insights into feature importance, its predictive accuracy for disease risk was limited. This highlights the complexity of biological data and the need for potentially more advanced techniques or domain-specific feature engineering.

Future work could involve:

*   **Advanced Feature Engineering**: Extracting more sophisticated features from the raw DNA sequences, such as specific motif occurrences or structural properties.
*   **Deep Learning Models**: Exploring neural network architectures (e.g., Convolutional Neural Networks or Recurrent Neural Networks) that are well-suited for sequence data analysis.
*   **Larger and More Diverse Datasets**: Utilizing larger and more biologically diverse datasets to improve model generalization and performance.
*   **Hyperparameter Tuning**: Optimizing the hyperparameters of the chosen machine learning model to potentially improve its performance.

This project serves as a solid foundation for understanding and applying data science principles to biological data, and provides a clear path for future enhancements to achieve higher predictive accuracy.

