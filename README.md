# Data Science Project: Synthetic DNA Analysis

This repository contains a data science project focused on analyzing a synthetic DNA dataset. The project demonstrates a complete data science workflow, from data collection and exploration to cleaning, preprocessing, exploratory data analysis (EDA), visualization, statistical modeling, and interpretation of results.

## Project Structure

*   `synthetic_dna_dataset.csv`: The original dataset used in this project.
*   `cleaned_synthetic_dna_dataset.csv`: The cleaned and preprocessed dataset.
*   `explore_data.py`: Python script for initial data exploration.
*   `clean_data.py`: Python script for data cleaning and preprocessing.
*   `eda_and_visualization.py`: Python script for exploratory data analysis and generating visualizations.
*   `model_and_analysis.py`: Python script for statistical analysis and machine learning modeling.
*   `gc_content_distribution.png`: Visualization of GC Content Distribution.
*   `gc_sequence_length_scatter.png`: Visualization of GC Content vs. Sequence Length.
*   `class_label_count.png`: Visualization of Class Label Counts.
*   `disease_risk_distribution.png`: Visualization of Disease Risk Distribution.
*   `data_science_project_report.md`: A detailed report summarizing the project, methodology, findings, and conclusions.
*   `README.md`: This README file.

## How to Run the Project

1.  **Clone the repository (or download the files).**

2.  **Install the required Python libraries:**
    ```bash
    pip install pandas matplotlib seaborn scikit-learn
    ```

3.  **Run the scripts in sequence:**
    *   **Data Exploration:**
        ```bash
        python3 explore_data.py
        ```
    *   **Data Cleaning and Preprocessing:**
        ```bash
        python3 clean_data.py
        ```
    *   **EDA and Visualization:**
        ```bash
        python3 eda_and_visualization.py
        ```
        (This will generate the PNG visualization files in the project directory.)
    *   **Statistical Analysis and Modeling:**
        ```bash
        python3 model_and_analysis.py
        ```

4.  **Review the Report:**
    Open `data_science_project_report.md` to read the comprehensive project report.

## Key Findings

*   The most influential features for predicting disease risk were `kmer_3_freq` and individual nucleotide counts (`Num_A`, `Num_T`, `Num_C`, `Num_G`).
*   High-level features like `Mutation_Flag`, `Class_Label`, and `Sequence_Length` had limited predictive power.
*   The Random Forest model achieved an accuracy of approximately 35.11% for disease risk prediction, indicating the need for more advanced techniques or features for better performance.

## Future Work

*   Explore advanced feature engineering from raw DNA sequences.
*   Implement deep learning models for sequence data analysis.
*   Utilize larger and more diverse datasets.
*   Perform hyperparameter tuning for model optimization.

This project serves as a strong foundation for demonstrating data science skills and can be extended for further research and development.

