# Text-Classification_Using_AI
This project provides a comprehensive workflow for categorizing customer remarks, particularly focusing on issues in the energy sector. It leverages both rule-based time categorization and advanced machine learning (using BERT and sentence transformers) for more nuanced classification. The project also includes functionalities for handling multilingual remarks and suggesting new categories based on uncategorized feedback using the Gemini API.

## Table of Contents

1.  [Project Description](#project-description)
2.  [Setup and Installation](#setup-and-installation)
3.  [Project Workflow](#project-workflow)
4.  [Code Breakdown](#code-breakdown)
    *   [Initial Setup and Library Installation](#initial-setup-and-library-installation)
    *   [Unsupervised/Rule-Based Categorization Functions](#unsupervisedrule-based-categorization-functions)
    *   [Labeled Data Preparation](#labeled-data-preparation)
    *   [Supervised ML Model Training and Evaluation](#supervised-ml-model-training-and-evaluation)
    *   [Supervised ML Categorization Pipeline (Time-Aware)](#supervised-ml-categorization-pipeline-time-aware)
    *   [Other Remarks Reallocation and Suggestion (Gemini API)](#other-remarks-reallocation-and-suggestion-gemini-api)
    *   [Environment Requirements Output](#environment-requirements-output)
5.  [Usage](#usage)
6.  [Input and Output Files](#input-and-output-files)
7.  [Future Improvements](#future-improvements)

## Project Description

The primary goal of this project is to automatically categorize unstructured customer remarks to identify common issues and patterns. It employs a multi-pronged approach:

*   **Rule-Based Time Categorization:** Quickly categorizes remarks containing explicit time references (e.g., "within 4 hours").
*   **Machine Learning Classification:** Trains a BERT-based model on labeled data to categorize remarks into predefined non-time-based categories. This model incorporates time-related features to improve accuracy.
*   **Language Segregation:** Separates English remarks from other languages.
*   **Uncategorized Remark Analysis:** Uses the Gemini API to analyze remarks that don't fit into existing categories and suggests potential new categories.

The output is a structured Excel file with remarks organized into distinct category columns.

## Setup and Installation

This project requires Python with several libraries. The provided notebook cells handle most of the installation.

**Prerequisites:**

*   Google Colab environment or a local Python environment with GPU support (recommended for Sentence Transformers and BERT).
*   Access to the internet for package downloads and Gemini API calls.
*   A Gemini API key (set in Colab secrets as `GOOGLE_API_KEY`).

**Installation Steps:**

1.  **Install Required Libraries:** Run the code cells designed for package installation. The notebook attempts to install core libraries like `langdetect`, `sentence-transformers`, `transformers`, `torch`, `scikit-learn`, and `pandas`. There are also attempts to install `cudf` using RAPIDS installers or conda, though these can sometimes be environment-dependent and may require troubleshooting or adjusting based on your specific Colab runtime or local environment.
2.  **Set up Gemini API:** Ensure your `GOOGLE_API_KEY` is added to Colab's Secrets or set as an environment variable.
3.  **Download NLTK Data:** The script automatically downloads necessary NLTK data (stopwords).

**Note on cudf/RAPIDS:** The installation of `cudf` can be sensitive to the environment (CUDA version, Python version). The notebook includes attempts to install it via `rapids-colab` or a direct conda installation using Mambaforge. If these steps fail, you may need to troubleshoot your environment, consider using a Colab runtime specifically configured for RAPIDS, or modify the code to use pandas instead where cudf is used.

## Project Workflow

The notebook is structured to follow a sequential workflow:

1.  **Initial Setup and Library Installation:** Installs core libraries and sets up the Gemini API key.
2.  **Data Loading and Preprocessing (Unsupervised):** Loads the raw remarks data and performs cleaning and initial time-based categorization. This part also includes language segregation and initial clustering for the unsupervised approach (though the final output might prioritize the ML approach if used).
3.  **Labeled Data Preparation (for ML):** Loads a separate labeled dataset (`Book1.xlsx`) and transforms it into a format suitable for training a supervised classifier.
4.  **Supervised ML Model Training and Evaluation:** Trains a BERT-based text classification model using the labeled data, incorporating time features, and evaluates its performance.
5.  **Prediction on New Remarks:** Uses the trained ML model to predict categories for a new set of raw remarks (`Supply.xlsx`). The results are structured into category columns.
6.  **Other Remarks Reallocation (Gemini API):** Processes remarks that were not categorized by either the rule-based or ML methods. It uses the Gemini API to attempt categorization into existing columns or suggest new categories based on their content.
7.  **Save Results:** The final categorized data is saved to an Excel file.

## Code Breakdown

This section provides a description of the logical code sections and their primary functions.

### Initial Setup and Library Installation

This section includes code for installing necessary Python packages like `langdetect`, `sentence-transformers`, `transformers`, `torch`, `scikit-learn`, `pandas`, and attempts to install `cudf` using various methods. It also contains code to set up the Gemini API key and download NLTK data.

### Unsupervised/Rule-Based Categorization Functions

This section contains functions for an unsupervised approach to categorization. It includes:
*   Importing necessary libraries and setting up the Gemini API key.
*   Defining universal stopwords.
*   Functions for cleaning text, performing vectorized time categorization using regex, extracting keywords using TF-IDF, generating cluster names using the Gemini API, ensuring unique names, checking semantic similarity of names using the Gemini API, and merging similar columns.
*   Functions for loading and saving Excel files.
*   Functions for segregating remarks by language and clustering remarks using sentence transformers and cuML KMeans.
*   A `main` function that orchestrates the steps of the unsupervised workflow.

### Labeled Data Preparation

This section includes code that loads a manually labeled dataset from an Excel file (`Book1.xlsx`). It transforms the data from a wide format (categories as columns) into a long format suitable for supervised machine learning, with columns for cleaned remarks and their corresponding category labels. It also handles potential pandas-generated column suffixes and identifies the unique category labels for training.

### Supervised ML Model Training and Evaluation

This section focuses on preparing the labeled data for training, building, and evaluating a supervised text classification model. It includes:
*   Splitting the labeled data into training and testing sets.
*   Generating sentence embeddings for the remarks using a Sentence Transformer model.
*   Training a Logistic Regression classifier on the sentence embeddings and evaluating its performance using metrics like accuracy and the classification report.

### Supervised ML Categorization Pipeline (Time-Aware)

This is a comprehensive section that integrates various steps for a supervised ML pipeline that is aware of time features. It includes:
*   Robust library installation attempts, including specific methods for `cudf`/RAPIDS and ensuring compatible versions of `transformers` and `accelerate`.
*   Text cleaning function.
*   A function to extract numerical time features from remarks.
*   A custom PyTorch `Dataset` class for handling data for the Hugging Face `Trainer`.
*   A main pipeline function (`main_classification_pipeline`) that loads both labeled data (for training) and new, uncategorized data (for prediction).
*   It prepares labeled data, splits it, extracts and *scales* time features, generates embeddings, combines features, trains a BERT-based `BertForSequenceClassification` model using the `Trainer`, and evaluates it.
*   It then processes the new remarks, extracts features, makes predictions using the trained model, and structures the output into a compacted Excel file with remarks grouped by predicted category.
*   This section also includes code to save the trained embedding model, classifier model, and scaler for future use.
*   Helper functions for computing evaluation metrics and a function to evaluate a saved model are also included here.

### Other Remarks Reallocation and Suggestion (Gemini API)

This section processes the output from previous categorization steps, specifically focusing on remarks that ended up in an 'Others' or uncategorized column. It includes:
*   Code to load the categorized data.
*   Functions that use the Gemini API (`google.generativeai`) to:
    *   Attempt to categorize remarks from the 'Others' column into existing columns based on an *exact* semantic match.
    *   Analyze remaining uncategorized remarks and suggest new category names with descriptions and estimated counts based on common themes found in the text.
*   A function to orchestrate this reallocation and suggestion process.
*   Code to save the updated data, showing which remarks (if any) were moved from 'Others' to other columns.

### Environment Requirements Output

This section contains a simple command to generate a `requirements.txt` file, which lists the Python packages and their versions installed in the current environment. This file is helpful for recreating the exact environment later.

## Usage

1.  **Upload Data Files:** Upload your labeled data (`Book1.xlsx`) and new remarks data (`Supply.xlsx`) to your Colab environment or the specified file paths. Ensure the column names in your files match the ones used in the code (e.g., "REMARKS", "From When Issue Is Coming", and the category names in `Book1.xlsx`).
2.  **Run Notebook Cells:** Execute the notebook cells sequentially from top to bottom.
3.  **Address Errors:** Pay attention to output messages and errors, especially during installation steps. You might need to restart the runtime, adjust file paths, or troubleshoot library compatibility issues (particularly with `cudf`) based on the errors.
4.  **Review Output:** The primary output will be an Excel file (`categorized_remarks_ML_model_compacted_time_aware.xlsx` by default) containing your new remarks categorized by the ML model and structured into columns. The output from the Gemini API step (`output.xlsx`) will show remarks reallocated from the 'Others' column and will print suggested new categories in the console output.

## Input and Output Files

*   **Input:**
    *   `Book1.xlsx`: Excel file containing manually labeled remarks in wide format (each column is a category). Assumed to have a header row.
    *   `Supply.xlsx`: Excel file containing new, uncategorized remarks to be processed. Assumed to have a column named "REMARKS" and optionally "From When Issue Is Coming".
*   **Intermediate Output:**
    *   `/content/label_encoder.pkl`: Pickle file saving the `LabelEncoder` used for mapping string labels to integers.
    *   `/content/remarks_dataset.pt`: PyTorch file saving the processed dataset (tokenized remarks and encoded labels).
    *   `/content/bert_trained`: Directory saving the trained BERT model and tokenizer.
    *   `sentence_transformer_model.pkl`: Joblib file saving the Sentence Transformer model.
    *   `logistic_regression_classifier.pkl`: Joblib file saving the trained Logistic Regression classifier.
    *   `scaler_for_time_features.pkl`: Joblib file saving the StandardScaler used for time features.
*   **Final Output:**
    *   `categorized_remarks_ML_model_compacted_time_aware.xlsx`: Excel file containing the new remarks from `Supply.xlsx`, categorized by the ML model and structured into columns.
    *   `output.xlsx`: Excel file generated by the Gemini API step, showing remarks from the original 'Others' column that were reallocated or remained uncategorized.
    *   `requirements.txt`: Text file listing the installed Python packages.

## Future Improvements

*   **Hyperparameter Tuning:** Optimize hyperparameters for the BERT model and training arguments for better performance.
*   **Cross-Validation:** Implement cross-validation during ML training for more robust model evaluation.
*   **More Sophisticated Time Feature Extraction:** Explore more advanced methods for extracting time information and potentially integrating date/time parsing.
*   **Handling Imbalanced Data:** Investigate techniques specifically for handling class imbalance if some categories have significantly fewer remarks.
*   **Fine-tuning BERT:** Instead of using Sentence Embeddings and Logistic Regression, fine-tune the BERT model directly on the classification task.
*   **Enhanced Gemini Integration:** Refine prompts and logic for Gemini API calls for both categorization and new category suggestion. Consider using Gemini's function calling capabilities if applicable.
*   **Web Application/API:** Develop a user interface or API to easily upload files and get categorized results.
*   **Active Learning:** Implement a process where the model identifies remarks it is uncertain about, and a human reviewer provides labels to improve the model iteratively.
