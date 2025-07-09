# Text-Classification_Using_AI
# Text-Classification_Using_AI: Advanced Text Clustering and Categorization

This repository contains a Python script designed for advanced text clustering and categorization of textual remarks, particularly useful for analyzing customer feedback, incident reports, or any large collection of unstructured text data. While the repository name suggests "Text Classification," the core functionality of this script focuses on **unsupervised text clustering and automated category naming** using modern AI techniques.

The script automatically groups similar remarks, assigns meaningful names to these groups, and outputs the results into a structured Excel file. It also includes capabilities for handling multi-language data by translating non-English remarks to English before processing.

## ✨ Features

* **Automated Text Clustering:** Groups similar textual remarks using a combination of HDBSCAN and Agglomerative Clustering.
* **Semantic Embeddings:** Leverages `SentenceTransformer` models to convert text into numerical representations (embeddings) that capture semantic meaning.
* **Intelligent Category Naming:** Automatically generates descriptive names for the identified clusters using TF-IDF and RAKE keyword extraction.
* **Multi-language Support:** Detects and translates non-English remarks to English using `googletrans` to enable unified clustering.
* **Noise Handling:** Intelligently attempts to assign "uncategorized" (noise) remarks from HDBSCAN to their nearest existing clusters, reducing the "Uncategorized Remarks" column.
* **Configurable Output:** Allows control over the maximum number of final output categories/columns.
* **Excel Output:** Saves the categorized remarks into a clean, wide-format Excel file for easy analysis.
* **GPU Acceleration (Optional):** Supports using NVIDIA GPUs (CUDA) for faster embedding generation if available and configured.

## 🚀 Getting Started

Follow these steps to set up and run the script on your local machine or a remote desktop.

### Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.8+**: Download from [python.org](https://www.python.org/downloads/).
* **pip**: Python's package installer (usually comes with Python).
* **Excel File**: Your input data should be in an Excel file (`.xlsx`) with a column containing the remarks you want to categorize.

### Installation

1.  **Clone the Repository:**
    First, clone this GitHub repository to your local machine:
    ```bash
    git clone [https://github.com/aadipatodia/Text-Classification_Using_AI.git](https://github.com/aadipatodia/Text-Classification_Using_AI.git)
    cd Text-Classification_Using_AI
    ```

2.  **Create a Virtual Environment (Recommended):**
    It's best practice to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects.
    ```bash
    python3 -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    * **Windows (Command Prompt):**
        ```cmd
        venv\Scripts\activate
        ```
    * **Windows (PowerShell):**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    * **macOS / Linux:**
        ```bash
        source venv/bin/activate
        ```
    You should see `(venv)` at the beginning of your terminal prompt, indicating the environment is active.

4.  **Install Dependencies:**
    Install all the required Python libraries. This might take a few minutes as it downloads large models for `sentence-transformers`.

    ```bash
    pip install pandas openpyxl sentence-transformers hdbscan scikit-learn rake-nltk nltk langdetect numpy googletrans==4.0.0-rc1
    ```
    * **Note on `googletrans`**: The version `4.0.0-rc1` is specified as it often works better than the latest stable `3.0.0` due to API changes. If you encounter issues, you might need to explore alternatives or other versions.

5.  **Download NLTK Data:**
    The script will automatically attempt to download `stopwords` from NLTK. If it fails or you prefer to do it manually:
    ```python
    python -c "import nltk; nltk.download('stopwords')"
    ```

### GPU Setup (Optional, for faster processing)

If your remote desktop has an **NVIDIA GPU** and you want to utilize it:

1.  **Ensure NVIDIA Drivers and CUDA Toolkit are Installed:**
    * Check your GPU model and driver version (`nvidia-smi` on Linux/WSL, Device Manager on Windows).
    * Download and install the latest compatible NVIDIA drivers from [NVIDIA's website](https://www.nvidia.com/drivers/).
    * Download and install the appropriate CUDA Toolkit version from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads) that matches your driver.
    * Verify installation: `nvcc --version` (for CUDA Toolkit) and `nvidia-smi` (for driver/CUDA version).

2.  **Install PyTorch with CUDA Support:**
    * **Deactivate** your virtual environment first (`deactivate`).
    * **Uninstall** any existing PyTorch: `pip uninstall torch torchvision torchaudio -y`
    * Go to the [PyTorch Get Started Locally](https://pytorch.org/get-started/locally/) page.
    * Select your OS, `pip` package, Python language, and **CRITICALLY, select the CUDA version that matches your system's CUDA Toolkit.**
    * Copy the exact `pip install` command provided by the PyTorch website.
    * **Re-activate** your virtual environment (`source venv/bin/activate` or `venv\Scripts\activate`).
    * Run the copied `pip install` command.

3.  **Modify `my_clustering_script.py`:**
    Open your `my_clustering_script.py` file and change the `device` parameter from `"cpu"` to `"cuda"` in the `preprocess_and_embed` function call and `name_model_for_clustering` initialization within `main()`:

    ```python
    # Inside main() function:
    # For the main embedding model:
    processed_remarks, embeddings = preprocess_and_embed(remark_texts_for_clustering, 0.8, "cuda")

    # For the name clustering model:
    name_model_for_clustering = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    ```

## 🏃‍♀️ Usage

1.  **Place your Excel file:**
    Rename your input Excel file to `Book1.xlsx` and place it in the same directory as `my_clustering_script.py`. If your file has a different name, update `excel_file_path` in `main()`.

2.  **Specify the remarks column:**
    Ensure the `text_column_name` variable in the `main()` function is set to the exact name of the column containing the remarks in your Excel file (e.g., `"REMARKS"`).

3.  **Adjust Parameters (Optional):**
    You can fine-tune the clustering behavior by modifying the parameters in the `main()` function:
    * `max_remark_clusters_limit`: Maximum number of initial remark clusters (default: 10).
    * `max_name_clusters_limit`: **Crucial!** Maximum number of final output columns/categories (default: 5). Keep this low to avoid too many columns.
    * `hdbscan_min_cluster_size`, `hdbscan_min_samples`: HDBSCAN density parameters (default: 2).
    * `assign_noise_to_nearest_cluster`: Set to `True` to attempt assigning HDBSCAN noise to existing clusters (default: `True`).
    * `noise_assignment_distance_threshold`: How close a noise point must be to a cluster to be assigned (default: 0.5). Adjust between 0.4-0.7.
    * `embedding_boilerplate_min_df`: Threshold for boilerplate phrase removal (default: 0.8).
    * `target_column_name_words`: Desired word count for generated category names (default: 7).

4.  **Run the Script:**
    With your virtual environment active, run the script from your terminal:
    ```bash
    python my_clustering_script.py
    ```

5.  **Check the Output:**
    A new Excel file named `clustered_remarks_named_output.xlsx` will be created in the parent directory (`../clustered_remarks_named.xlsx` as per your code). This file will contain your remarks categorized into different columns.

## 📂 Project Structure
.
├── Book1.xlsx                 # Your input Excel file (example)
├── my_clustering_script.py    # The main Python script for clustering and categorization
├── venv/                      # Python virtual environment (created upon setup)
└── README.md                  # This file

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

## 📄 License

This project is licensed under the [MIT License](LICENSE). (You should create a `LICENSE` file in your repository if you haven't already).
