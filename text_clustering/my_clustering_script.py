# in powershell - $env:GEMINI_API_KEY = "your_api_key_here"


import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import re
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords as nltk_stopwords
import nltk
from langdetect import detect, DetectorFactory
import google.generativeai as genai

# --- GEMINI API INTEGRATION ---
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("❌ ERROR: GEMINI_API_KEY environment variable not set.")
    print("Please set it on your command line before running the script.")
    exit()

DetectorFactory.seed = 0

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

UNIVERSAL_STOPWORDS = set(nltk_stopwords.words('english') + [
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'from', 'by', 'this', 'that', 'it', 'its', 'her', 'their', 'our',
    'what', 'where', 'how', 'why', 'who', 'whom', 'which', 'whether',
    'yesterday', 'today', 'tomorrow', 'morning', 'evening', 'night', 'day', 'days', 'hr', 'hrs', 'hour', 'hours', 'time', 'date', 'week', 'month', 'year', 'ago',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero',
    'consumer', 'customer', 'number', 'no', 'code', 'id', 'location', 'address', 'phone', 'mobile', 'call', 'report', 'registered',
    'ok', 'yes', 'no', 'not', 'hi', 'hello', 'sir', 'madam', 'pls', 'please', 'regards', 'type', 'urban', 'complaint', 'detail', 'general',
    'kv', 'tf', 'na', "service", "request", "feedback", "query", "regarding", "about", "given"
])

def clean_text(text: str) -> str:
    """Removes extra whitespace from a string."""
    return re.sub(r'\s+', ' ', text.strip())

def get_top_keywords(remarks: list[str], n_keywords: int = 10) -> list[str]:
    """Extracts top keywords from a list of remarks using TF-IDF."""
    if not remarks or len(remarks) < 2:
        return []
        
    try:
        vectorizer = TfidfVectorizer(
            stop_words=list(UNIVERSAL_STOPWORDS),
            ngram_range=(1, 2),
            min_df=0.1
        )
        tfidf_matrix = vectorizer.fit_transform(remarks)
        feature_names = vectorizer.get_feature_names_out()
        
        scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
        top_indices = scores.argsort()[-n_keywords:][::-1]
        
        top_keywords = [feature_names[i] for i in top_indices]
        return top_keywords
    except ValueError:
        return []

def get_genai_cluster_name(cluster_texts: list[str], top_keywords: list[str]) -> str:
    """Generates a category name using the Gemini model based on keywords and remarks."""
    print("   [Gen AI Naming] Sending prompt to Gemini model...")
    if not cluster_texts:
        return "Uncategorized Remarks"
    
    sample_size = min(20, len(cluster_texts))
    text_sample = "\n".join(cluster_texts[:sample_size])

    prompt = f"""
    You are an expert at analyzing customer feedback. Your task is to provide a single, concise, and professional category name for a group of similar remarks. The name should be no more than 7 words.

    The most representative keywords for this category are: {', '.join(top_keywords)}.

    A sample of the remarks for this category:
    ---
    {text_sample}
    ---

    Based on these keywords and remarks, the best category name is:
    """

    try:
        model = genai.GenerativeModel('models/gemma-3n-e2b-it')
        response = model.generate_content(prompt)
        name = response.text.strip()
        return name
        
    except Exception as e:
        print(f"   [Gen AI Naming] ERROR: API call failed. Falling back to a generic name. Error: {e}")
        return "Generic Unnamed Category"

def load_excel_file(file_path: str, column: str) -> tuple[list[str], pd.DataFrame]:
    """Loads remarks from an Excel file."""
    print(f"Loading data from '{file_path}'...")
    if not os.path.exists(file_path): raise FileNotFoundError(f"Excel file not found at '{file_path}'.")
    df = pd.read_excel(file_path)
    print(f"Loaded {len(df)} rows.")
    if column not in df.columns: raise KeyError(f"Column '{column}' not found. Available columns: {df.columns.tolist()}")
    remarks_list = df[column].astype(str).tolist()
    print(f"Extracted {len(remarks_list)} raw remarks from column '{column}'.")
    return remarks_list, df

def save_results(df: pd.DataFrame, output_path: str):
    """Saves the clustered results to an Excel file."""
    print(f"\nSaving results to '{output_path}'...")
    df.to_excel(output_path, index=False)
    print("Saved successfully.")

def segregate_remarks_by_language(raw_remarks: list[str], min_text_for_detection: int = 10) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Segregates remarks into English and other languages."""
    print(f"Starting language segregation for {len(raw_remarks)} remarks...")
    english_remarks_with_indices, other_remarks_with_indices = [], []
    for i, remark in enumerate(raw_remarks):
        cleaned_remark_for_lang_detect = clean_text(remark.lower())
        if len(cleaned_remark_for_lang_detect) < min_text_for_detection or not any(char.isalpha() for char in cleaned_remark_for_lang_detect):
            other_remarks_with_indices.append((i, remark))
            continue
        try:
            detected_lang = detect(cleaned_remark_for_lang_detect)
            if detected_lang == 'en':
                english_remarks_with_indices.append((i, remark))
            else:
                other_remarks_with_indices.append((i, remark))
        except Exception:
            other_remarks_with_indices.append((i, remark))
    print(f"Segregation complete. English: {len(english_remarks_with_indices)}, Other: {len(other_remarks_with_indices)}")
    return english_remarks_with_indices, other_remarks_with_indices

def get_unique_name(base_name: str, existing_names: set, suffix_identifier: str = "") -> str:
    """Generates a unique name by adding an alphabetic/numerical suffix if necessary."""
    name = re.sub(r'[^a-zA-Z\s]', '', base_name).strip()
    name = re.sub(r'\s+', ' ', name).strip()
    
    if not name:
        name = "Generic Category"

    alpha_suffix_idx = 0
    numeric_suffix_idx = 0
    
    original_base = name

    while name.lower() in existing_names: 
        if alpha_suffix_idx < 26:
            name = f"{original_base} {chr(65 + alpha_suffix_idx)}"
            alpha_suffix_idx += 1
        else:
            numeric_suffix_idx += 1
            alpha_suffix_idx_for_num = (alpha_suffix_idx - 26) % 26
            name = f"{original_base} {chr(65 + alpha_suffix_idx_for_num)}{numeric_suffix_idx}"
            alpha_suffix_idx += 1
            
    return name

def is_semantically_similar(name1: str, name2: str) -> bool:
    """Uses Gemini to determine if two phrases are semantically similar."""
    print(f"  [Gen AI Merging] Checking similarity between '{name1}' and '{name2}'...")
    prompt = f"""
    Are the following two phrases synonyms or do they convey the same meaning?
    Phrase 1: "{name1}"
    Phrase 2: "{name2}"
    Answer with a single word: "YES" or "NO".
    """
    try:
        model = genai.GenerativeModel('models/gemma-3n-e2b-it')
        response = model.generate_content(prompt)
        # Check if the response contains "YES" as a word.
        # This is more robust than a simple .strip().lower() check
        return "yes" in response.text.strip().lower()
    except Exception as e:
        print(f"   [Gen AI Merging] ERROR: API call for merging failed. Error: {e}")
        return False

def merge_similar_columns(df: pd.DataFrame, min_columns: int = 5) -> pd.DataFrame:
    """Merges columns with semantically similar names."""
    print("\n--- Merging similar columns (Semantic Match) ---")
    
    columns_to_process = [col for col in df.columns if 'Remarks' not in col]
    
    merged_mapping = {}
    
    sorted_cols = sorted(columns_to_process)
    
    for i in range(len(sorted_cols)):
        col1 = sorted_cols[i]
        if col1 in merged_mapping.values():
            continue

        for j in range(i + 1, len(sorted_cols)):
            col2 = sorted_cols[j]
            if col2 in merged_mapping:
                continue

            if is_semantically_similar(col1, col2):
                current_num_columns = len(columns_to_process) - len(merged_mapping)
                if current_num_columns - 1 >= min_columns:
                    print(f"   Merging '{col2}' into '{col1}' (Semantic Match)")
                    merged_mapping[col2] = col1
                else:
                    print(f"   Skipping merge of '{col2}' and '{col1}' to maintain min column count of {min_columns}.")
    
    final_data_dict = defaultdict(list)
    
    # We need to process the data to create the new merged DataFrame
    temp_df = df.copy()
    for source_col, target_col in merged_mapping.items():
        if target_col not in temp_df.columns:
            temp_df[target_col] = np.nan
        temp_df[target_col] = temp_df[target_col].fillna(temp_df[source_col])
        del temp_df[source_col]
        
    # Re-order columns for clarity
    merged_cols = list(set(merged_mapping.values()))
    original_unmerged = [col for col in df.columns if col not in merged_mapping and col not in merged_mapping.values()]
    final_column_order = original_unmerged + sorted(merged_cols)
    
    df_merged = temp_df[final_column_order]

    print("   Merging complete.")
    return df_merged

def main():
    excel_file_path = "./Meter.xlsx"
    text_column_name = "REMARKS"
    output_excel_path_wide_format = "../clustered_remarks_named.xlsx" 

    max_remark_clusters_limit = 10
    min_remark_clusters_limit_after_merge = 5
    
    print("\n--- Starting Text Clustering and Categorization Script (TF-IDF & K-Means) ---")
    try:
        raw_remarks_list, _ = load_excel_file(excel_file_path, text_column_name) 
        english_remarks_w_indices, other_remarks_w_indices = segregate_remarks_by_language(raw_remarks_list)
        
        english_remark_texts = [r_text for _, r_text in english_remarks_w_indices]
        english_remark_original_indices = [original_idx for original_idx, _ in english_remarks_w_indices]

        original_indexed_cluster_labels = np.full(len(raw_remarks_list), -2, dtype=int) 
        final_column_name_map = {}
        
        if english_remark_texts:
            print("\n--- Processing English Remarks for Clustering ---")
            
            vectorizer = TfidfVectorizer(stop_words=list(UNIVERSAL_STOPWORDS), ngram_range=(1, 3), min_df=5)
            tfidf_matrix = vectorizer.fit_transform(english_remark_texts)
            
            print(f"   [Clustering] TF-IDF matrix created with {tfidf_matrix.shape[1]} features.")
            
            n_clusters = min(max_remark_clusters_limit, len(english_remark_texts))
            kmeans_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            initial_cluster_labels = kmeans_model.fit_predict(tfidf_matrix)

            final_cluster_labels = initial_cluster_labels
            print(f"   [Clustering] K-Means complete. Found {len(set(final_cluster_labels))} clusters.")

            for i, clustered_label in enumerate(final_cluster_labels):
                original_indexed_cluster_labels[english_remark_original_indices[i]] = clustered_label
            
            final_unique_clusters = sorted([c for c in set(final_cluster_labels) if c != -1])
            print(f"   [Gen AI Naming] Naming {len(final_unique_clusters)} final clusters.")

            used_final_names = set()
            for cluster_id in final_unique_clusters:
                cluster_texts_original = [english_remark_texts[j] for j, label in enumerate(final_cluster_labels) if label == cluster_id]
                
                top_keywords = get_top_keywords(cluster_texts_original)
                print(f"   [Keywords] Top keywords for cluster {cluster_id}: {', '.join(top_keywords)}")
                
                proposed_final_name = get_genai_cluster_name(cluster_texts_original, top_keywords)
                final_name = get_unique_name(proposed_final_name, used_final_names, str(cluster_id))
                final_column_name_map[cluster_id] = final_name
                used_final_names.add(final_name.lower())
                print(f"   Final Category Name: '{final_name}'")
            
        else:
            print("No English remarks were found in the input data. Skipping English remark clustering.")
        
        final_wide_data_columns = {}
        for cluster_id, final_col_name in final_column_name_map.items():
            remarks_in_column = [
                raw_remarks_list[idx] for idx, label in enumerate(original_indexed_cluster_labels) if label == cluster_id
            ]
            final_wide_data_columns[final_col_name] = remarks_in_column

        # Handle uncategorized English remarks
        uncategorized_english_remarks = [raw_remarks_list[i] for i, label in enumerate(original_indexed_cluster_labels) if label == -1]
        if uncategorized_english_remarks:
            col_name_base = "Uncategorized English Remarks"
            col_name = get_unique_name(col_name_base, set(final_wide_data_columns.keys()).union(set(final_column_name_map.values())), "uncat_en")
            final_wide_data_columns[col_name] = uncategorized_english_remarks
            print(f"\nAdded column: '{col_name}' for {len(uncategorized_english_remarks)} remarks.")

        # Handle other language remarks
        if other_remarks_w_indices:
            col_name_base = "Other Language Remarks"
            col_name = get_unique_name(col_name_base, set(final_wide_data_columns.keys()).union(set(final_column_name_map.values())), "other_lang")
            final_wide_data_columns[col_name] = [r_text for _, r_text in other_remarks_w_indices]
            print(f"Added column: '{col_name}' for {len(other_remarks_w_indices)} remarks.")

        df_results_wide = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in final_wide_data_columns.items() ]))
        
        # --- NEW STEP: Merge similar columns
        df_final = merge_similar_columns(df_results_wide, min_columns=min_remark_clusters_limit_after_merge)

        print("\n--- Script Execution Complete ---")
        save_results(df_final, output_excel_path_wide_format)
        print("\n--- Sample Results (Wide Format) ---")
        print(df_final.head())

    except FileNotFoundError as fnfe:
        print(f"\n❌ ERROR: File not found. Please check 'excel_file_path'. Details: {fnfe}")
        exit(1)
    except KeyError as ke:
        print(f"\n❌ ERROR: Column not found. Please check 'text_column_name'. Details: {ke}")
        exit(1)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
