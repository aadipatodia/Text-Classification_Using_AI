# in powershell - $env:GEMINI_API_KEY = "your_api_key_here"



import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import hdbscan
import re
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from collections import defaultdict, OrderedDict
import hashlib
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
# Configure the API with your key from the environment variable

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


# Define universal stopwords for TF-IDF

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

def get_genai_cluster_name(cluster_texts: list[str]) -> str:
    """Generates a category name using the Gemini model based on the cluster's remarks."""
    print("   [Gen AI Naming] Sending prompt to Gemini model...")
    if not cluster_texts:
        return "Uncategorized Remarks"
    
    # Take a representative sample of remarks to stay within token limits and reduce cost
    sample_size = min(30, len(cluster_texts))
    text_sample = "\n".join(cluster_texts[:sample_size])

    # Craft the prompt to guide the AI
    prompt = f"""
    You are an expert at analyzing customer feedback. Your task is to provide a single, concise, and professional category name for a group of similar remarks. The name should be no more than 7 words.

    Remarks for this category:
    ---
    {text_sample}
    ---

    Based on these remarks, the best category name is:
    """

    try:
        # Use the Gemini Pro model
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate the response
        response = model.generate_content(prompt)
        
        # Extract the text and clean it
        name = response.text.strip()
        
        return name
        
    except Exception as e:
        print(f"   [Gen AI Naming] ERROR: API call failed. Falling back to a generic name. Error: {e}")
        
    # Fallback name if the API call fails or returns an empty response
    return "Generic Unnamed Category"

def preprocess_and_embed(remarks_list: list[str], min_doc_frequency: float, device: str):
    """Preprocesses remarks and generates embeddings using a BERT-based SentenceTransformer."""
    print(f"   [Preprocessing] Started for {len(remarks_list)} remarks.")
    if not remarks_list: return [], np.array([])

    vectorizer = TfidfVectorizer(
        min_df=min_doc_frequency, max_df=1.0, ngram_range=(2, 5),
        stop_words=list(UNIVERSAL_STOPWORDS), token_pattern=r'\b[a-zA-Z]{2,}\b'
    )
    try:
        vectorizer.fit(remarks_list)
        boilerplate_phrases = sorted([
            phrase for phrase in vectorizer.vocabulary_
            if sum(1 for remark in remarks_list if phrase in remark.lower()) / len(remarks_list) >= min_doc_frequency
        ], key=len, reverse=True)
        processed_remarks = []
        for remark in remarks_list:
            cleaned_remark = re.sub(r'\s+', ' ', remark.lower()).strip() 
            for bp in boilerplate_phrases:
                cleaned_remark = re.sub(r'\b' + re.escape(bp) + r'\b', ' ', cleaned_remark)
            processed_remarks.append(re.sub(r'\s+', ' ', cleaned_remark).strip())
        print(f"   [Preprocessing] Found {len(boilerplate_phrases)} boilerplate phrases. Finished.")
    except ValueError:
        print("   [Preprocessing] No boilerplate phrases found or error during vectorization. Returning cleaned remarks directly.")
        processed_remarks = [clean_text(r) for r in remarks_list]

    model = SentenceTransformer("bert-base-nli-mean-tokens", device=device)
    embeddings = model.encode(processed_remarks, show_progress_bar=False)
    print(f"   [Preprocessing] Embeddings generated ({embeddings.shape[0]} remarks, {embeddings.shape[1]} dimensions).")
    return processed_remarks, embeddings

def run_hdbscan_and_agglomerate(embeddings: np.ndarray, initial_labels: np.ndarray, max_clusters: int):
    """Performs HDBSCAN and optional Agglomerative Clustering."""
    unique_initial_clusters = sorted([c for c in set(initial_labels) if c != -1])
    final_cluster_labels = np.copy(initial_labels)
    
    if len(unique_initial_clusters) > max_clusters:
        print(f"   [Clustering] {len(unique_initial_clusters)} initial clusters exceed max ({max_clusters}). Agglomerating...")
        cluster_centroids = [np.mean(embeddings[initial_labels == cid], axis=0) for cid in unique_initial_clusters if len(embeddings[initial_labels == cid]) > 0]
        original_cluster_ids_for_agg = [cid for cid in unique_initial_clusters if len(embeddings[initial_labels == cid]) > 0]
        
        agg_n_clusters = min(max_clusters, len(cluster_centroids))
        if agg_n_clusters > 1:
            agg_clusterer = AgglomerativeClustering(n_clusters=agg_n_clusters, metric='euclidean', linkage='ward')
            agg_labels = agg_clusterer.fit_predict(np.array(cluster_centroids))
            
            original_to_new_id_map = {original_id: new_id for original_id, new_id in zip(original_cluster_ids_for_agg, agg_labels)}
            final_cluster_labels = np.array([original_to_new_id_map.get(label, -1) for label in initial_labels])
            print(f"   [Clustering] Agglomeration complete. Final clusters: {len(set(final_cluster_labels)) - (1 if -1 in final_cluster_labels else 0)}.")
        else:
            print("   [Clustering] Not enough centroids for agglomeration or target cluster is 1.")
    return final_cluster_labels, unique_initial_clusters

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


def main():
    excel_file_path = "./Meter.xlsx"
    text_column_name = "REMARKS"
    output_excel_path_wide_format = "../clustered_remarks_named.xlsx" 

    # Set the total number of primary remark clusters to 15.
    # This will result in 17 columns total (15 clusters + 2 other columns).
    max_remark_clusters_limit = 15
    
    max_name_clusters_limit = 5
    hdbscan_min_cluster_size = 2
    hdbscan_min_samples = 2
    assign_noise_to_nearest_cluster = True
    embedding_boilerplate_min_df = 0.8

    print("\n--- Starting Text Clustering and Categorization Script (Gen AI Naming) ---")

def is_semantically_similar(name1: str, name2: str) -> bool:
    """Uses Gemini to determine if two phrases are semantically similar."""
    print(f"   [Gen AI Merging] Checking similarity between '{name1}' and '{name2}'...")
    prompt = f"""
    Are the following two phrases synonyms or do they convey the same meaning?
    Phrase 1: "{name1}"
    Phrase 2: "{name2}"
    Answer with a single word: "YES" or "NO".
    """
    try:
        model = genai.GenerativeModel('models/gemma-3n-e2b-it')
        response = model.generate_content(prompt)
        return response.text.strip().lower() == 'yes'
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

            # Check semantic similarity using the new function
            if is_semantically_similar(col1, col2):
                current_num_columns = len(columns_to_process) - len(merged_mapping)
                if current_num_columns - 1 >= min_columns:
                    print(f"   Merging '{col2}' into '{col1}' (Semantic Match)")
                    merged_mapping[col2] = col1
                else:
                    print(f"   Skipping merge of '{col2}' and '{col1}' to maintain min column count of {min_columns}.")
    
    final_data_dict = defaultdict(list)
    
    for _, row in df.iterrows():
        row_data = {}
        for col in df.columns:
            if col not in merged_mapping:
                row_data[col] = row[col]
            else:
                primary_col = merged_mapping[col]
                if primary_col not in row_data:
                    row_data[primary_col] = row[primary_col]
                
                if pd.notna(row[col]) and row_data[primary_col] != row[col]:
                    if pd.notna(row_data[primary_col]):
                        row_data[primary_col] += " " + row[col]
                    else:
                        row_data[primary_col] = row[col]

        for col, val in row_data.items():
            final_data_dict[col].append(val)
    
    df_merged = pd.DataFrame.from_dict(final_data_dict, orient='index').transpose()

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

            processed_remarks, embeddings = preprocess_and_embed(english_remark_texts, embedding_boilerplate_min_df, "cpu")
            
            hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, min_samples=hdbscan_min_samples, metric='euclidean', prediction_data=True)
            initial_cluster_labels = hdbscan_clusterer.fit_predict(embeddings)
            print(f"   [Clustering] Initial HDBSCAN complete. Found {len(set(initial_cluster_labels)) - (1 if -1 in initial_cluster_labels else 0)} clusters.")

            # --- FIX FOR IndexError ---
            if assign_noise_to_nearest_cluster and -1 in initial_cluster_labels and len(set(initial_cluster_labels)) > 1:
                print(f"   [Noise Handling] Attempting to assign {np.sum(initial_cluster_labels == -1)} noise points.")
                
                # Get the sorted list of unique, non-noise cluster labels
                unique_cluster_labels = sorted([c for c in hdbscan_clusterer.labels_ if c != -1])
                
                # Get the membership probabilities for all points
                probabilities = hdbscan.prediction.membership_vector(hdbscan_clusterer, embeddings)
                
                noise_indices = np.where(initial_cluster_labels == -1)[0]
                
                for i in noise_indices:
                    # Find the index of the highest probability
                    best_prob_idx = np.argmax(probabilities[i, :])
                    
                    # Use this index to find the corresponding original HDBSCAN cluster label
                    best_cluster_label = unique_cluster_labels[best_prob_idx]
                    
                    # Assign the noise point to this cluster
                    initial_cluster_labels[i] = best_cluster_label
                    
                print(f"   [Noise Handling] Reassignment complete. New noise points: {np.sum(initial_cluster_labels == -1)}.")

            final_cluster_labels, unique_initial_clusters = run_hdbscan_and_agglomerate(
                embeddings, initial_cluster_labels, max_remark_clusters_limit
            )

            
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
            

            # --- FIX: Iterate over the final clusters, not the initial ones ---

            final_unique_clusters = sorted([c for c in set(final_cluster_labels) if c != -1])
            print(f"   [Gen AI Naming] Naming {len(final_unique_clusters)} final clusters.")

            used_final_names = set()
            for cluster_id in final_unique_clusters:
                cluster_texts_original = [english_remark_texts[j] for j, label in enumerate(final_cluster_labels) if label == cluster_id]
                proposed_final_name = get_genai_cluster_name(cluster_texts_original)
                
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
