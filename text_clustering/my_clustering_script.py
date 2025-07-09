import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import hdbscan
import re
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from collections import defaultdict, OrderedDict
import hashlib
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords as nltk_stopwords
import nltk
from langdetect import detect, DetectorFactory

# Set seed for reproducibility in language detection
DetectorFactory.seed = 0

# Download NLTK stopwords if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Define universal stopwords for RAKE and TF-IDF
UNIVERSAL_RAKE_STOPWORDS = set(nltk_stopwords.words('english') + [
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'from', 'by', 'this', 'that', 'it', 'its', 'his', 'her', 'their', 'our',
    'what', 'where', 'how', 'why', 'who', 'whom', 'which', 'whether',
    'yesterday', 'today', 'tomorrow', 'morning', 'evening', 'night', 'day', 'days', 'hr', 'hrs', 'hour', 'hours', 'time', 'date', 'week', 'month', 'year', 'ago',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero',
    'consumer', 'customer', 'number', 'no', 'code', 'id', 'location', 'address', 'phone', 'mobile', 'call', 'report', 'registered',
    'ok', 'yes', 'no', 'not', 'hi', 'hello', 'sir', 'madam', 'pls', 'please', 'regards', 'type', 'urban', 'complaint', 'detail', 'general',
    'kv', 'tf', 'na', "service", "request", "feedback", "query", "regarding", "about", "given"
])

# Define words to be removed from final category names
# Removed 'supply' from this list as per user's request.
UNWANTED_NAME_WORDS = UNIVERSAL_RAKE_STOPWORDS.union({
    "problem", "issue", "fault", "category", "item", "uncategorized",
    "detail", "general", "line", "complaint", "output", "summary",
    "generated", "concise", "text", "description", "call", "remark", "remarks",
    "request", "due", "status", "action", "info", "data", "type", "current", "specific", "check", "point",
    "followup", "case", "system", "management", "update", "customer", "account", "last", "coming", "failed",
    "resolution", "resolved", "solving", "fixing", "solution", "inquiry", "query", "asking", "asked", "related", "concerning"
})

def clean_text(text: str) -> str:
    """Removes extra whitespace from a string."""
    return re.sub(r'\s+', ' ', text.strip())

def clean_final_name(name: str) -> str:
    """Cleans and capitalizes a potential category name, ensuring no repeated words.
    Removes digits and special characters. Returns an empty string if no meaningful words remain."""
    name = name.lower()
    words = [word for word in name.split() if word not in UNWANTED_NAME_WORDS]
    
    # Remove duplicate words while maintaining order
    name_words_unique = list(OrderedDict.fromkeys(words))
    name = " ".join(name_words_unique)

    # This is the line that removes digits and special characters
    name = re.sub(r'[^a-zA-Z\s]', '', name).strip()
    name = re.sub(r'\s+', ' ', name).strip() # Consolidate multiple spaces

    common_short_words = {"an", "on", "in", "to", "at", "by", "of", "or", "go", "no", "up", "us", "my", "me", "he", "she", "we", "is", "as", "if", "it", "do"}
    name_words_filtered = [word for word in name.split() if len(word) > 2 or word in common_short_words]
    name = " ".join(name_words_filtered)
    name = re.sub(r'\s+', ' ', name).strip()

    return " ".join([word.capitalize() for word in name.split()]) if name else ""

def get_rake_category_name(consolidated_text: str) -> str:
    """Extracts a category name using RAKE. Returns empty string if no good name found."""
    r = Rake(stopwords=UNIVERSAL_RAKE_STOPWORDS, min_length=1, max_length=5)
    r.extract_keywords_from_text(consolidated_text)
    ranked_phrases_with_scores = r.get_ranked_phrases_with_scores()

    candidate_names = []
    if ranked_phrases_with_scores:
        for score, phrase in ranked_phrases_with_scores:
            if 2 <= len(phrase.split()) <= 6:
                cleaned_candidate = clean_final_name(phrase)
                if cleaned_candidate:
                    candidate_names.append((score, cleaned_candidate))
            if len(candidate_names) >= 5: break
            
        if candidate_names:
            candidate_names.sort(key=lambda x: (x[0], len(x[1].split())), reverse=True)
            return candidate_names[0][1]
    
    if ranked_phrases_with_scores and ranked_phrases_with_scores[0][1]:
        cleaned_top_phrase = clean_final_name(" ".join(ranked_phrases_with_scores[0][1].split()[:6]))
        if cleaned_top_phrase:
            return cleaned_top_phrase
    
    return ""

def get_tfidf_cluster_name(cluster_texts: list[str], target_word_count: int = 8) -> str:
    """Generates a cluster name using TF-IDF, aiming for a specific word count.
    Returns empty string if no good name found."""
    if not cluster_texts:
        return ""

    vectorizer = TfidfVectorizer(
        stop_words=list(UNIVERSAL_RAKE_STOPWORDS), ngram_range=(2, 7),
        max_features=500,
        token_pattern=r'\b[a-zA-Z]{2,}\b'
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(cluster_texts)
        feature_names = vectorizer.get_feature_names_out()
        overall_tfidf_scores = tfidf_matrix.sum(axis=0).A1
        
        top_indices = overall_tfidf_scores.argsort()[::-1] 
        
        selected_terms = []
        seen_words = set()

        for idx in top_indices:
            term = feature_names[idx]
            term_words = term.split()
            
            is_new = True
            for word in term_words:
                if word in seen_words and len(term_words) > 1:
                    is_new = False
                    break
            
            if is_new and len(term) > 1 and not term.isdigit() and not all(c.isdigit() for c in term.replace('.', '')) \
               and term.lower() not in UNIVERSAL_RAKE_STOPWORDS: 
                
                selected_terms.append(term)
                for word in term_words:
                    seen_words.add(word)
                
                current_word_count = len(" ".join(selected_terms).split())
                if current_word_count >= target_word_count:
                    break

        if selected_terms:
            proposed_name = " ".join(selected_terms)
            final_name = clean_final_name(proposed_name)
            
            if final_name and len(final_name.split()) >= (target_word_count // 2): # At least half the target words
                return final_name
            
    except ValueError:
        pass

    print(f"  [TF-IDF Naming] Failed to generate {target_word_count} meaningful terms. Falling back to RAKE.")
    rake_name = get_rake_category_name(' '.join(cluster_texts))
    if rake_name:
        return rake_name
    
    return ""


def preprocess_and_embed(remarks_list: list[str], min_doc_frequency: float, device: str):
    """Preprocesses remarks and generates embeddings."""
    print(f"  [Preprocessing] Started for {len(remarks_list)} remarks.")
    if not remarks_list: return [], np.array([])

    vectorizer = TfidfVectorizer(
        min_df=min_doc_frequency, max_df=1.0, ngram_range=(2, 5),
        stop_words=list(UNIVERSAL_RAKE_STOPWORDS), token_pattern=r'\b[a-zA-Z]{2,}\b'
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
        print(f"  [Preprocessing] Found {len(boilerplate_phrases)} boilerplate phrases. Finished.")
    except ValueError:
        print("  [Preprocessing] No boilerplate phrases found or error during vectorization. Returning cleaned remarks directly.")
        processed_remarks = [clean_text(r) for r in remarks_list]

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
    embeddings = model.encode(processed_remarks, show_progress_bar=False)
    print(f"  [Preprocessing] Embeddings generated ({embeddings.shape[0]} remarks, {embeddings.shape[1]} dimensions).")
    return processed_remarks, embeddings

def run_hdbscan_and_agglomerate(embeddings: np.ndarray, initial_labels: np.ndarray, max_clusters: int):
    """Performs HDBSCAN and optional Agglomerative Clustering."""
    unique_initial_clusters = sorted([c for c in set(initial_labels) if c != -1])
    final_cluster_labels = np.copy(initial_labels)
    
    if len(unique_initial_clusters) > max_clusters:
        print(f"  [Clustering] {len(unique_initial_clusters)} initial clusters exceed max ({max_clusters}). Agglomerating...")
        cluster_centroids = [np.mean(embeddings[initial_labels == cid], axis=0) for cid in unique_initial_clusters if len(embeddings[initial_labels == cid]) > 0]
        original_cluster_ids_for_agg = [cid for cid in unique_initial_clusters if len(embeddings[initial_labels == cid]) > 0]
        
        agg_n_clusters = min(max_clusters, len(cluster_centroids))
        if agg_n_clusters > 1:
            agg_clusterer = AgglomerativeClustering(n_clusters=agg_n_clusters, metric='euclidean', linkage='ward')
            agg_labels = agg_clusterer.fit_predict(np.array(cluster_centroids))
            
            original_to_new_id_map = {original_id: new_id for original_id, new_id in zip(original_cluster_ids_for_agg, agg_labels)}
            final_cluster_labels = np.array([original_to_new_id_map.get(label, -1) for label in initial_labels])
            print(f"  [Clustering] Agglomeration complete. Final clusters: {len(set(final_cluster_labels)) - (1 if -1 in final_cluster_labels else 0)}.")
        else:
            print("  [Clustering] Not enough centroids for agglomeration or target cluster is 1.")
    return final_cluster_labels, unique_initial_clusters

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
            (english_remarks_with_indices if detected_lang == 'en' else other_remarks_with_indices).append((i, remark))
        except Exception: # Catch any langdetect errors
            other_remarks_with_indices.append((i, remark))
    print(f"Segregation complete. English: {len(english_remarks_with_indices)}, Other: {len(other_remarks_with_indices)}")
    return english_remarks_with_indices, other_remarks_with_indices

def get_unique_name(base_name: str, existing_names: set, suffix_identifier: str = "") -> str:
    """Generates a unique name by adding an alphabetic/numerical suffix if necessary,
       ensuring no special characters or non-sequential numbers."""
    name = base_name
    # Clean the base name once more to ensure it's purely alphanumeric before suffixing
    name = re.sub(r'[^a-zA-Z\s]', '', name).strip()
    name = re.sub(r'\s+', ' ', name).strip()
    
    # If cleaning results in an empty name, default to "Generic Category" for suffixing
    if not name:
        name = "Generic Category"

    alpha_suffix_idx = 0
    numeric_suffix_idx = 0
    
    original_base = name # Store original clean base for consistent suffixing

    while name.lower() in existing_names: 
        if alpha_suffix_idx < 26: # Try A, B, C...
            name = f"{original_base} {chr(65 + alpha_suffix_idx)}"
            alpha_suffix_idx += 1
        else: # After Z, combine with numbers: A1, B1, ... Z1, A2, B2...
            numeric_suffix_idx += 1
            alpha_suffix_idx_for_num = (alpha_suffix_idx - 26) % 26 # Reset alphabetical part for new number
            name = f"{original_base} {chr(65 + alpha_suffix_idx_for_num)}{numeric_suffix_idx}"
            alpha_suffix_idx += 1 # Continue incrementing to ensure we eventually move to next number
            
    return name

def main():
    excel_file_path = "./Book1.xlsx"
    text_column_name = "REMARKS"
    output_excel_path_wide_format = "../clustered_remarks_named.xlsx" # Changed output file name for clarity 
    max_remark_clusters_limit = 10
    max_name_clusters_limit = 5 
    target_column_name_words = 7

    print("\n--- Starting Text Clustering and Categorization Script (TF-IDF Naming - NO EXTERNAL APIs) ---")
    try:
        raw_remarks_list, _ = load_excel_file(excel_file_path, text_column_name) 
        english_remarks_w_indices, other_remarks_w_indices = segregate_remarks_by_language(raw_remarks_list)
        
        english_remark_texts = [r_text for _, r_text in english_remarks_w_indices]
        english_remark_original_indices = [original_idx for original_idx, _ in english_remarks_w_indices]

        original_indexed_cluster_labels = np.full(len(raw_remarks_list), -2, dtype=int) 
        final_column_name_map = {}
        unique_initial_summaries = [] 
        summary_to_remark_cluster_ids = defaultdict(list)
        
        if english_remark_texts:
            print("\n--- Processing English Remarks for Clustering ---")
            processed_remarks, embeddings = preprocess_and_embed(english_remark_texts, 0.8, "cpu")
            hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, metric='euclidean', prediction_data=True)
            initial_cluster_labels = hdbscan_clusterer.fit_predict(embeddings)
            print(f"  [Clustering] Initial HDBSCAN complete. Found {len(set(initial_cluster_labels)) - (1 if -1 in initial_cluster_labels else 0)} clusters.")

            final_cluster_labels, unique_remark_clusters = run_hdbscan_and_agglomerate(
                embeddings, initial_cluster_labels, max_remark_clusters_limit
            )
            
            for i, clustered_label in enumerate(final_cluster_labels):
                original_indexed_cluster_labels[english_remark_original_indices[i]] = clustered_label

            used_temp_names = set()
            for cluster_id in unique_remark_clusters:
                cluster_texts_processed = [processed_remarks[j] for j, label in enumerate(final_cluster_labels) if label == cluster_id]
                
                proposed_name = get_tfidf_cluster_name(cluster_texts_processed, target_word_count=target_column_name_words)
                
                # If TF-IDF/RAKE returned an empty string, assign a generic name with category identifier
                if not proposed_name: 
                    proposed_name = f"Generic Remark Category For Cluster {cluster_id}" 
                
                unique_name = get_unique_name(proposed_name, used_temp_names, str(cluster_id)) 
                unique_initial_summaries.append(unique_name)
                used_temp_names.add(unique_name.lower())
                summary_to_remark_cluster_ids[unique_name].append(cluster_id)
            print(f"\nGenerated {len(unique_initial_summaries)} initial cluster summaries (TF-IDF).")
            
            if unique_initial_summaries:
                print("\n--- Clustering Initial Category Names (TF-IDF based) ---")
                name_model_for_clustering = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu") 
                name_embeddings = name_model_for_clustering.encode(unique_initial_summaries, show_progress_bar=False)
                
                n_name_clusters_for_agg = min(max_name_clusters_limit, len(unique_initial_summaries))
                if n_name_clusters_for_agg < 2 and len(unique_initial_summaries) > 1: n_name_clusters_for_agg = 2
                elif n_name_clusters_for_agg == 0: n_name_clusters_for_agg = 1

                name_cluster_labels = AgglomerativeClustering(n_clusters=n_name_clusters_for_agg, metric='euclidean', linkage='ward').fit_predict(name_embeddings)
                print(f"Clustered initial names into {len(set(name_cluster_labels))} final groups.")
                
                print("\n--- Finalizing Column Names based on Merged Clusters (TF-IDF Final Naming) ---")
                used_final_names = set()
                for name_cluster_id in sorted(list(set(name_cluster_labels))):
                    remarks_for_merged_category_processed = []
                    for temp_summary_name in [unique_initial_summaries[i] for i, label in enumerate(name_cluster_labels) if label == name_cluster_id]:
                        for remark_cluster_id in summary_to_remark_cluster_ids.get(temp_summary_name, []):
                            remarks_for_merged_category_processed.extend([
                                processed_remarks[j] for j, label in enumerate(final_cluster_labels) if label == remark_cluster_id
                            ])
                    
                    proposed_final_name = get_tfidf_cluster_name(remarks_for_merged_category_processed, target_word_count=target_column_name_words)
                    
                    if not proposed_final_name:
                        proposed_final_name = f"Merged Generic Category Type For Cluster {name_cluster_id}"
                    
                    final_name = get_unique_name(proposed_final_name, used_final_names, str(name_cluster_id))
                    
                    final_column_name_map[name_cluster_id] = final_name
                    used_final_names.add(final_name.lower()) 
                    print(f"  Final Category Name: '{final_name}'")
            else:
                print("No initial English summaries were generated for further name clustering.")
        else:
            print("No English remarks were found in the input data. Skipping English remark clustering.")
        
        final_wide_data_columns = {}
        for name_cluster_id, final_col_name in final_column_name_map.items():
            remarks_in_column = []
            if english_remark_texts:
                for temp_summary_name in [unique_initial_summaries[i] for i, label in enumerate(name_cluster_labels) if label == name_cluster_id]:
                    for remark_cluster_id in summary_to_remark_cluster_ids.get(temp_summary_name, []):
                        remarks_in_column.extend([
                            raw_remarks_list[idx] for idx, label in enumerate(original_indexed_cluster_labels) if label == remark_cluster_id
                        ])
            final_wide_data_columns[final_col_name] = remarks_in_column

        # Handle uncategorized English remarks (HDBSCAN noise)
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

        # Pad columns to equal length for DataFrame creation
        max_col_len = max((len(col_data) for col_data in final_wide_data_columns.values()), default=0)
        for category_name, remarks_list_for_col in final_wide_data_columns.items():
            if len(remarks_list_for_col) < max_col_len:
                final_wide_data_columns[category_name].extend([np.nan] * (max_col_len - len(remarks_list_for_col)))
        
        df_results_wide = pd.DataFrame(final_wide_data_columns)
        
        print("\n--- Script Execution Complete ---")
        save_results(df_results_wide, output_excel_path_wide_format)
        print("\n--- Sample Results (Wide Format) ---")
        print(df_results_wide.head())

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