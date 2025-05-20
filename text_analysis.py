import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys
import re # Import regular expression module for potential advanced cleaning

# --- Download required NLTK data ---
# Downloads 'punkt' for tokenization and 'stopwords' for filtering common words.
# quiet=True suppresses output if already downloaded.
print("Checking/Downloading required NLTK data...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("NLTK data is available.")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    print("Please check your internet connection or NLTK setup.")
    # sys.exit(1) # Optional: Exit if NLTK data is critical

# --- Function Definitions ---

def load_json_file(filename):
    """Load a JSON file containing a list of objects."""
    try:
        print(f"Attempting to load JSON file: {filename}...")
        with open(filename, 'r', encoding='utf-8') as f:
            # Use json.load for standard JSON array file [{}, {}]
            data = json.load(f)
        print(f"Successfully loaded {len(data)} records from {filename}")
        return data
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except json.JSONDecodeError as jde:
        print(f"Error: The file '{filename}' is not valid JSON. Please check the format.")
        print(f"Details: {jde}")
        # Attempt to load as JSON Lines (objects separated by newlines) as a fallback
        try:
            print(f"Attempting to load {filename} as JSON Lines format...")
            data = []
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line: # Avoid empty lines
                        data.append(json.loads(line))
            print(f"Successfully loaded {len(data)} records from {filename} using JSON Lines format.")
            return data
        except Exception as e:
             print(f"Failed to load as JSON Lines format as well. Error: {e}")
             return None
    except Exception as e:
        print(f"An unexpected error occurred while loading '{filename}': {e}")
        return None

def analyze_basic_stats(df):
    """Performs and plots basic analysis on message statistics (sender, intent)."""
    if df is None or df.empty:
        print("DataFrame is empty or None. Skipping basic stats analysis.")
        return df

    print("\n--- Analyzing Basic Message Statistics ---")
    print(f"Total messages loaded: {len(df)}")

    # Sender analysis
    if 'sender' in df.columns:
        print("\nDistribution of messages by sender:")
        sender_counts = df['sender'].value_counts()
        sender_pcts = df['sender'].value_counts(normalize=True) * 100
        sender_summary = pd.DataFrame({'count': sender_counts, 'percentage': sender_pcts.round(1)})
        print(sender_summary)
    else:
        print("\nWarning: 'sender' column not found. Skipping sender analysis.")

    # Intent analysis
    if 'intent' in df.columns:
        print("\nDistribution of messages by intent:")
        intent_counts = df['intent'].value_counts()
        intent_pcts = df['intent'].value_counts(normalize=True) * 100
        intent_summary = pd.DataFrame({'count': intent_counts, 'percentage': intent_pcts.round(1)})
        print(intent_summary)

        # Plot intent distribution if there are intents to plot
        if not intent_pcts.empty:
            plt.figure(figsize=(12, max(6, len(intent_counts) * 0.5)))
            sns.barplot(x=intent_pcts.values, y=intent_pcts.index, palette="viridis")
            plt.title('Distribution of Messages by Intent')
            plt.xlabel('Percentage of Messages (%)')
            plt.ylabel('Intent')
            for i, v in enumerate(intent_pcts):
                plt.text(v + 0.5, i, f'{v:.1f}%', va='center')
            plt.xlim(0, max(intent_pcts.values) * 1.1 if intent_pcts.values.size > 0 else 10) # Adjust x-limit
            plt.tight_layout()
            plt.show()
        else:
            print("No intent data found to plot.")
    else:
        print("\nWarning: 'intent' column not found. Skipping intent analysis.")

    # Optional: Analyze Date/Time if 'createdAt' exists and is parsed correctly
    if 'createdAt' in df.columns:
        try:
            # Attempt to convert the nested date structure to datetime
            # Handle both string dates and the {"$date": ...} structure
            def extract_date(x):
                if isinstance(x, dict) and '$date' in x:
                    return x['$date']
                return x # Assume it might already be a parsable string

            df['timestamp'] = pd.to_datetime(df['createdAt'].apply(extract_date), errors='coerce')
            # Drop rows where conversion failed
            df.dropna(subset=['timestamp'], inplace=True)

            if not df['timestamp'].empty:
                print("\nAnalyzing message timestamps...")
                # Example: Plot messages over time (e.g., per day)
                plt.figure(figsize=(12, 6))
                df['timestamp'].dt.floor('D').value_counts().sort_index().plot(kind='line', marker='o')
                plt.title('Number of Messages Per Day')
                plt.xlabel('Date')
                plt.ylabel('Number of Messages')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            else:
                print("Could not extract valid timestamps from 'createdAt' column.")

        except Exception as e:
            print(f"\nWarning: Could not process 'createdAt' column for time analysis. Error: {e}")


    return df # Return the dataframe

def analyze_content(df):
    """Performs word frequency and sentiment analysis on the 'content' column."""
    if df is None or df.empty:
        print("DataFrame is empty or None. Skipping content analysis.")
        return df

    if 'content' not in df.columns:
        print("Warning: 'content' column not found. Skipping content analysis.")
        return df

    print("\n--- Analyzing Message Content ---")

    # Ensure 'content' is string, fill missing values
    df['content'] = df['content'].fillna('').astype(str)
    print("Checked 'content' column, filled potential missing values.")

    # --- Word Frequency Analysis ---
    try:
        print("\nStarting Word Frequency Analysis...")
        # Define stopwords
        stop_words = set(stopwords.words('english'))
        # More comprehensive custom stopwords - add domain-specific words if needed
        custom_stopwords = {
            '', ' ', 'also', 'would', 'could', 'should', 'might', 'may', 'like', 'im',
            'know', 'get', 'dont', 'cant', 'well', 'us', 'one', 'see', 'use', 'need',
            'want', 'make', 'go', 'com', # Often from URLs if not cleaned
            # Consider adding words specific to your chat context if they are noise
            'salary', 'range', 'market', 'expectations', # From your example, if too frequent
            'engineer', 'software', 'los', 'angeles' # From your example
            }
        stop_words.update(custom_stopwords)
        print(f"Using {len(stop_words)} stopwords (English default + custom).")

        # Define text processing function
        def process_text(text):
            """Cleans, tokenizes, and filters text."""
            try:
                # 1. Lowercase
                text = text.lower()
                # 2. Remove URLs (optional but recommended)
                text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
                # 3. Remove email addresses (optional)
                text = re.sub(r'\S+@\S+', '', text)
                # 4. Remove numbers / amounts with symbols (e.g., $146,079) - keep words
                text = re.sub(r'[\$\d,\.]+', '', text) # Removes monetary values, numbers with commas/dots
                # 5. Remove punctuation and non-alphanumeric (keep spaces)
                text_cleaned = ''.join(char for char in text if char.isalnum() or char.isspace())
                # 6. Tokenize
                tokens = word_tokenize(text_cleaned)
                # 7. Filter tokens
                filtered_tokens = [
                    word for word in tokens
                    if word.isalnum()           # Ensure purely alphanumeric
                       and word not in stop_words # Check against stopwords
                       and len(word) > 2          # Keep words longer than 2 chars
                       # 'isnumeric' check is less needed after step 4, but keep as safeguard
                       and not word.isnumeric()
                ]
                return filtered_tokens
            except Exception as e:
                # print(f"Error processing text chunk: '{text[:50]}...': {e}")
                return []

        # --- Debugging Print Statements (Uncomment ONE line inside process_text if needed) ---
        # print(f"Original: '{text[:100]}'")
        # print(f"Cleaned: '{text_cleaned[:100]}'")
        # print(f"Tokens: {tokens[:20]}")
        # print(f"Filtered Tokens: {filtered_tokens[:20]}")
        # print("-" * 20)
        # --- End Debugging ---


        # Process all messages
        print("Processing texts (tokenizing, cleaning, filtering)...")
        processed_tokens_list = df['content'].apply(process_text)
        all_words = [word for sublist in processed_tokens_list for word in sublist]
        print(f"Found {len(all_words)} words after processing and filtering.") # THIS IS THE KEY OUTPUT TO CHECK

        if not all_words:
            print("No words were found after processing all messages. Skipping frequency analysis and plot.")
        else:
            # Calculate word frequencies
            fdist = FreqDist(all_words)
            total_processed_words = sum(fdist.values())
            print(f"Total unique words found: {len(fdist)}")

            if total_processed_words == 0:
                print("Word frequency distribution is empty. Skipping frequency plot.")
            else:
                # Plotting Top N Words
                top_words_count = 20
                common_words = fdist.most_common(top_words_count)

                if not common_words:
                    print("No common words found to plot.")
                else:
                    top_words_df = pd.DataFrame(common_words, columns=['word', 'count'])
                    top_words_df['percentage'] = (top_words_df['count'] / total_processed_words) * 100

                    plt.figure(figsize=(12, 8))
                    top_words_df = top_words_df.sort_values('percentage', ascending=True)
                    bars = plt.barh(top_words_df['word'], top_words_df['percentage'], color='skyblue')

                    plt.title(f'Top {top_words_count} Most Frequent Words (% of Total Processed Words)')
                    plt.xlabel('Percentage (%)')
                    plt.ylabel('Words')
                    for bar in bars:
                        width = bar.get_width()
                        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                                f'{width:.1f}%', ha='left', va='center')
                    plt.xlim(0, top_words_df['percentage'].max() * 1.15)
                    plt.tight_layout()
                    plt.show()

                    print("\nTop 10 most frequent words:")
                    for word, count in fdist.most_common(10):
                        percentage = (count / total_processed_words) * 100
                        print(f"- {word}: {count} times ({percentage:.1f}%)")

    except Exception as e:
        print(f"\nAn error occurred during Word Frequency Analysis: {e}")
        import traceback
        traceback.print_exc()

    # --- Sentiment Analysis ---
    try:
        print("\nStarting Sentiment Analysis using VADER...")
        analyzer = SentimentIntensityAnalyzer()

        def get_sentiment_score(text):
            """Calculates VADER compound sentiment score. Returns 0.0 on error."""
            try:
                return analyzer.polarity_scores(str(text))['compound']
            except Exception as e:
                # print(f"Error getting sentiment for text: '{text[:50]}...': {e}")
                return 0.0

        # Apply sentiment analysis
        df['sentiment'] = df['content'].apply(get_sentiment_score)
        print("Sentiment scores calculated and added as 'sentiment' column.")

        # Plot Sentiment Distribution
        if not df['sentiment'].empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['sentiment'], bins=30, kde=True, color='lightcoral')
            plt.title('Distribution of Message Sentiment Scores')
            plt.xlabel('Sentiment Score (VADER Compound: -1 Negative to +1 Positive)')
            plt.ylabel('Number of Messages')
            plt.axvline(x=-0.05, color='red', linestyle='--', alpha=0.7, label='Negative Threshold (-0.05)')
            plt.axvline(x=0.05, color='green', linestyle='--', alpha=0.7, label='Positive Threshold (0.05)')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()

            # Print overall sentiment statistics
            print("\nOverall Sentiment Distribution:")
            negative_pct = (df['sentiment'] < -0.05).mean() * 100
            neutral_pct = ((df['sentiment'] >= -0.05) & (df['sentiment'] <= 0.05)).mean() * 100
            positive_pct = (df['sentiment'] > 0.05).mean() * 100
            print(f"- Negative (< -0.05): {negative_pct:.1f}%")
            print(f"- Neutral [-0.05 to 0.05]: {neutral_pct:.1f}%")
            print(f"- Positive (> 0.05): {positive_pct:.1f}%")

            print("\nSentiment Score Statistics:")
            print(df['sentiment'].describe().round(3))
        else:
            print("No sentiment scores generated to plot or analyze.")

    except Exception as e:
        print(f"\nAn error occurred during Sentiment Analysis: {e}")
        import traceback
        traceback.print_exc()

    return df # Return the dataframe with 'sentiment' column (and potentially 'timestamp')

# --- Main Execution Block ---
if __name__ == "__main__":
    print("=============================================")
    print(" Starting Chat Message Analysis Script ")
    print("=============================================")

    # !!! --- IMPORTANT: SET YOUR JSON FILENAME HERE --- !!!
    json_filename = 'Parallel-Prod.AssistMessage.json'  # <--- CHANGE THIS TO YOUR ACTUAL FILENAME
    # !!! -------------------------------------------- !!!

    # Step 1: Load the data
    raw_data = load_json_file(json_filename)

    # Proceed only if data loading was successful
    if raw_data is not None and isinstance(raw_data, list) and len(raw_data) > 0:
        # Step 2: Convert raw data to Pandas DataFrame
        try:
            # pd.DataFrame handles list of dicts directly.
            # Nested dicts (like _id, createdAt) become object columns.
            df = pd.DataFrame(raw_data)
            print(f"\nData successfully converted to DataFrame.")
            print(f"DataFrame shape: {df.shape[0]} rows, {df.shape[1]} columns")
            print("Columns found:", df.columns.tolist())
            # print("\nDataFrame head:\n", df.head()) # Uncomment to see first few rows

            # Optional: Flatten nested structures if needed for specific analysis
            # Example: Flatten '_id' into 'id'
            # try:
            #     df['id'] = df['_id'].apply(lambda x: x['$oid'] if isinstance(x, dict) and '$oid' in x else None)
            #     # df = df.drop(columns=['_id']) # Optionally drop original nested column
            #     print("Extracted '_id.$oid' into 'id' column.")
            # except Exception as e:
            #     print(f"Could not flatten '_id' column: {e}")

            # Step 3: Perform analyses
            df = analyze_basic_stats(df) # Basic sender, intent, time analysis
            df = analyze_content(df)     # Word frequency and sentiment on 'content'

            print("\n--- Analysis Complete ---")

            # Optional: Save the analyzed DataFrame
            # output_filename = 'analyzed_chat_data.csv'
            # try:
            #    df.to_csv(output_filename, index=False, encoding='utf-8')
            #    print(f"\nAnalyzed DataFrame saved successfully to {output_filename}")
            # except Exception as e:
            #    print(f"\nError saving DataFrame to CSV '{output_filename}': {e}")

        # Handle errors during DataFrame creation or analysis phases
        except ValueError as ve:
            print(f"\nError creating DataFrame: {ve}.")
            print("Check if the JSON structure is a list of records (dictionaries).")
        except KeyError as ke:
            print(f"\nError accessing key during analysis: {ke}. Ensure required columns (e.g., 'content', 'sender', 'intent') exist or handle their absence.")
        except Exception as e:
            print(f"\nAn unexpected error occurred during DataFrame conversion or analysis: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback

    elif raw_data is None:
        print("\nExiting script because the JSON data could not be loaded.")
    else:
        print("\nLoaded data is not in the expected format (list of records). Exiting.")


    print("\n=============================================")
    print(" Script Finished ")
    print("=============================================")

