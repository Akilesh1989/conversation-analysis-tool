import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from heapq import nlargest
import pandas as pd

# Explicitly set NLTK data path
NLTK_DATA_PATH = '/Users/akilesh/nltk_data'
nltk.data.path.append(NLTK_DATA_PATH)

def simple_sentence_tokenize(text):
    """Fallback sentence tokenizer using simple rules"""
    # Split on common sentence endings
    sentences = []
    current = []
    
    # Split on potential sentence boundaries
    for word in text.replace('?', '.').replace('!', '.').split('.'):
        word = word.strip()
        if word:
            if not current:
                current.append(word)
            else:
                current.append(word)
                sentences.append(' '.join(current))
                current = []
    
    if current:
        sentences.append(' '.join(current))
    
    return sentences

def summarize_text(text, num_sentences=3):
    try:
        # Try NLTK tokenizer first
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            print(f"NLTK tokenizer failed, using simple tokenizer: {str(e)}")
            sentences = simple_sentence_tokenize(text)
        
        if not sentences:
            return "Could not generate summary: no sentences found."
        
        # Tokenize words and remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
        except:
            # Fallback to empty set if stopwords fail to load
            stop_words = set()
        
        # Simple word tokenization as fallback
        word_tokens = text.lower().split()
        word_tokens = [word for word in word_tokens if word.isalnum() and word not in stop_words]
        
        # Calculate word frequencies
        freq_dist = FreqDist(word_tokens)
        
        # Calculate sentence scores based on word frequencies
        sentence_scores = {}
        for sentence in sentences:
            words = sentence.lower().split()
            for word in words:
                if word in freq_dist:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = freq_dist[word]
                    else:
                        sentence_scores[sentence] += freq_dist[word]
        
        # Get the top N sentences with highest scores
        summary_sentences = nlargest(min(num_sentences, len(sentences)), 
                                   sentence_scores, 
                                   key=sentence_scores.get)
        
        # Join sentences and return summary
        summary = ' '.join(summary_sentences)
        return summary
    except Exception as e:
        print(f"Error in summarization: {str(e)}")
        return "Error generating summary for this conversation."

def process_conversations(df, output_file):
    """Process conversations and write summaries to file"""
    try:
        # Group messages by conversation_id
        conversations = df.groupby('conversation_id')['message'].apply(' '.join).reset_index()
        
        # Process each conversation and write summaries
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in conversations.iterrows():
                conversation_id = row['conversation_id']
                messages = row['message']
                summary = summarize_text(messages)
                f.write(f"Conversation {conversation_id}:\n{summary}\n\n")
                print(f"Processed conversation {conversation_id}")
                
        print(f"Summaries written to {output_file}")
        
    except Exception as e:
        print(f"Error processing conversations: {str(e)}")
        raise

def main():
    try:
        # Specify file paths
        input_file = "assignment_details/sample_topical_chat.csv"
        output_file = "summarization/text_summaries.txt"
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Read the CSV file
        print("Reading input file...")
        df = pd.read_csv(input_file)
        
        # Process conversations and generate summaries
        print("Generating summaries...")
        process_conversations(df, output_file)
        
        print("Summary generation completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
