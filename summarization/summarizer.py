import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from heapq import nlargest
from typing import Optional

class TextSummarizer:
    def __init__(self):
        """Initialize the summarizer and download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text: str) -> tuple:
        """
        Preprocess the text for summarization
        
        Args:
            text: Input text to preprocess
            
        Returns:
            tuple: (sentences, word_freq) where sentences is list of sentences
                  and word_freq is frequency distribution of words
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # Tokenize words and remove stopwords
        word_tokens = word_tokenize(text.lower())
        word_tokens = [word for word in word_tokens 
                      if word.isalnum() and word not in self.stop_words]
        
        # Calculate word frequencies
        word_freq = FreqDist(word_tokens)
        
        return sentences, word_freq
    
    def score_sentences(self, sentences: list, word_freq: FreqDist) -> dict:
        """
        Score sentences based on word frequencies
        
        Args:
            sentences: List of sentences
            word_freq: Frequency distribution of words
            
        Returns:
            dict: Mapping of sentences to their scores
        """
        sentence_scores = {}
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            word_count = len([word for word in words if word.isalnum()])
            
            if word_count == 0:
                continue
                
            for word in words:
                if word in word_freq:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_freq[word]
                    else:
                        sentence_scores[sentence] += word_freq[word]
            
            # Normalize by sentence length
            sentence_scores[sentence] = sentence_scores[sentence] / word_count
            
        return sentence_scores
    
    def summarize(self, text: str, num_sentences: Optional[int] = None, ratio: Optional[float] = None) -> str:
        """
        Generate a summary of the input text
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary (overrides ratio if provided)
            ratio: Fraction of original sentences to keep (default: 0.3)
            
        Returns:
            str: Summarized text
        """
        try:
            # Handle empty or invalid input
            if not text or not isinstance(text, str):
                raise ValueError("Input must be a non-empty string")
            
            # Preprocess text
            sentences, word_freq = self.preprocess_text(text)
            
            if len(sentences) == 0:
                return text
            
            # Score sentences
            sentence_scores = self.score_sentences(sentences, word_freq)
            
            # Determine number of sentences for summary
            if num_sentences is None:
                ratio = ratio or 0.3
                num_sentences = max(1, int(len(sentences) * ratio))
            else:
                num_sentences = min(num_sentences, len(sentences))
            
            # Select top sentences
            summary_sentences = nlargest(num_sentences, 
                                      sentence_scores, 
                                      key=sentence_scores.get)
            
            # Reorder sentences to maintain original flow
            summary_sentences.sort(key=sentences.index)
            
            # Join sentences
            summary = ' '.join(summary_sentences)
            
            return summary
            
        except Exception as e:
            print(f"Error in summarization: {str(e)}")
            return text

def summarize_text(text: str, num_sentences: Optional[int] = None, ratio: Optional[float] = None) -> str:
    """
    Wrapper function for text summarization
    
    Args:
        text: Input text to summarize
        num_sentences: Number of sentences in summary
        ratio: Fraction of original sentences to keep
        
    Returns:
        str: Summarized text
    """
    summarizer = TextSummarizer()
    return summarizer.summarize(text, num_sentences, ratio)

if __name__ == "__main__":
    # Test the summarizer
    test_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.

    The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.

    AI applications include advanced web search engines (e.g., Google Search), recommendation systems (used by YouTube, Amazon, and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Tesla), automated decision-making, and competing at the highest level in strategic game systems (such as chess and Go).
    """
    
    # Test with number of sentences
    summary1 = summarize_text(test_text, num_sentences=2)
    print("Summary (2 sentences):")
    print(summary1)
    print("\n")
    
    # Test with ratio
    summary2 = summarize_text(test_text, ratio=0.5)
    print("Summary (50% of original):")
    print(summary2) 