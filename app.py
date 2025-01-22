import os
import dotenv
from q_and_a.build import load_faiss_data, build_faiss_index
from q_and_a.query import query_index
import streamlit as st
from sentence_transformers import SentenceTransformer
import pandas as pd
from sentiment_analysis.predict import predict_single_text
from summarization.summarizer import summarize_text
from sentiment_analysis.report_generator import show_sentiment_report
import hmac

# Load environment variables
dotenv.load_dotenv()

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["username"], os.environ.get("STREAMLIT_USERNAME", "")) and \
           hmac.compare_digest(st.session_state["password"], os.environ.get("STREAMLIT_PASSWORD", "")):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
            del st.session_state["username"]  # Don't store username
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="Conversation Analysis Tool",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stTextArea>div>div>textarea {
        background-color: #f0f2f6;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .css-1v0mbdj.etr89bj1 {
        margin-top: 2em;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1E88E5;
        margin-bottom: 2rem;
    }
    h2 {
        color: #424242;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize constants
MODELS_DIR = os.path.join(os.path.dirname(__file__), "q_and_a", "FAISS_MODELS")

def main():
    if not check_password():
        st.stop()  # Do not continue if check_password is not True.
        
    # Header with gradient background
    st.markdown("""
        <div style='background: linear-gradient(to right, #1E88E5, #4CAF50); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h1 style='color: white; margin: 0;'>Conversation Analysis Tool</h1>
            <p style='color: white; font-size: 1.2em;'>Analyze conversations using AI-powered tools</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs with custom styling
    tabs = st.tabs([
        "💭 Sentiment Analysis",
        "📝 Summarization", 
        "❓ Q&A"
    ])
    
    # Sentiment Analysis Tab
    with tabs[0]:
        st.markdown("<h2>Sentiment Analysis</h2>", unsafe_allow_html=True)
        
        analysis_tab, report_tab, process_tab = st.tabs([
            "✍️ Analyze Text", 
            "📊 Model Report",
            "⚙️ Technical Details"
        ])
        
        with analysis_tab:
            with st.container():
                user_input = st.text_area(
                    "Enter text to analyze:",
                    height=150,
                    placeholder="Type or paste your text here...",
                    help="Enter any text to analyze its sentiment"
                )
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)
                
                if analyze_button and user_input:
                    with st.spinner("Analyzing sentiment..."):
                        try:
                            model_path = os.path.join(os.path.dirname(__file__), 
                                                    "sentiment_analysis", 
                                                    "xgboost_all-MiniLM-L6-v2.pkl")
                            sentiment, confidence = predict_single_text(user_input, model_path)
                            
                            # Results in a nice card
                            st.markdown(f"""
                                <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                                    <h3 style='color: #1E88E5; margin-bottom: 1rem;'>Analysis Results</h3>
                                    <div style='display: flex; justify-content: space-between; margin-bottom: 1rem;'>
                                        <div>
                                            <p style='color: #666; margin-bottom: 0.5rem;'>Detected Sentiment</p>
                                            <h4 style='color: #4CAF50; margin: 0;'>{sentiment}</h4>
                                        </div>
                                        <div>
                                            <p style='color: #666; margin-bottom: 0.5rem;'>Confidence Score</p>
                                            <h4 style='color: #1E88E5; margin: 0;'>{confidence:.2f}</h4>
                                        </div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Emoji display
                            emoji_map = {
                                "Happy": "😊", "Sad": "😢", "Angry": "😠",
                                "Surprised": "😮", "Fearful": "😨", "Disgusted": "🤢",
                                "Curious to dive deeper": "🤔", "Neutral": "😐"
                            }
                            if sentiment in emoji_map:
                                st.markdown(f"<h1 style='text-align: center;'>{emoji_map[sentiment]}</h1>", 
                                          unsafe_allow_html=True)
                                
                        except Exception as e:
                            st.error(f"Error analyzing sentiment: {str(e)}")
                elif analyze_button:
                    st.warning("Please enter some text to analyze.")
        
        with report_tab:
            show_sentiment_report()
        
        with process_tab:
            # Read process.txt file
            process_path = os.path.join(os.path.dirname(__file__), 
                                      "sentiment_analysis", 
                                      "process.txt")
            try:
                with open(process_path, 'r') as f:
                    process_text = f.read()
                st.write(process_text)
            except Exception as e:
                st.error(f"Error reading process details: {str(e)}")
    
    # Summarization Tab
    with tabs[1]:
        st.markdown("<h2>Text Summarization</h2>", unsafe_allow_html=True)
        
        summarize_tab, process_tab, technical_tab = st.tabs([
            "📝 Summarize", 
            "⚙️ Process",
            "🔧 Technical Details"
        ])
        
        with summarize_tab:
            # Existing summaries in a collapsible card
            with st.expander("📚 View Existing Summaries", expanded=False):
                try:
                    summaries_path = os.path.join("summarization", "text_summaries.txt")
                    with open(summaries_path, 'r') as f:
                        summaries = f.read()
                    st.text_area("", value=summaries, height=300, disabled=True)
                except Exception as e:
                    st.warning("No existing summaries found.")
            
            # New summarization
            st.markdown("<h3 style='color: #1E88E5;'>Generate New Summary</h3>", unsafe_allow_html=True)
            user_input = st.text_area(
                "Enter text to summarize:",
                height=200,
                placeholder="Paste your text here...",
                help="Copy text from any source and paste it here"
            )
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                num_sentences = st.slider("Number of sentences in summary:", 1, 10, 3)
            
            if st.button("📝 Generate Summary", use_container_width=True):
                if user_input:
                    with st.spinner("Generating summary..."):
                        summary = summarize_text(user_input, num_sentences=num_sentences)
                        st.success("Summary Generated!")
                        
                        # Display summary in a card
                        st.markdown(f"""
                            <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                                <h3 style='color: #1E88E5; margin-bottom: 1rem;'>Summary</h3>
                                <p style='color: #424242; line-height: 1.6;'>{summary}</p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Please enter some text to summarize.")
        
        with process_tab:
            # Read process.txt file
            process_path = os.path.join(os.path.dirname(__file__), 
                                      "summarization", 
                                      "process.txt")
            try:
                with open(process_path, 'r') as f:
                    process_text = f.read()
                st.write(process_text)
            except Exception as e:
                st.error(f"Error reading process details: {str(e)}")
                
        with technical_tab:
            # Read process.txt file
            process_path = os.path.join(os.path.dirname(__file__), 
                                      "summarization", 
                                      "process.txt")
            try:
                with open(process_path, 'r') as f:
                    process_text = f.read()
                st.write(process_text)
            except Exception as e:
                st.error(f"Error reading process details: {str(e)}")
    
    # Q&A Tab
    with tabs[2]:
        st.markdown("<h2>Question Answering</h2>", unsafe_allow_html=True)
        
        qa_tab, technical_tab = st.tabs([
            "❓ Ask Questions",
            "🔧 Technical Details"
        ])
        
        with qa_tab:
            try:
                available_indexes = [f for f in os.listdir(MODELS_DIR) if f.endswith('.index')]
                if not available_indexes:
                    st.warning("⚠️ No conversation data available.")
                    if st.button("🔨 Build Index"):
                        with st.spinner("Building index..."):
                            try:
                                csv_path = os.path.join(os.path.dirname(__file__), 
                                                      "sample_topical_chat.csv")
                                index, chunks = build_faiss_index(csv_path)
                                st.success("✅ Index built successfully!")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error building index: {str(e)}")
                else:
                    idx = available_indexes[0]
                    params = idx.replace('.index', '').split('_')
                    chunk_size = int(params[1])
                    overlap_size = int(params[3])
                    
                    with st.form(key='query_form'):
                        query = st.text_input(
                            "Your question:",
                            placeholder="Ask anything about the conversations...",
                            help="Type your question and press Enter or click Submit"
                        )
                        submit_button = st.form_submit_button('🔍 Submit', type='primary')
                    
                    if submit_button and query:
                        with st.spinner("Processing query..."):
                            try:
                                index, chunks = load_faiss_data(chunk_size, overlap_size)
                                response_data = query_index(query, index, chunks)
                                
                                # Display response in a card
                                st.markdown(f"""
                                    <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 2rem;'>
                                        <h3 style='color: #1E88E5; margin-bottom: 1rem;'>Answer</h3>
                                        <p style='color: #424242; line-height: 1.6;'>{response_data["answer"]}</p>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Referenced conversations
                                st.markdown("<h3 style='color: #4CAF50;'>Referenced Conversations</h3>", 
                                          unsafe_allow_html=True)
                                for ref in response_data["references"]:
                                    with st.expander(
                                        f"Conversation {ref['conversation_id']} " +
                                        f"(Similarity: {ref['similarity_score']:.4f})"
                                    ):
                                        st.code(ref['text'])
                                
                            except Exception as e:
                                st.error(f"❌ Error occurred: {str(e)}")
                                
            except Exception as e:
                st.error(f"Error accessing FAISS models directory: {str(e)}")
                st.info("Please ensure the FAISS_MODELS directory exists in the q_and_a folder.")
                
        with technical_tab:
            # Read process.txt file
            process_path = os.path.join(os.path.dirname(__file__), 
                                      "q_and_a", 
                                      "process.txt")
            try:
                with open(process_path, 'r') as f:
                    process_text = f.read()
                st.write(process_text)
            except Exception as e:
                st.error(f"Error reading process details: {str(e)}")

if __name__ == "__main__":
    main()