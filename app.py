import streamlit as st
import os
from typing import List, Dict
from groq import Groq
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import chromadb
from chromadb.utils import embedding_functions
import tempfile
import shutil

class H1BEvaluationSystem:
    def __init__(self, groq_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Create a new temporary directory for each session
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_client = chromadb.PersistentClient(path=self.temp_dir)
        
        # Use sentence-transformers for embeddings
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create a new collection
        self.collection = self.chroma_client.create_collection(
            name="h1b_documents",
            embedding_function=self.embedding_function
        )
        
    def cleanup(self):
        """Clean up the temporary directory"""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text content from uploaded PDF file."""
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text.strip()
    
    def process_documents(self, resume_file, letter_file):
        """Process resume and sponsorship letter documents."""
        try:
            # Extract text from documents
            resume_text = self.extract_text_from_pdf(resume_file)
            letter_text = self.extract_text_from_pdf(letter_file)
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Process resume chunks
            resume_chunks = text_splitter.split_text(resume_text)
            resume_ids = [f"resume_chunk_{i}" for i in range(len(resume_chunks))]
            resume_metadatas = [{"source": "resume", "text": chunk} for chunk in resume_chunks]
            
            # Process letter chunks
            letter_chunks = text_splitter.split_text(letter_text)
            letter_ids = [f"letter_chunk_{i}" for i in range(len(letter_chunks))]
            letter_metadatas = [{"source": "sponsorship_letter", "text": chunk} for chunk in letter_chunks]
            
            # Combine all documents
            all_chunks = resume_chunks + letter_chunks
            all_ids = resume_ids + letter_ids
            all_metadatas = resume_metadatas + letter_metadatas
            
            # Add documents in batches
            batch_size = 50
            for i in range(0, len(all_chunks), batch_size):
                end_idx = min(i + batch_size, len(all_chunks))
                self.collection.add(
                    documents=all_chunks[i:end_idx],
                    ids=all_ids[i:end_idx],
                    metadatas=all_metadatas[i:end_idx]
                )
            
            return True
            
        except Exception as e:
            raise Exception(f"Error processing documents: {str(e)}")
    
    def evaluate_candidate(self) -> Dict:
        """Evaluate the candidate based on stored documents."""
        try:
            evaluation_prompt = """
            You are an experienced immigration officer evaluating an H-1B visa sponsorship application.
            Review the provided resume and sponsorship letter focusing on key eligibility criteria.
            
            Focus your evaluation on these essential H-1B requirements:
            
            1. Educational Requirements:
               - Bachelor's degree or higher in a related field
               - Field of study relevance to position
            
            2. Position Requirements:
               - Qualifies as a "specialty occupation"
               - Requires specialized knowledge
               - Complexity level appropriate for H-1B
            
            3. Experience and Skills:
               - Relevant technical/professional skills
               - Experience level matches position
               - Current market demand for skills
            
            4. Sponsorship Justification:
               - Clear business need
               - Position requirements
               - Wage level appropriateness
            
            Context from documents:
            {context}
            
            Provide a balanced evaluation following this format:
            
            QUALIFICATION SUMMARY:
            - Educational background analysis
            - Degree relevance to position
            - Key credentials
            
            POSITION ANALYSIS:
            - Specialty occupation qualification
            - Required skills and knowledge
            - Position level appropriateness
            
            EXPERIENCE ASSESSMENT:
            - Relevant skills and expertise
            - Experience quality
            - Market demand for skills
            
            SPONSORSHIP STRENGTH:
            - Business need justification
            - Position requirements alignment
            - Overall case merits
            
            KEY CONSIDERATIONS:
            - Major strengths
            - Areas needing attention
            - Documentation suggestions
            
            FINAL ASSESSMENT:
            - Overall case strength (Strong/Moderate/Weak)
            - Key recommendations
            - Next steps
            
            Focus on substantive factors that directly impact H-1B eligibility. Avoid over-emphasis on minor details that don't affect the core requirements.
            """
            
            # Query the vector store for relevant context
            results = self.collection.query(
                query_texts=["education experience skills sponsorship qualifications position requirements"],
                n_results=10
            )
            
            # Combine retrieved documents into context
            context = "\n\n".join(results['documents'][0])
            
            # Replace placeholder in prompt
            final_prompt = evaluation_prompt.format(context=context)
            
            # Get LLM response
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an experienced immigration officer focusing on key H-1B eligibility criteria. Provide balanced evaluation based on substantial factors."
                    },
                    {
                        "role": "user",
                        "content": final_prompt
                    }
                ],
                model="llama-3.2-90b-vision-preview",
                temperature=0.7,
                max_tokens=2500
            )
            
            return {
                "evaluation_result": response.choices[0].message.content
            }
            
        except Exception as e:
            raise Exception(f"Error during evaluation: {str(e)}")

def main():
    st.set_page_config(page_title="H1B Application Evaluator", page_icon="📋", layout="wide")
    
    st.title("H1B Visa Application Evaluation System")
    st.markdown("""
    This system provides a detailed evaluation of H1B visa applications based on the candidate's resume 
    and sponsorship letter. Upload your documents below for a thorough analysis.
    """)
    
    # Initialize session states
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = None
    if 'previous_files' not in st.session_state:
        st.session_state.previous_files = None
    
    # API Key input
    groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
    
    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
    with col2:
        letter_file = st.file_uploader("Upload Sponsorship Letter (PDF)", type="pdf")
    
    # Check if files have changed
    current_files = (resume_file, letter_file) if resume_file and letter_file else None
    if current_files != st.session_state.previous_files:
        if st.session_state.evaluator:
            st.session_state.evaluator.cleanup()
            st.session_state.evaluator = None
        st.session_state.previous_files = current_files
    
    if st.button("Evaluate Application", disabled=not (groq_api_key and resume_file and letter_file)):
        try:
            with st.spinner("Processing documents and generating evaluation..."):
                # Initialize new evaluator if needed
                if not st.session_state.evaluator:
                    st.session_state.evaluator = H1BEvaluationSystem(groq_api_key)
                
                # Process documents
                st.info("Processing uploaded documents...")
                st.session_state.evaluator.process_documents(resume_file, letter_file)
                
                # Generate evaluation
                st.info("Generating evaluation...")
                result = st.session_state.evaluator.evaluate_candidate()
                
                # Display results in an expanded section
                with st.expander("📊 Evaluation Results", expanded=True):
                    st.markdown(result["evaluation_result"])
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if st.session_state.evaluator:
                st.session_state.evaluator.cleanup()
                st.session_state.evaluator = None

if __name__ == "__main__":
    main()