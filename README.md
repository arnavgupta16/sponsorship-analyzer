# sponsorship-analyzer

An AI-powered system that evaluates H1B visa applications by analyzing resumes and sponsorship letters using the Groq LLM API and vector database technology.

## Features

- PDF document processing for resumes and sponsorship letters
- Advanced text analysis using Groq LLM
- Vector-based document similarity search
- Comprehensive H1B eligibility evaluation
- Interactive Streamlit web interface
- Automatic resource cleanup and management

## Prerequisites

- Python 3.8 or higher
- Groq API key ([Get your API key here](https://console.groq.com))
- Sufficient disk space for temporary file storage
- Internet connection for API access

## Installation

1. Clone the repository:
```bash
git clone https://github.com/arnavgupta16/sponsorship-analyzer
cd sponsorship-analyzer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Enter your Groq API key in the secure input field

4. Upload the required documents:
   - Candidate's resume (PDF format)
   - Sponsorship letter (PDF format)

5. Click "Evaluate Application" to generate the analysis

## Evaluation Criteria

The system evaluates applications based on key H1B visa requirements:

1. Educational Qualifications
   - Degree level and relevance
   - Field of study alignment

2. Position Requirements
   - Specialty occupation qualification
   - Required knowledge and skills
   - Position complexity

3. Experience and Skills
   - Technical/professional expertise
   - Experience relevance
   - Market demand

4. Sponsorship Justification
   - Business need
   - Position requirements
   - Wage level appropriateness

## System Architecture

- **Frontend**: Streamlit web interface
- **Document Processing**: PyPDF2 for PDF text extraction
- **Text Analysis**: Groq LLM API
- **Vector Storage**: ChromaDB for document similarity search
- **Embeddings**: Sentence Transformers for text vectorization

## File Structure

```
h1b-evaluator/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── README.md          # Documentation
└── temp/              # Temporary storage (auto-generated)
```

## Error Handling

The system includes robust error handling for:
- Invalid API keys
- PDF processing errors
- Network connectivity issues
- Resource cleanup
- Session management

## Best Practices

1. Always use a secure and valid Groq API key
2. Ensure PDF documents are properly formatted
3. Clear browser cache if experiencing issues
4. Monitor system resources during large document processing

## Limitations

- Supports PDF documents only
- Requires active internet connection
- Processing time depends on document size
- Temporary files need sufficient disk space

## Troubleshooting

1. **API Key Issues**:
   - Verify API key validity
   - Check internet connection
   - Ensure proper key format

2. **Document Processing Errors**:
   - Confirm PDF format
   - Check file permissions
   - Verify file isn't corrupted

3. **Performance Issues**:
   - Clear browser cache
   - Restart application
   - Check system resources

## Security Considerations

- API keys are handled securely
- Documents are processed locally
- Temporary files are automatically cleaned up
- No data is permanently stored

