# NLP-Assisted Qualitative Coding

This repository contains Python scripts for using a semantic shift algorithm for chunking text documents and classifying chunks using GPT-4. This repository is a companion to the paper "Integrating Large Language Models into Qualitative Methods in Implementation Science: A Proof-of-Concept Study" by the Authors Lia Chin-Purcell, Elena Rosenberg-Carlson, Helene Chokron-Garneau and Mark McGovern.

## Features

- **Semantic Chunking**: Splits text documents into semantically coherent chunks using Sentence Transformers and cosine similarity.
- **GPT-Based Classification**: Uses GPT-4 via Secure GPT, a Stanford-hosted instance of GPT-4, to classify text segments according to a codebook
- **JSON Output**: Exports results in structured JSON format.

## Set up

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file in the project directory
   - Add your API credentials:
   ```
   API_URL=your_api_endpoint_url
   API_KEY=your_api_key
   ```
4. Add your codebook to the `codebook.txt` file.

5. Adapt the scripts to your needs based on your coding guidelines, dataset, and task.

## Usage

Process a single document:

```bash
python code_classifier.py input_file.txt codebook.txt output.json
```

### Input Format

**Text Files**: The input text file should be a plain text file with one paragraph per line. The tool expects speaker labels like "interviewer:" and "interviewee:" 

**Codebook Format**: The script is set up to read in a codebook as a text file with the following format:

```
Code Name: Description of the code
Example: Example text for this code
Example: Another example

Parent Code: Child Code: Description of child code
Example: Example for child code

Another Code (Parent Code): Description of parent code
Example: Example for parent code
```

### Output Format

The tool generates a JSON file with the following structure:

```json
{
  "document_id": "your_document",
  "chunks": [
    {
      "chunk_index": 1,
      "text": "chunk text here",
      "similarity": 0.45,
      "codes": [
        {
          "code": "Code Name",
          "explanation": "Explanation for why this code was selected",
          "certainty": "high"
        }
      ]
    }
  ]
}
```

## Configuration

### API Configuration

The tool uses environment variables for API configuration:
- `API_URL`: Your API endpoint URL
- `API_KEY`: Your API key (e.g., OpenAI subscription key)

### Chunking Parameters

The semantic chunking uses adaptive thresholds based on context:
- Clarification requests: Very low threshold (0.01) to keep together
- Interviewer questions: Higher threshold (0.7) to split
- Default: Moderate threshold (0.35) for general content

## Requirements

- Python 3.7+
- See `requirements.txt` for package dependencies

## Dependencies

- `sentence-transformers`: For semantic embeddings and chunking
- `numpy`: For numerical operations
- `requests`: For API calls
- `python-dotenv`: For environment variable management
