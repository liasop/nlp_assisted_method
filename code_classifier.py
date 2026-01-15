"""
Script for classifying text chunks using GPT-based code classification.
This version supports multiple codes per chunk when appropriate.
"""

from semantic_chunker import SemanticChunker
import os
import json
from typing import List, Dict, Any
import re
import requests
from dotenv import load_dotenv
from requests.exceptions import RequestException
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()


def classify_chunk(chunk: Dict[str, Any], codebook: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Classify a chunk using GPT API based on the provided codebook.
    
    Args:
        chunk: A dictionary containing the chunk text and metadata
        codebook: Dictionary containing code definitions and examples
        
    Returns:
        The same chunk dictionary with added classification information
    """
    api_url = os.getenv("API_URL")
    api_key = os.getenv("API_KEY")

    # Format codebook for the prompt
    codebook_text = ""
    for code, details in codebook.items():
        codebook_text += f"\n{code}:\n{details['description']}\n"
        if details['examples']:
            codebook_text += "Examples:\n"
            for example in details['examples']:
                if example:  # Only add non-empty examples
                    codebook_text += f"- {example}\n"
    
    prompt = f"""
    You are a qualitative researcher coding interview transcripts. You need to classify the following text segment 
    according to this codebook. Please analyze this text carefully and assign a code or codes according to the codebook descriptions.
    
    CODEBOOK:
    {codebook_text}
    
    TEXT TO CODE:
    {chunk['text']}
    
    Previous Context (if available):
    {chunk.get('previous_text', 'No previous context available')}
    
    Project aim: [Your research aim here]
    
    Coding Style Guidelines:
        Your goal is to capture and categorize information of relevance to the research aim.
        To do this, you must determine what the participants are talking about throughout their
        interviews. We make this determination by:
        - Close-reading the transcripts.
        - Cueing into key elements of context and content: the question that was asked, who the participant is discussing (e.g., their organization, an entity outside of their organization, the broader network), what they're discussing (e.g., services they provide, a collaborative relationship), and how they're framing the discussion (e.g., a challenge, an opportunity).
        - Please note that the participant's response may not always directly match the question asked immediately before it. Use the context to understand what the response refers to, but rely on the codebook to determine the appropriate code for text.
        - Child codes under a parent category are labeled "Parent Category Name: Child Code Name" (e.g., "Organization Characteristics: Opioid Use Services" is the label for the opioid use services child code).
        - Codes in the "organization characteristics" parent category, including the parent code and child codes, should only be applied when the participant is discussing their own organization or entities within their organization. Codes in the "organization collaboration" parent category, including the parent code and child codes, should only be applied when the participant is discussing a collaboration between their organization and a specific entity external to their organization. Codes in the "network characteristics" parent category, including the parent code and child codes, should only be applied when the participant is discussing attributes of the broader network of service entities.
        - The "organization characteristics," "organization collaboration," and "network characteristics" parent codes should be used to capture general information related to their respective parent category that is not captured by another code in that category. Per the code descriptions, these parent codes should only be used after determining the other codes in said category don't apply.

    Multiple Coding:
        - You may apply multiple codes to the text segment if it is highly aligned with the codebook description for multiple codes.
    
    Please provide your response in JSON format with these fields:
    {{
        "codes": [
            {{
                "code": "string",  # The code from the codebook
                "explanation": "string",  # Explanation for why this code was selected
                "certainty": "high|medium|low"  # Confidence level: "high" if very confident, "medium" if somewhat confident, "low" if uncertain
            }}
        ]
    }}
    
    """
    
    payload = json.dumps({
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    
    headers = {
        'Ocp-Apim-Subscription-Key': api_key,
        'Content-Type': 'application/json'
    }
    
    try:
        max_retries = 20
        base_delay = 1
        max_delay = 60
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Calculate exponential backoff with a maximum cap
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    print(f"Rate limit hit. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}...")
                    import time
                    time.sleep(delay)
                
                response = requests.post(api_url, headers=headers, data=payload)
                
                if response.status_code == 429:  # Too Many Requests
                    if attempt == max_retries - 1:
                        raise RequestException(f"Rate limit exceeded after {max_retries} attempts with maximum delay of {max_delay} seconds")
                    continue
                
                response.raise_for_status()
                return_data = response.json()
                
                # Get and clean the content from the response
                content = return_data['choices'][0]['message']['content']
                
                # Extract JSON from the content
                try:
                    # First try: direct JSON parsing
                    classification = json.loads(content)
                except json.JSONDecodeError:
                    # Second try: extract JSON from markdown code blocks
                    if '```json' in content:
                        json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                        if json_match:
                            try:
                                classification = json.loads(json_match.group(1))
                            except json.JSONDecodeError:
                                cleaned_content = json_match.group(1).strip()
                                classification = json.loads(cleaned_content)
                        else:
                            raise ValueError("Could not find JSON content in markdown block")
                    else:
                        # Try finding any JSON-like structure
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            try:
                                classification = json.loads(json_match.group(0))
                            except json.JSONDecodeError:
                                raise ValueError("Could not parse JSON content")
                
                # Update the chunk with classification information
                codes = classification.get('codes', [])
                
                # Store the original codes array
                chunk['codes'] = codes
                
                return chunk
                
            except RequestException as e:
                if attempt == max_retries - 1:
                    raise
                continue
                
    except Exception as e:
        print(f"Error classifying chunk: {e}")
        chunk.update({
            'classification_error': str(e)
        })
        
    return chunk


def classify_chunk_with_index(args):
    """Wrapper function for parallel processing that includes index for ordering.
    
    Args:
        args: Tuple of (index, chunk, codebook)
        
    Returns:
        Tuple of (index, classified_chunk)
    """
    index, chunk, codebook = args
    classified_chunk = classify_chunk(chunk, codebook)
    return (index, classified_chunk)


def load_codebook(codebook_path: str = "your_codebook.txt") -> Dict[str, Dict[str, str]]:
    """Load and parse the codebook file.
    
    Args:
        codebook_path: Path to the codebook text file
        
    Returns:
        Dictionary mapping code names to their descriptions and examples.
    """
    with open(codebook_path, 'r') as f:
        content = f.read()
    
    codes = {}
    current_code = None
    current_text = []
    current_examples = []
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        if ':' in line and not line.startswith('Example'):  # New code definition
            if current_code:  # Save previous code if exists
                codes[current_code] = {
                    'description': '\n'.join(current_text).strip(),
                    'examples': current_examples
                }
                current_examples = []  # Reset examples for next code
            
            # Handle the new format with parent:child: description
            if '(Parent Code)' in line:
                # Parent code format: "Code Name (Parent Code): description"
                current_code = line.split(':')[0].strip()
                current_text = [line.split(':', 1)[1].strip()]
            else:
                # Determine if this is a child code (Parent: Child: description) or standalone code (Code: description)
                parts = line.split(':')
                
                if len(parts) >= 3:
                    # Check if this is a child code by looking for the pattern "Parent: Child"
                    first_part = parts[0].strip()
                    second_part = parts[1].strip()
                    
                    # If the second part is short (< 50 chars), it's a child code
                    if len(second_part) < 50:
                        # This looks like a child code: "Parent: Child: description"
                        current_code = ':'.join(parts[:2]).strip()
                        current_text = [':'.join(parts[2:]).strip()]
                    else:
                        # This looks like a standalone code with multiple colons: "Code: description: more description"
                        current_code = first_part
                        current_text = [':'.join(parts[1:]).strip()]
                elif len(parts) == 2:
                    # Only 2 parts: Code, Description (Parent Code)
                    current_code = parts[0].strip()
                    current_text = [parts[1].strip()]
                else:
                    # Fallback
                    current_code = line.split(':')[0].strip()
                    current_text = [line.split(':', 1)[1].strip()]
                    
        elif line.startswith('Example'):  # Example for current code
            if current_code:
                example = line.split(':', 1)[1].strip() if ':' in line else ''
                current_examples.append(example)
        else:
            if current_code:  # Continue current code description
                current_text.append(line)
    
    # Save last code
    if current_code:
        codes[current_code] = {
            'description': '\n'.join(current_text).strip(),
            'examples': current_examples
        }
    
    return codes


def get_semantic_chunks(file_path: str) -> List[Dict[str, Any]]:
    """Process a text file to create chunks based on neighboring paragraph comparison.
    
    Args:
        file_path: Path to the text file to process
        
    Returns:
        List of dictionaries containing chunked text with metadata
    """
    chunker = SemanticChunker()
    return chunker._process_txt_neighboring_comparison(file_path)


def save_results_to_json(chunks: List[Dict[str, Any]], output_path: str, document_id: str = None):
    """Save classified chunks to a JSON file.
    
    Args:
        chunks: List of classified chunks with codes
        output_path: Path where the JSON file should be saved
        document_id: Optional document identifier
    """
    # Prepare output structure
    output_data = {
        "document_id": document_id or "your_document",
        "chunks": []
    }
    
    for idx, chunk in enumerate(chunks, start=1):
        chunk_data = {
            "chunk_index": idx,
            "text": chunk.get('text', ''),
            "similarity": chunk.get('similarity', None)
        }
        
        # Add codes if available
        codes = chunk.get('codes', [])
        if codes:
            chunk_data["codes"] = []
            for code_obj in codes:
                chunk_data["codes"].append({
                    "code": code_obj.get('code', 'N/A'),
                    "explanation": code_obj.get('explanation', 'N/A'),
                    "certainty": code_obj.get('certainty', 'medium')
                })
        else:
            chunk_data["codes"] = []
        
        # Add error information if classification failed
        if 'classification_error' in chunk:
            chunk_data["classification_error"] = chunk['classification_error']
        
        output_data["chunks"].append(chunk_data)
    
    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_path}")


def process_document(file_path: str, codebook_path: str, output_path: str = None, 
                    max_workers: int = 5):
    """Process a single document: chunk it and classify the chunks.
    
    Args:
        file_path: Path to the text file to process
        codebook_path: Path to the codebook file
        output_path: Path for the output JSON file (default: based on input filename)
        max_workers: Number of parallel workers for classification (default: 5)
    """
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = f"{base_name}_classified.json"
    
    # Get document ID from filename
    document_id = os.path.splitext(os.path.basename(file_path))[0]
    
    print(f"Processing document: {file_path}")
    
    # Load codebook
    print(f"Loading codebook from {codebook_path}...")
    codebook = load_codebook(codebook_path)
    print(f"Codebook loaded with {len(codebook)} codes")
    
    # Get chunks
    print(f"Chunking document...")
    chunks = get_semantic_chunks(file_path)
    print(f"Created {len(chunks)} chunks")
    
    # Add previous_text to each chunk (except the first one)
    for i in range(1, len(chunks)):
        chunks[i]['previous_text'] = chunks[i-1]['text']
    
    # Classify chunks in parallel
    print(f"Classifying {len(chunks)} chunks...")
    classification_args = [(i, chunk, codebook) for i, chunk in enumerate(chunks)]
    
    classified_chunks = [None] * len(chunks)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(classify_chunk_with_index, args): args[0] 
                          for args in classification_args}
        
        for future in as_completed(future_to_index):
            try:
                index, classified_chunk = future.result()
                classified_chunks[index] = classified_chunk
                completed += 1
                print(f"Completed chunk {completed}/{len(chunks)} (chunk {index+1})...")
            except Exception as e:
                index = future_to_index[future]
                print(f"Error classifying chunk {index+1}: {e}")
                classified_chunks[index] = chunks[index]
                classified_chunks[index]['classification_error'] = str(e)
                completed += 1
    
    # Save results
    print(f"Saving results to {output_path}...")
    save_results_to_json(classified_chunks, output_path, document_id)
    
    print(f"✓ Successfully processed document")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    # Example usage - modify these paths as needed
    
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python code_classifier.py <input_file.txt> <codebook.txt> [output_file.json]")
        print("\nExample:")
        print("  python code_classifier.py data/your_data.txt your_codebook.txt output.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    codebook_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    if not os.path.exists(codebook_file):
        print(f"Error: Codebook file not found: {codebook_file}")
        sys.exit(1)
    
    process_document(input_file, codebook_file, output_file)
