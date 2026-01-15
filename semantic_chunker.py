"""
Semantic chunker for text documents using neighboring paragraph comparison.
This module provides functionality to split text documents into semantically coherent chunks.
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any
import re


class SemanticChunker:
    """Class for semantically chunking text documents based on paragraph similarity."""
    
    def __init__(self):
        """Initialize the semantic chunker with a sentence transformer model."""
        self.model = SentenceTransformer('all-mpnet-base-v2')
        
        # Initialize clarification detection prototypes
        self.clarification_prototypes = [
            "can you say that again",
            "just say that to me one more time",
            "say that again",
            "repeat that",
            "what do you mean",
            "please clarify",
            "i don't understand",
            "can you repeat that",
            "what was that",
            "i didn't hear",
            "could you clarify",
            "pardon me",
            "excuse me",
            "sorry, what",
            "what did you say",
            "can you explain",
            "i'm not sure what you mean",
            "could you repeat",
            "what does that mean",
            "i don't follow",
            "can you go back",
            "start over",
            "what are you asking",
            "tell me a little more about what you mean",
            "i'm not sure i understand it",
            "can you explain that better",
            "what exactly do you mean",
            "i need more clarification",
            "can you elaborate on that",
            "i don't quite understand",
            "could you be more specific",
            "what do you mean by that",
            "i'm confused about that",
            "can you say that one more time",
            "just repeat that",
            "say that one more time"
        ]
        self.clarification_embeddings = self.model.encode(self.clarification_prototypes, normalize_embeddings=True)

    def is_clarification_request(self, text: str, threshold: float = 0.6) -> bool:
        """Check if text is a clarification request using semantic similarity."""
        # Clean the text by removing speaker labels and extra whitespace
        cleaned_text = re.sub(r'^(interviewer|interviewee(?:\s+\d+)?):\s*', '', text, flags=re.IGNORECASE).strip()
        
        if not cleaned_text:
            return False
            
        # Get embedding for the input text
        text_embedding = self.model.encode(cleaned_text, normalize_embeddings=True)
        
        # Calculate similarity with all clarification prototypes
        similarities = util.cos_sim(text_embedding, self.clarification_embeddings)
        max_similarity = similarities.max().item()
        
        return max_similarity > threshold

    def _process_txt_neighboring_comparison(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a text file using neighboring paragraph comparison algorithm.
        
        Args:
            file_path: Path to the text file to process
            
        Returns:
            List of dictionaries containing chunked text with metadata
        """
        chunks = []
        
        # Read the text file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into paragraphs (lines separated by newlines)
        raw_paragraphs = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Collect all paragraphs, keeping track of which ones are metadata
        all_paragraphs = []
        metadata_paragraph_indices = set()
        
        # Patterns for metadata paragraphs that should not be compared
        # Match paragraphs that are ENTIRELY pauses, end markers, or crosstalk, and paragraphs that are just speaker labels with metadata
        pause_pattern = re.compile(r'^\[Pause.*?\]\s*$')
        end_audio_pattern = re.compile(r'^\[End of Audio\]\s*$')
        crosstalk_pattern = re.compile(r'^\[Crosstalk.*?\]\s*$')
        # Patterns for speaker labels with metadata
        speaker_pause_pattern = re.compile(r'^(interviewer|interviewee(?:\s+\d+)?):\s*\[Pause.*?\]\s*\.?\s*$', re.IGNORECASE)
        speaker_end_audio_pattern = re.compile(r'^(interviewer|interviewee(?:\s+\d+)?):\s*\[End of Audio\]\s*\.?\s*$', re.IGNORECASE)
        speaker_crosstalk_pattern = re.compile(r'^(interviewer|interviewee(?:\s+\d+)?):\s*\[Crosstalk.*?\]\s*\.?\s*$', re.IGNORECASE)
        metadata_patterns = [pause_pattern, end_audio_pattern, crosstalk_pattern, 
                           speaker_pause_pattern, speaker_end_audio_pattern, speaker_crosstalk_pattern]
        
        # Pattern to match standalone timestamps (e.g., [00:00:00] or [00:05:00] with nothing else)
        standalone_timestamp_pattern = re.compile(r'^\[\d+:\d+:\d+\]\s*$')
        
        # First pass: collect all paragraphs and merge split content
        filtered_paragraphs = []
        for text in raw_paragraphs:
=            if standalone_timestamp_pattern.match(text):
                continue
            filtered_paragraphs.append(text)
        
        # Second pass: merge paragraphs that were split by page breaks
        # Look for paragraphs that start with timestamps and merge with previous paragraph
        for i, text in enumerate(filtered_paragraphs):
            # Check if this paragraph starts with a timestamp (indicates page break split)
            if re.match(r'^\[\d+:\d+(?::\d+)?\]', text):
                if all_paragraphs:  # Make sure we have a previous paragraph
                    content_without_timestamp = re.sub(r'^\[\d+:\d+(?::\d+)?\]\s*', '', text)
                    all_paragraphs[-1] += ' ' + content_without_timestamp
                else:
                    # If no previous paragraph, just remove timestamp and add
                    content_without_timestamp = re.sub(r'^\[\d+:\d+(?::\d+)?\]\s*', '', text)
                    all_paragraphs.append(content_without_timestamp)
            else:
                # Regular paragraph - add as is
                all_paragraphs.append(text)
        
        # Third pass: merge short paragraphs with previous paragraphs
        merged_paragraphs = []
        for i, text in enumerate(all_paragraphs):
            # Check if it's a metadata paragraph (now only pauses, since timestamps are removed)
            is_metadata = any(pattern.match(text) for pattern in metadata_patterns)
            if is_metadata:
                metadata_paragraph_indices.add(len(merged_paragraphs))
                merged_paragraphs.append(text)
            elif len(text) < 30:  # Short paragraph - append to previous
                if merged_paragraphs:  # If we have a previous paragraph
                    merged_paragraphs[-1] += '\n' + text
                else:  # If no previous paragraph, just add it
                    merged_paragraphs.append(text)
            elif (text.strip().lower().startswith('interviewer:') and 
                  len(text) < 50 and 
                  not text.strip().endswith('?')):
                # Short interviewer statement that's not a question - merge with previous
                if merged_paragraphs:  # If we have a previous paragraph
                    merged_paragraphs[-1] += '\n' + text
                else:  # If no previous paragraph, just add it
                    merged_paragraphs.append(text)
            else:  # Regular paragraph
                merged_paragraphs.append(text)
        
        # Update all_paragraphs to use merged paragraphs
        all_paragraphs = merged_paragraphs
        
        # Step 1: Compute all cosine similarities between neighboring paragraphs
        similarities = []
        
        for i in range(len(all_paragraphs) - 1):
            current_para = all_paragraphs[i]
            next_para = all_paragraphs[i + 1]
            
            # Skip comparison if either paragraph is metadata
            if (i in metadata_paragraph_indices or (i + 1) in metadata_paragraph_indices):
                similarities.append(1.0)  # Default high similarity for skipped paragraphs
                continue
            
            # Get embeddings
            current_embedding = self.model.encode(current_para, convert_to_numpy=True)
            next_embedding = self.model.encode(next_para, convert_to_numpy=True)
            
            # Calculate cosine similarity
            similarity = np.dot(current_embedding, next_embedding) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(next_embedding)
            )
            similarity_score = float(similarity)
            similarities.append(similarity_score)
        
        # Step 2: Find split points where similarity is below threshold
        split_points = []
        for i, similarity in enumerate(similarities):
            current_para = all_paragraphs[i]
            next_para = all_paragraphs[i + 1]
            
            # Check for clarification requests from interviewees using semantic similarity
            is_next_clarification = False
            is_current_clarification = False
            
            # Check if next paragraph is a clarification
            if re.match(r'^interviewee(?:\s+\d+)?:', next_para, re.IGNORECASE):
                cleaned_text = re.sub(r'^(interviewer|interviewee(?:\s+\d+)?):\s*', '', next_para, flags=re.IGNORECASE).strip()
                if cleaned_text:
                    text_embedding = self.model.encode(cleaned_text, normalize_embeddings=True)
                    clarification_similarities = util.cos_sim(text_embedding, self.clarification_embeddings)
                    clarification_similarity = clarification_similarities.max().item()
                    is_next_clarification = clarification_similarity > 0.6
            
            # Check if current paragraph is a clarification
            if re.match(r'^interviewee(?:\s+\d+)?:', current_para, re.IGNORECASE):
                cleaned_current_text = re.sub(r'^(interviewer|interviewee(?:\s+\d+)?):\s*', '', current_para, flags=re.IGNORECASE).strip()
                if cleaned_current_text:
                    current_text_embedding = self.model.encode(cleaned_current_text, normalize_embeddings=True)
                    current_clarification_similarities = util.cos_sim(current_text_embedding, self.clarification_embeddings)
                    current_clarification_similarity = current_clarification_similarities.max().item()
                    is_current_clarification = current_clarification_similarity > 0.6
            
            # Helper function to detect if text contains a question
            def contains_question(text):
                """Check if text contains question indicators beyond just ending with ?"""
                text_lower = text.lower()
                # Normalize different apostrophe types
                text_lower = text_lower.replace(''', "'").replace(''', "'")
                question_indicators = [
                    'can you', 'could you', 'would you', 'do you', 'did you', 'have you', 'are you',
                    'is there', 'are there', 'what', 'how', 'when', 'where', 'why', 'which', 'who',
                    'i\'m wondering', 'i wonder', 'tell me about', 'can you talk about',
                    'i\'m curious', 'i\'d like to know', 'i want to know', 'i\'m interested in',
                    'wondering if you can', 'wondering if', 'can you identify', 'can you tell'
                ]
                return any(indicator in text_lower for indicator in question_indicators)
            
            # Check if current paragraph is interviewer question (not followed by clarification)
            is_current_interviewer_question = (current_para.lower().startswith('interviewer') and 
                                            (current_para.strip().endswith('?') or contains_question(current_para)) and 
                                            not is_next_clarification)
            
            # Check if current starts with interviewer and next doesn't start with interviewee
            is_interviewer_not_followed_by_interviewee = (current_para.lower().startswith('interviewer') and 
                                                         not next_para.lower().startswith('interviewee'))
            
            # Check if next paragraph is interviewer question
            is_next_interviewer_question = (next_para.lower().startswith('interviewer') and 
                                         (next_para.strip().endswith('?') or contains_question(next_para)))
            
            # Check if both current and next start with interviewer (but next is not a question)
            is_both_interviewer = (current_para.lower().startswith('interviewer') and 
                                 next_para.lower().startswith('interviewer') and 
                                 not is_next_interviewer_question)
            
            # Determine threshold based on the cases (check most specific first)
            # Cases that should almost never split (threshold 0.01)
            if (is_next_clarification or is_current_clarification or 
                is_interviewer_not_followed_by_interviewee):
                effective_threshold = 0.01
            # Next interviewer question: almost always split (threshold 0.7)
            elif is_next_interviewer_question:
                effective_threshold = 0.7
            # Interviewer questions/continuations: very dissimilar to split (threshold 0.15)
            elif is_both_interviewer or is_current_interviewer_question:
                effective_threshold = 0.15
            # Default: everything else (threshold 0.35)
            else:
                effective_threshold = 0.35
            
            if similarity < effective_threshold:
                split_points.append(i + 1)
        
        # Step 3: Create chunks based on split points
        current_section = "Introduction"
        start_idx = 0
        
        for split_idx in split_points:
            # Create chunk from start_idx to split_idx
            chunk_paragraphs = all_paragraphs[start_idx:split_idx]
            if chunk_paragraphs and len(chunk_paragraphs) > 0:
                chunk_text = '\n'.join(chunk_paragraphs)
                # Remove timestamps from the beginning of lines
                lines = chunk_text.split('\n')
                cleaned_lines = []
                for line in lines:
                    cleaned_line = re.sub(r'^\[\d+:\d+(?::\d+)?\]\s*', '', line)
                    cleaned_lines.append(cleaned_line)
                chunk_text = '\n'.join(cleaned_lines)
                # Skip very short or empty chunks (less than 10 characters) and chunks that are just standalone timestamps
                chunk_text_stripped = chunk_text.strip()
                if len(chunk_text_stripped) >= 10 and not standalone_timestamp_pattern.match(chunk_text_stripped):
                    chunks.append({
                        'section_marker': current_section,
                        'text': chunk_text,
                        'source': os.path.basename(file_path),
                        'shift_type': 'semantic_shift',
                        'similarity': similarities[split_idx - 1] if split_idx > 0 else 1.0,
                        'shift_cause': all_paragraphs[split_idx][:200] if split_idx < len(all_paragraphs) else "",
                        'paragraphs': chunk_paragraphs.copy()
                    })
            start_idx = split_idx
        
        # Add final chunk if there are remaining paragraphs
        if start_idx < len(all_paragraphs):
            chunk_paragraphs = all_paragraphs[start_idx:]
            if chunk_paragraphs:
                chunk_text = '\n'.join(chunk_paragraphs)
                # Remove timestamps from the beginning of lines
                lines = chunk_text.split('\n')
                cleaned_lines = []
                for line in lines:
                    cleaned_line = re.sub(r'^\[\d+:\d+(?::\d+)?\]\s*', '', line)
                    cleaned_lines.append(cleaned_line)
                chunk_text = '\n'.join(cleaned_lines)
                chunk_text_stripped = chunk_text.strip()
                # Skip final chunk if it's just a standalone timestamp
                if not standalone_timestamp_pattern.match(chunk_text_stripped):
                    chunks.append({
                        'section_marker': current_section,
                        'text': chunk_text,
                        'source': os.path.basename(file_path),
                        'shift_type': 'final_chunk',
                        'similarity': 1.0,
                        'paragraphs': chunk_paragraphs.copy()
                    })
        
        return chunks
