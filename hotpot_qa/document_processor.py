"""
Document processor for HotpotQA dataset.

Transforms HotpotQA dataset into documents suitable for indexing by
Azure AI Search and FAISS.
"""

import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import os
import argparse
import urllib.parse
import base64
import re


@dataclass
class Document:
    """Represents a document for indexing."""
    id: str
    title: str
    content: str
    source: str = "hotpotqa"


@dataclass 
class QuestionAnswerPair:
    """Represents a question-answer pair with metadata."""
    id: str
    question: str
    answer: str
    question_type: str
    difficulty_level: str
    supporting_facts: List[Tuple[str, int]]
    source: str = "hotpotqa"


class DocumentProcessor:
    """Processes HotpotQA dataset into indexable documents."""
    
    def __init__(self, dataset_path: str):
        """Initialize with path to HotpotQA dataset."""
        self.dataset_path = dataset_path
    
    def sanitize_document_id(self, raw_id: str) -> str:
        """Create a URL-safe document ID for Azure AI Search.
        
        Azure Search keys can only contain letters, digits, underscore (_), 
        dash (-), or equal sign (=). We'll use a combination of character 
        replacement and base64 encoding for problematic characters.
        """
        # First, try simple character replacement for common cases
        sanitized = raw_id.lower()
        
        # Replace spaces and common separators with underscores
        sanitized = re.sub(r'[\s\-\.]+', '_', sanitized)
        
        # Replace other problematic characters with safe equivalents
        char_map = {
            '(': '_lp_',  # left parenthesis
            ')': '_rp_',  # right parenthesis  
            '[': '_lb_',  # left bracket
            ']': '_rb_',  # right bracket
            '{': '_lc_',  # left curly
            '}': '_rc_',  # right curly
            ',': '_comma_',
            ';': '_semi_',
            ':': '_colon_',
            '!': '_excl_',
            '?': '_quest_',
            '@': '_at_',
            '#': '_hash_',
            '$': '_dollar_',
            '%': '_pct_',
            '^': '_caret_',
            '&': '_amp_',
            '*': '_star_',
            '+': '_plus_',
            '/': '_slash_',
            '\\': '_bslash_',
            '|': '_pipe_',
            '<': '_lt_',
            '>': '_gt_',
            '"': '_quote_',
            "'": '_apos_'
        }
        
        for char, replacement in char_map.items():
            sanitized = sanitized.replace(char, replacement)
        
        # Remove any remaining non-allowed characters
        sanitized = re.sub(r'[^a-zA-Z0-9_\-=]', '_', sanitized)
        
        # Collapse multiple underscores
        sanitized = re.sub(r'_{2,}', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure the ID isn't empty and isn't too long (Azure has limits)
        if not sanitized:
            sanitized = 'doc'
        
        # If still too long (>1024 chars), use base64 encoding of the original
        if len(sanitized) > 500:  # Conservative limit
            encoded = base64.urlsafe_b64encode(raw_id.encode('utf-8')).decode('ascii')
            # Remove padding to avoid issues
            encoded = encoded.rstrip('=')
            sanitized = f"b64_{encoded}"
        
        return sanitized
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the HotpotQA dataset."""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def process_record(self, record: Dict[str, Any]) -> List[Document]:
        """
        Process a single HotpotQA record into multiple documents.
        
        Each context entry becomes a separate document for better
        granular retrieval.
        """
        documents = []
        record_id = record.get('_id', '')
        
        context = record.get('context', [])
        for i, (title, content_list) in enumerate(context):
            # Join content sentences into a single text
            content = ' '.join(content_list) if isinstance(content_list, list) else str(content_list)
            
            # Create a raw ID first, then sanitize it
            raw_id = f"{record_id}_{i}_{title}"
            doc_id = self.sanitize_document_id(raw_id)
            
            document = Document(
                id=doc_id,
                title=title,
                content=content,
                source="hotpotqa"
            )
            documents.append(document)
            
        return documents
    
    def extract_qa_pair(self, record: Dict[str, Any]) -> QuestionAnswerPair:
        """
        Extract question-answer pair with type and level information from a record.
        
        Args:
            record: HotpotQA dataset record
            
        Returns:
            QuestionAnswerPair with extracted information
        """
        record_id = record.get('_id', '')
        question = record.get('question', '')
        answer = record.get('answer', '')
        question_type = record.get('type', 'unknown')
        difficulty_level = record.get('level', 'unknown')
        supporting_facts = record.get('supporting_facts', [])
        
        # Convert supporting facts to list of tuples
        supporting_facts_tuples = [(fact[0], fact[1]) for fact in supporting_facts if len(fact) >= 2]
        
        return QuestionAnswerPair(
            id=record_id,
            question=question,
            answer=answer,
            question_type=question_type,
            difficulty_level=difficulty_level,
            supporting_facts=supporting_facts_tuples,
            source="hotpotqa"
        )
    
    def process_all(self) -> List[Document]:
        """Process entire dataset into documents."""
        dataset = self.load_dataset()
        all_documents = []
        
        for record in dataset:
            documents = self.process_record(record)
            all_documents.extend(documents)
            
        return all_documents
    
    def extract_all_qa_pairs(self) -> List[QuestionAnswerPair]:
        """Extract all question-answer pairs from the dataset."""
        dataset = self.load_dataset()
        qa_pairs = []
        
        for record in dataset:
            qa_pair = self.extract_qa_pair(record)
            qa_pairs.append(qa_pair)
            
        return qa_pairs
    
    def save_documents(self, documents: List[Document], output_path: str):
        """Save processed documents to JSON file."""
        doc_dicts = []
        for doc in documents:
            doc_dicts.append({
                'id': doc.id,
                'title': doc.title, 
                'content': doc.content,
                'source': doc.source
            })
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_dicts, f, ensure_ascii=False, indent=2)
            
        print(f"Saved {len(documents)} documents to {output_path}")

    def save_qa_pairs(self, qa_pairs: List[QuestionAnswerPair], output_path: str):
        """Save question-answer pairs to JSON file."""
        qa_dicts = []
        for qa in qa_pairs:
            qa_dicts.append({
                'id': qa.id,
                'question': qa.question,
                'answer': qa.answer,
                'type': qa.question_type,
                'level': qa.difficulty_level,
                'supporting_facts': qa.supporting_facts,
                'source': qa.source
            })
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_dicts, f, ensure_ascii=False, indent=2)
            
        print(f"Saved {len(qa_pairs)} question-answer pairs to {output_path}")

    def get_qa_statistics(self, qa_pairs: List[QuestionAnswerPair]) -> Dict[str, Any]:
        """Get statistics about question-answer pairs."""
        if not qa_pairs:
            return {}
        
        # Count by type
        type_counts = {}
        for qa in qa_pairs:
            type_counts[qa.question_type] = type_counts.get(qa.question_type, 0) + 1
        
        # Count by level
        level_counts = {}
        for qa in qa_pairs:
            level_counts[qa.difficulty_level] = level_counts.get(qa.difficulty_level, 0) + 1
        
        return {
            'total_pairs': len(qa_pairs),
            'types': type_counts,
            'levels': level_counts
        }


def main(dataset_path: str = "hotpot_dev_fullwiki_v1.json", 
         extract_qa_pairs: bool = False,
         documents_output: str = "hotpot_documents.json",
         qa_pairs_output: str = "hotpot_qa_pairs.json"):
    """Main function to process HotpotQA dataset."""
    
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        return
        
    processor = DocumentProcessor(dataset_path)
    
    # Process documents (original functionality)
    print("Processing documents...")
    documents = processor.process_all()
    processor.save_documents(documents, documents_output)
    print(f"Document processing complete. Generated {len(documents)} documents.")
    
    # Extract question-answer pairs (optional new functionality)
    if extract_qa_pairs:
        print("\nExtracting question-answer pairs...")
        qa_pairs = processor.extract_all_qa_pairs()
        processor.save_qa_pairs(qa_pairs, qa_pairs_output)
        
        # Display statistics
        stats = processor.get_qa_statistics(qa_pairs)
        print(f"\nQuestion-Answer Pair Statistics:")
        print(f"Total pairs: {stats['total_pairs']}")
        print(f"Question types: {stats['types']}")
        print(f"Difficulty levels: {stats['levels']}")
    else:
        print("\nSkipping Q&A pair extraction (use --extract-qa-pairs to enable)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process HotpotQA dataset into documents and optionally extract Q&A pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python document_processor.py
  python document_processor.py --extract-qa-pairs
  python document_processor.py --dataset custom_dataset.json --extract-qa-pairs
  python document_processor.py --dataset data.json --documents-output docs.json --qa-pairs-output qa.json --extract-qa-pairs"""
    )
    
    parser.add_argument(
        "--dataset", 
        default="hotpot_dev_fullwiki_v1.json",
        help="Path to the HotpotQA dataset JSON file (default: hotpot_dev_fullwiki_v1.json)"
    )
    
    parser.add_argument(
        "--extract-qa-pairs",
        action="store_true",
        help="Extract question-answer pairs with type and level information"
    )
    
    parser.add_argument(
        "--documents-output",
        default="hotpot_documents.json",
        help="Output path for processed documents (default: hotpot_documents.json)"
    )
    
    parser.add_argument(
        "--qa-pairs-output",
        default="hotpot_qa_pairs.json",
        help="Output path for extracted Q&A pairs (default: hotpot_qa_pairs.json)"
    )
    
    args = parser.parse_args()
    
    main(
        dataset_path=args.dataset,
        extract_qa_pairs=args.extract_qa_pairs,
        documents_output=args.documents_output,
        qa_pairs_output=args.qa_pairs_output
    )