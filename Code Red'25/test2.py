import os
import re
import json
import torch
import spacy
from typing import List, Dict, Optional

# PDF Extraction Libraries
try:
    import fitz  # PyMuPDF
except ImportError:
    os.system('pip install pymupdf')
    import fitz

try:
    import PyPDF2
except ImportError:
    os.system('pip install PyPDF2')
    import PyPDF2

# Transformers and ML Libraries
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,  # Changed from previous approach
    pipeline
)

class AdvancedPDFQuestionEvaluator:
    def __init__(self, 
                 model_name="microsoft/Orca-2-13b",  # More compatible causal LM
                 nlp_model='en_core_web_sm'):
        """
        Initialize PDF processing with advanced NLP and language models
        
        :param model_name: Hugging Face model for question generation
        :param nlp_model: spaCy NLP model
        """
        # Ensure spaCy model is downloaded
        try:
            self.nlp = spacy.load(nlp_model)
        except OSError:
            print(f"Downloading {nlp_model} spaCy model...")
            os.system(f'python -m spacy download {nlp_model}')
            self.nlp = spacy.load(nlp_model)

        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            '''
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
                '''
            model_kwargs = {
                "torch_dtype": torch.float32,
               "device_map": "auto" if self.device == "cuda" else None,
               "low_cpu_mem_usage": True,
            }

            print("Note: Loading a large language model on CPU might be slow and memory-intensive.")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            print(f"Model loading error: {e}")
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Robust PDF text extraction with multiple methods
        
        :param pdf_path: Path to the PDF file
        :return: Extracted text as a string
        """
        try:
            # First try PyMuPDF (most reliable)
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            doc.close()
            return full_text
        except Exception as pdf_mupdf_error:
            print(f"PyMuPDF extraction failed: {pdf_mupdf_error}")
            
            try:
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    full_text = ""
                    for page in pdf_reader.pages:
                        full_text += page.extract_text() + "\n"
                return full_text
            except Exception as pdf_fallback_error:
                print(f"PyPDF2 extraction failed: {pdf_fallback_error}")
                raise ValueError(f"Could not extract text from PDF: {pdf_path}")

    def generate_questions(self, pdf_path: str, num_questions: int = 5) -> List[Dict]:
        """
        Generate sophisticated questions using advanced language model
        
        :param pdf_path: Path to the PDF file
        :param num_questions: Number of questions to generate
        :return: List of generated questions
        """
        # Extract text from PDF
        full_text = self.extract_text_from_pdf(pdf_path)
        
        # Truncate text to first 4000 characters to avoid overwhelming the model
        context_text = full_text[:4000]
        
        # Prepare prompt for question generation
        prompt = f"""Based on the following academic text, generate {num_questions} sophisticated, 
        multi-layered, and intellectually challenging questions that:
        - Require deep critical thinking
        - Explore complex conceptual relationships
        - Demand comprehensive understanding
        - Encourage interdisciplinary analysis

        Text Context:
        {context_text}

        Output Format (JSON):
        [
          {{
            "question": "Detailed question text",
            "conceptual_domains": ["Domain1", "Domain2"],
            "complexity_level": "High"
          }}
        ]

        Generated Questions:
        """

        try:
            generation_args = {
                "max_new_tokens": 512,
                "return_full_text": False,
                "do_sample":True,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            full_response = self.generator(prompt, **generation_args)[0]['generated_text']
            
            # Extract JSON-like structure
            questions_match = re.findall(r'\[.*?\]', full_response, re.DOTALL)
            
            if questions_match:
                try:
                    questions = eval(questions_match[0])
                    return questions[:num_questions]
                except:
                    # Fallback questions
                    return [
                        {
                            "question": f"Critically analyze the interdisciplinary connections in the text about {os.path.basename(pdf_path)}",
                            "conceptual_domains": ["Interdisciplinary Analysis"],
                            "complexity_level": "High"
                        }
                    ]
            else:
                # Fallback questions
                return [
                    {
                        "question": f"Critically analyze the interdisciplinary connections in the text about {os.path.basename(pdf_path)}",
                        "conceptual_domains": ["Interdisciplinary Analysis"],
                        "complexity_level": "High"
                    }
                ]

        except Exception as e:
            print(f"Error generating questions: {e}")
            return [
                {
                    "question": f"Critically analyze the interdisciplinary connections in the text about {os.path.basename(pdf_path)}",
                    "conceptual_domains": ["Interdisciplinary Analysis"],
                    "complexity_level": "High"
                }
            ]

    def process_pdf(self, 
                    pdf_path: str, 
                    student_answers: Optional[List[str]] = None) -> List[Dict]:
        """
        Comprehensive PDF processing workflow
        
        :param pdf_path: Path to PDF file
        :param student_answers: Optional list of student answers
        :return: Processed results
        """
        try:
            # Generate questions
            questions = self.generate_questions(pdf_path)
            
            # Prepare results
            results = []
            
            # If no student answers provided, create placeholders
            if not student_answers:
                student_answers = ["" for _ in questions]
            
            # Prepare results with questions and blank evaluations
            for question, student_answer in zip(questions, student_answers):
                results.append({
                    "question": question,
                    "student_answer": student_answer
                })
            
            # Save results
            self.save_results(results, pdf_path)
            
            return results

        except Exception as e:
            print(f"Error processing PDF: {e}")
            return []

    def save_results(self, 
                     results: List[Dict], 
                     pdf_path: str, 
                     filename: Optional[str] = None) -> None:
        """
        Save comprehensive evaluation results
        
        :param results: Processed results
        :param pdf_path: Source PDF path
        :param filename: Optional custom filename
        """
        if not filename:
            filename = f"question_results_{os.path.basename(pdf_path).replace('.pdf', '')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"Questions generated and saved to {filename}")

def install_dependencies():
    """
    Install required libraries
    """
    dependencies = [
        'pymupdf', 
        'PyPDF2', 
        'transformers', 
        'torch', 
        'spacy'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            print(f"Installing {dep}...")
            os.system(f'pip install {dep}')
    
    # Download spaCy model
    os.system('python -m spacy download en_core_web_sm')

def main():
    # Install dependencies
    install_dependencies()
    
    # Initialize advanced PDF question generator
    evaluator = AdvancedPDFQuestionEvaluator()
    
    # Path to PDF (replace with your actual PDF path)
    pdf_path = r"c:\Users\sharo\OneDrive\Desktop\Amity College\Hackathon\Code Red'25\sample1.pdf"
    
    # Process PDF and generate questions
    results = evaluator.process_pdf(pdf_path)
    
    # Print generated questions
    for result in results:
        print("\nQuestion:", result['question']['question'])
        print("Conceptual Domains:", result['question'].get('conceptual_domains', []))
        print("Complexity Level:", result['question'].get('complexity_level', 'Not Specified'))

if __name__ == "__main__":
    main()