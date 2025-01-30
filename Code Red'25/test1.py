import os
import torch
import PyPDF2
import json
import re
import nltk

from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class HuggingFacePDFQuestionEvaluator:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.3"):
        """
        Initialize with a high-performance open-source model from Hugging Face
        
        :param model_name: Hugging Face model identifier
        """
        # Download NLTK resources
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            print(f"NLTK download warning: {e}")

        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer,
                
            )
        except Exception as e:
            print(f"Model loading error: {e}")
            raise

    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file
        
        :param pdf_path: Path to the PDF file
        :return: Extracted text as a string
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                for page in pdf_reader.pages:
                    full_text += page.extract_text() + "\n"
                return full_text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def generate_questions(self, pdf_path, num_questions=5):
        """
        Generate sophisticated questions based on PDF content
        
        :param pdf_path: Path to the PDF file
        :param num_questions: Number of questions to generate
        :return: List of generated questions
        """
        # Extract text from PDF
        full_text = self.extract_text_from_pdf(pdf_path)
        
        # Preprocess text (split into sentences)
        sentences = sent_tokenize(full_text)
        
        # Prepare prompt for question generation
        prompt = f"""
        Analyze the following academic text and generate {num_questions} sophisticated, 
        multi-layered, and intellectually challenging questions that:
        - Require deep critical thinking
        - Explore complex conceptual relationships
        - Demand comprehensive understanding
        - Encourage interdisciplinary analysis

        Text Context:
        {' '.join(sentences[:10])}  # Use first few sentences as context

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
            # Generate questions
            generation_args = {
                "max_new_tokens": 512,
                "return_full_text": False,
                "temperature": 0.7,
                "do_sample": True,
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
                    # Fallback parsing
                    return [
                        {
                            "question": f"Analyze the complex relationships in the given text about {pdf_path.split('/')[-1]}",
                            "conceptual_domains": ["Interdisciplinary Analysis"],
                            "complexity_level": "High"
                        }
                    ]
            else:
                # Fallback questions
                return [
                    {
                        "question": f"Analyze the complex relationships in the given text about {pdf_path.split('/')[-1]}",
                        "conceptual_domains": ["Interdisciplinary Analysis"],
                        "complexity_level": "High"
                    }
                ]

        except Exception as e:
            print(f"Error generating questions: {e}")
            # Fallback questions
            return [
                {
                    "question": f"Analyze the complex relationships in the given text about {pdf_path.split('/')[-1]}",
                    "conceptual_domains": ["Interdisciplinary Analysis"],
                    "complexity_level": "High"
                }
            ]

    def evaluate_answer(self, question, student_answer, pdf_text):
        """
        Perform comprehensive, nuanced answer evaluation
        
        :param question: Generated question dictionary
        :param student_answer: Student's submitted answer
        :param pdf_text: Original PDF text for context
        :return: Multidimensional evaluation
        """
        # Create a sophisticated evaluation prompt
        evaluation_prompt = f"""
        Conduct an ultra-precise, multidimensional evaluation of the following academic response:

        ORIGINAL QUESTION:
        {question['question']}

        CONCEPTUAL DOMAINS:
        {', '.join(question.get('conceptual_domains', ['Undefined']))}

        STUDENT ANSWER:
        {student_answer}

        CONTEXTUAL REFERENCE TEXT:
        {pdf_text[:1000]}  # Include some context from original text

        Comprehensive Evaluation Criteria:
        1. Conceptual Accuracy (40 points)
        2. Critical Thinking Depth (25 points)
        3. Analytical Sophistication (20 points)
        4. Contextual Relevance (15 points)

        Output a detailed JSON evaluation focusing on:
        - Total score (0-100)
        - Criterion-specific scores
        - Key strengths
        - Improvement recommendations

        Evaluation JSON:
        """

        try:
            # Generate evaluation
            generation_args = {
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.85
            }
            
            full_response = self.generator(evaluation_prompt, **generation_args)[0]['generated_text']
            
            # Extract JSON-like structure
            eval_match = re.findall(r'\{.*?\}', full_response, re.DOTALL)
            
            if eval_match:
                try:
                    evaluation_dict = eval(eval_match[0])
                    return evaluation_dict
                except:
                    # Fallback evaluation
                    return self._get_fallback_evaluation()
            else:
                return self._get_fallback_evaluation()

        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return self._get_fallback_evaluation()

    def _get_fallback_evaluation(self):
        """
        Provides a standard fallback evaluation
        
        :return: Fallback evaluation dictionary
        """
        return {
            "total_score": 75,
            "criterion_scores": {
                "conceptual_accuracy": 30,
                "critical_thinking": 20,
                "analytical_sophistication": 15,
                "contextual_relevance": 10
            },
            "strengths": ["Shows basic understanding", "Attempts to address key points"],
            "improvement_recommendations": ["Deepen analysis", "Provide more contextual references"]
        }

    def process_pdf(self, pdf_path, student_answers=None):
        """
        Comprehensive PDF processing workflow
        
        :param pdf_path: Path to PDF file
        :param student_answers: Optional list of student answers
        :return: Processed results
        """
        # Extract full PDF text
        full_text = self.extract_text_from_pdf(pdf_path)
        
        # Generate questions
        questions = self.generate_questions(pdf_path)
        
        # Prepare results
        results = []
        
        # If no student answers provided, create placeholders
        if not student_answers:
            student_answers = ["" for _ in questions]
        
        # Evaluate each question-answer pair
        for question, student_answer in zip(questions, student_answers):
            evaluation = self.evaluate_answer(question, student_answer, full_text)
            
            results.append({
                "question": question,
                "student_answer": student_answer,
                "evaluation": evaluation
            })
        
        # Save results
        self.save_results(results, pdf_path)
        
        return results

    def save_results(self, results, pdf_path, filename=None):
        """
        Save comprehensive evaluation results
        
        :param results: Processed results
        :param pdf_path: Source PDF path
        :param filename: Optional custom filename
        """
        if not filename:
            filename = f"evaluation_results_{pdf_path.split('/')[-1].replace('.pdf', '')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"Comprehensive evaluation results saved to {filename}")

def main():
    # Initialize advanced PDF question evaluator
    evaluator = HuggingFacePDFQuestionEvaluator()
    
    # Path to PDF (replace with your actual PDF path)
    pdf_path = r"c:\Users\sharo\OneDrive\Desktop\Amity College\Hackathon\Code Red'25\sample1.pdf"

    
    # Optional: Provide student answers corresponding to generated questions
    student_answers = [
        "A comprehensive student response addressing the question's complexity.",
        "Another detailed answer demonstrating critical thinking.",
        "A third response exploring the conceptual nuances."
    ]
    
    # Process PDF and generate questions with evaluations
    results = evaluator.process_pdf(pdf_path, student_answers)
    
    # Print results
    for result in results:
        print("\nQuestion:", result['question']['question'])
        print("Score:", result['evaluation'].get('total_score', 'N/A'))
        print("Strengths:", result['evaluation'].get('strengths', []))
        print("Improvements:", result['evaluation'].get('improvement_recommendations', []))

if __name__ == "__main__":
    main()