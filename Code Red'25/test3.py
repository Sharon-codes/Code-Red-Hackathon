import os
import json
import fitz  # PyMuPDF
import spacy
import requests
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict

class PDFQuestionEvaluator:
    def __init__(self, nlp_model='en_core_web_sm', embedding_model='all-MiniLM-L6-v2', hf_api_key=None):
        """
        Initialize PDF processing with NLP for question generation and advanced embedding models for evaluation.
        """
        # Load spaCy model
        try:
            self.nlp = spacy.load(nlp_model)
        except OSError:
            print(f"Downloading {nlp_model} spaCy model...")
            spacy.cli.download(nlp_model)
            self.nlp = spacy.load(nlp_model)

        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Hugging Face API key
        self.hf_api_key = hf_api_key

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using PyMuPDF and return as a single string.
        """
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text().strip() + "\n"
            doc.close()
            return full_text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def generate_questions_with_hf(self, text: str, num_questions: int = 5) -> List[str]:
        """
        Generate high-quality questions using Hugging Face's hosted models.
        """
        if not self.hf_api_key:
            raise ValueError("Hugging Face API key is required for this functionality.")

        url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}

        # Truncate text to fit within model token limits (approximately 2000 characters)
        truncated_text = text[:2000]
        prompt = f"Generate {num_questions} thoughtful, quiz-style questions from the following text:\n{truncated_text}"

        try:
            print("Sending request to Hugging Face API...")
            response = requests.post(url, headers=headers, json={"inputs": prompt})
            response.raise_for_status()

            # Parse response
            result = response.json()
            print(f"API Response: {result}")

            if isinstance(result, list) and 'generated_text' in result[0]:
                questions = result[0]['generated_text'].strip().split("\n")
                return [q.strip() for q in questions if q.strip()]
            else:
                raise ValueError(f"Unexpected API response format: {result}")

        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}")
            return ["Error: Failed to generate questions using AI."]
        except Exception as e:
            print(f"Error parsing API response: {e}")
            return ["Error: Unexpected issue while processing API response."]

    def evaluate_answer(self, reference_text: str, question: str, student_answer: str) -> Dict:
        """
        Evaluate the student's answer using semantic similarity with embeddings.
        """
        # Encode reference text, question, and student answer
        embeddings = self.embedding_model.encode([reference_text, question, student_answer], convert_to_tensor=True)

        # Compute similarity scores
        ref_to_answer = util.pytorch_cos_sim(embeddings[0], embeddings[2]).item()
        question_to_answer = util.pytorch_cos_sim(embeddings[1], embeddings[2]).item()

        # Combine scores and scale to 100
        score = int(((ref_to_answer + question_to_answer) / 2) * 100)

        strengths = []
        improvements = []

        if score > 75:
            strengths.append("Answer is highly relevant and well-aligned with the context.")
            improvements.append("Consider providing more nuanced insights.")
        elif score > 50:
            strengths.append("Answer shows a basic understanding of the topic.")
            improvements.append("Expand on key points and provide more specific examples.")
        else:
            strengths.append("Answer attempts to address the question but lacks depth.")
            improvements.append("Focus on the main themes and provide more detailed responses.")

        return {
            "total_score": score,
            "strengths": strengths,
            "improvement_recommendations": improvements
        }

def interactive_questionnaire(pdf_path: str, hf_api_key: str):
    """
    Interactive questionnaire to generate questions, collect answers, and evaluate them.
    """
    evaluator = PDFQuestionEvaluator(hf_api_key=hf_api_key)

    # Extract text from the PDF
    pdf_text = evaluator.extract_text_from_pdf(pdf_path)

    if not pdf_text:
        print("Failed to extract text from the PDF.")
        return

    # Generate questions using Hugging Face API
    questions = evaluator.generate_questions_with_hf(pdf_text)

    print("\n==== Interactive Questionnaire ====")
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")

        # Get student's answer
        student_answer = input("Your Answer: ")

        # Evaluate the answer
        evaluation = evaluator.evaluate_answer(pdf_text, question, student_answer)

        # Display evaluation
        print("\n---- Evaluation ----")
        print(f"Score: {evaluation['total_score']} / 100")
        print("Strengths:")
        for strength in evaluation['strengths']:
            print(f"- {strength}")
        print("Improvement Recommendations:")
        for recommendation in evaluation['improvement_recommendations']:
            print(f"- {recommendation}")

        results.append({
            "question": question,
            "answer": student_answer,
            "evaluation": evaluation
        })

    # Save results if desired
    save_choice = input("Do you want to save the results? (yes/no): ").strip().lower()
    if save_choice in ['yes', 'y']:
        output_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'evaluation_results.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    pdf_path = input("Enter the path to the PDF file: ").strip()
    hf_api_key = input("Enter your Hugging Face API key: ").strip()
    interactive_questionnaire(pdf_path, hf_api_key)
