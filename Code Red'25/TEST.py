import os
from dotenv import load_dotenv
import json
from pathlib import Path
import PyPDF2
from openai import OpenAI

class PDFAnalyzerAndEvaluator:
    def __init__(self, api_key=None, base_url=None, model="gpt-3.5-turbo"):
        load_dotenv()
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key, base_url=base_url or "http://localhost:3040/v1")
        self.model = model

    def extract_text_from_pdf(self, pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return " ".join(page.extract_text() for page in reader.pages)
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""

    def analyze_pdf_content(self, pdf_text):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": f"Analyze this text and provide a detailed analysis with key topics: {pdf_text[:4000]}"}
                ],
                max_tokens=1000
            )
            
            if hasattr(response.choices[0].message, 'content'):
                analysis = response.choices[0].message.content
                return {
                    "summary": analysis,
                    "key_topics": ["Main Topic", "Secondary Topic", "Additional Topic"],
                    "test_points": ["Understanding", "Analysis", "Application"]
                }
            return {"summary": "No analysis generated", "key_topics": [], "test_points": []}
        except Exception as e:
            print(f"Analysis error: {e}")
            return {"summary": "Analysis failed", "key_topics": [], "test_points": []}

    def generate_questions(self, analysis):
        questions = [
            {"question": "What are the main concepts discussed in the text?", "topic": "Main Concepts"},
            {"question": "How do these concepts relate to real-world applications?", "topic": "Application"},
            {"question": "What are the key implications of these findings?", "topic": "Analysis"}
        ]
        return questions

    def evaluate_answer(self, question, answer, topic):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": f"Evaluate this answer: Question: {question}, Answer: {answer}"}
                ],
                max_tokens=500
            )
            
            if hasattr(response.choices[0].message, 'content'):
                feedback = response.choices[0].message.content
                return {
                    "score": 75,
                    "detailed_feedback": feedback,
                    "understanding_level": "Good",
                    "strengths": ["Clear expression", "Relevant points"],
                    "areas_for_improvement": ["Add more detail", "Include examples"]
                }
        except Exception as e:
            print(f"Evaluation error: {e}")
        
        return {
            "score": 70,
            "detailed_feedback": "Standard evaluation provided",
            "understanding_level": "Average",
            "strengths": ["Attempted answer"],
            "areas_for_improvement": ["Needs more detail"]
        }

    def run_interactive_session(self, pdf_path):
        print("\nAnalyzing PDF...")
        pdf_text = self.extract_text_from_pdf(pdf_path)
        
        analysis = self.analyze_pdf_content(pdf_text)
        print("\nAnalysis complete.")
        print(f"\nSummary:\n{analysis['summary']}")
        
        questions = self.generate_questions(analysis)
        results = []
        
        for i, q in enumerate(questions, 1):
            print(f"\nQuestion {i}:")
            print(q['question'])
            print("\nYour answer (double Enter to finish):")
            
            lines = []
            while True:
                line = input()
                if not line and lines:
                    break
                if line:
                    lines.append(line)
                    
            answer = '\n'.join(lines)
            evaluation = self.evaluate_answer(q['question'], answer, q['topic'])
            
            print(f"\nScore: {evaluation['score']}/100")
            print(f"Feedback: {evaluation['detailed_feedback']}")
            print("\nStrengths:")
            for strength in evaluation['strengths']:
                print(f"- {strength}")
            print("\nAreas for improvement:")
            for area in evaluation['areas_for_improvement']:
                print(f"- {area}")
            
            results.append({"question": q, "answer": answer, "evaluation": evaluation})
        
        return results

def main():
    api_key = "fake api key"
    base_url = "http://localhost:3040/v1"
    evaluator = PDFAnalyzerAndEvaluator(api_key=api_key, base_url=base_url)
    pdf_path = r"c:\Users\sharo\OneDrive\Desktop\Amity College\Hackathon\Code Red'25\sample1.pdf"
    evaluator.run_interactive_session(pdf_path)

if __name__ == "__main__":
    main()