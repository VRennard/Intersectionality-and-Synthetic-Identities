"""
LLM Prompt Interface for Survey Response Simulation
Supports both OpenAI ChatGPT and local Ollama models
"""

import json
import os
import sys
from typing import List, Dict, Tuple, Optional
import requests
from dataclasses import dataclass

# Try to import OpenAI, but it's optional
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import Anthropic, but it's optional
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Load prompt config (falls back to built-in defaults if file is missing)
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import simulation_config as _sim_cfg
except ImportError:
    _sim_cfg = None


@dataclass
class SurveyQuestion:
    """Represents a survey question with options"""
    question_id: str
    question_text: str
    options: List[str]
    wave: str = "W50"


@dataclass
class DemographicProfile:
    """Represents a demographic profile for simulation"""
    features: List[str]  # e.g., ["Age 18-24", "Income < $30k"]
    
    def format_for_prompt(self) -> str:
        """Format demographic profile for prompt"""
        formatted = ""
        for i, feature in enumerate(self.features, 1):
            formatted += f"        Feature {i}: {feature}\n"
        return formatted


class SurveyDataLoader:
    """Load survey data from JSON file"""
    
    def __init__(self, json_file: str = "survey_responses_W50.json"):
        self.json_file = json_file
        self.questions = []
        self.load_questions()
    
    def load_questions(self):
        """Load questions from JSON file"""
        if not os.path.exists(self.json_file):
            raise FileNotFoundError(f"Survey data file not found: {self.json_file}")
        
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        
        for item in data:
            question = SurveyQuestion(
                question_id=item['question_id'],
                question_text=item['question_text'],
                options=[resp['option'] for resp in item['responses']],
                wave=item.get('wave', 'W50')
            )
            self.questions.append(question)
    
    def get_question(self, question_id: str) -> Optional[SurveyQuestion]:
        """Get a specific question by ID"""
        for q in self.questions:
            if q.question_id == question_id:
                return q
        return None
    
    def get_questions_by_index(self, start: int = 0, end: int = None) -> List[SurveyQuestion]:
        """Get questions by index range"""
        return self.questions[start:end]
    
    def list_questions(self) -> List[Tuple[int, str, str]]:
        """List all questions with index, ID, and text"""
        return [(i, q.question_id, q.question_text) for i, q in enumerate(self.questions)]


class PromptBuilder:
    """Build prompts for LLM survey simulation"""

    # Loaded from simulation_config.py; falls back to built-in default.
    SYSTEM_MESSAGE = (
        getattr(_sim_cfg, 'SYSTEM_PROMPT', None)
        or "You are an expert demographic researcher and data simulator. "
           "Your task is to accurately model how specific, intersecting demographic "
           "groups would respond to survey questions based on sociological trends, "
           "polling data, and community consensus."
    )

    @staticmethod
    def build_prompt(demographic_profile: DemographicProfile, question: SurveyQuestion) -> str:
        """
        Build the user message for the LLM prompt

        Args:
            demographic_profile: DemographicProfile object
            question: SurveyQuestion object

        Returns:
            Formatted user message string
        """
        options_str = "\n".join(
            f"    {i}. {option}"
            for i, option in enumerate(question.options, 1)
        )

        template = getattr(_sim_cfg, 'USER_PROMPT_TEMPLATE', None)
        if template:
            return template.format(
                demographic_features=demographic_profile.format_for_prompt(),
                question_text=question.question_text,
                options=options_str,
            )

        # ── built-in fallback (mirrors simulation_config.py default) ──────────
        return f"""Demographic Profile:

{demographic_profile.format_for_prompt()}

The Task:
Act as a representative modeling this exact community. You are surveying exactly 1,000 individuals who fit this combined profile. Distribute these 1,000 individuals across the multiple-choice options provided below, reflecting how this specific demographic would realistically vote or respond.

The Question:
{question.question_text}

Options:
{options_str}

Output Constraints:

    You must output ONLY a raw Python list of integers. Example: [150, 250, 500, 100]

    Do NOT wrap the output in Markdown code blocks (do not use ```).

    Do NOT include any variable declarations, text, or explanations.

    The list must contain exactly as many integers as there are Options. The order of the integers must exactly match the order of the Options provided above.

    The sum of the integers in the list MUST equal exactly 1000."""


class ResponseParser:
    """Parse LLM responses to extract survey response distributions"""
    
    @staticmethod
    def parse_response(response_text: str, expected_count: int) -> Optional[List[int]]:
        """
        Parse LLM response to extract list of integers
        More lenient parsing with normalization to 1000
        
        Args:
            response_text: Raw response from LLM
            expected_count: Expected number of options
        
        Returns:
            List of integers or None if parsing failed
        """
        response_text = response_text.strip()
        
        # Try to extract list from response
        import re
        match = re.search(r'\[[\s\d,]+\]', response_text)
        if not match:
            return None
        
        list_str = match.group(0)
        try:
            result = eval(list_str)
            if not isinstance(result, list):
                return None
            
            # If count doesn't match, try to normalize
            if len(result) != expected_count:
                # If we have fewer values, pad with zeros
                if len(result) < expected_count:
                    result = result + [0] * (expected_count - len(result))
                # If we have more values, truncate
                elif len(result) > expected_count:
                    result = result[:expected_count]
            
            # Normalize sum to exactly 1000
            current_sum = sum(result)
            if current_sum == 0:
                return None
            
            if current_sum != 1000:
                # Scale to 1000
                result = [int(round(v * 1000 / current_sum)) for v in result]
                # Adjust last element to ensure exact sum of 1000
                current_sum = sum(result)
                if current_sum != 1000:
                    result[-1] += (1000 - current_sum)
            
            return result
        except:
            return None


class LLMInterface:
    """Interface for communicating with LLM models"""
    
    def __init__(self, model_type: str = "ollama", model_name: str = "mistral",
                 openai_api_key: Optional[str] = None, anthropic_api_key: Optional[str] = None):
        """
        Initialize LLM interface

        Args:
            model_type: "openai", "anthropic", or "ollama"
            model_name: Model name (e.g., "gpt-4o", "claude-haiku-4-5-20251001", "mistral")
            openai_api_key: OpenAI API key (required for openai)
            anthropic_api_key: Anthropic API key (required for anthropic)
        """
        self.model_type = model_type.lower()
        self.model_name = model_name

        if self.model_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Install with: pip install openai")
            if not openai_api_key:
                openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass as parameter.")
            openai.api_key = openai_api_key
            self._openai_api_key = openai_api_key

        elif self.model_type == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
            if not anthropic_api_key:
                anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if not anthropic_api_key:
                raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable or pass as parameter.")
            self._anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

        elif self.model_type == "ollama":
            self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            self._test_ollama_connection()
    
    def _test_ollama_connection(self):
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                print(f"Warning: Ollama server may not be responding correctly (status {response.status_code})")
        except requests.exceptions.ConnectionError:
            print(f"Warning: Could not connect to Ollama at {self.ollama_url}")
            print("Make sure Ollama is running. Start with: ollama serve")
    
    def call_model(self, system_message: str, user_message: str) -> str:
        """
        Call the LLM model
        
        Args:
            system_message: System prompt
            user_message: User message
        
        Returns:
            Model response text
        """
        if self.model_type == "openai":
            return self._call_openai(system_message, user_message)
        elif self.model_type == "anthropic":
            return self._call_anthropic(system_message, user_message)
        elif self.model_type == "ollama":
            return self._call_ollama(system_message, user_message)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _call_openai(self, system_message: str, user_message: str) -> str:
        """Call OpenAI API (v1 client)."""
        client = openai.OpenAI(api_key=self._openai_api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_message}
            ],
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    
    def _call_anthropic(self, system_message: str, user_message: str) -> str:
        """Call Anthropic API."""
        response = self._anthropic_client.messages.create(
            model=self.model_name,
            max_tokens=100,
            system=system_message,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text.strip()

    def _call_ollama(self, system_message: str, user_message: str) -> str:
        """Call local Ollama model"""
        url = f"{self.ollama_url}/api/chat"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 50,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        return result['message']['content'].strip()


class SinglePromptSimulator:
    """Simulate responses for a single demographic/question combination"""
    
    def __init__(self, model_type: str = "ollama", model_name: str = "mistral",
                 data_file: str = "survey_responses_W50.json", openai_api_key: Optional[str] = None):
        self.llm = LLMInterface(model_type=model_type, model_name=model_name, 
                               openai_api_key=openai_api_key)
        self.data_loader = SurveyDataLoader(data_file)
        self.prompt_builder = PromptBuilder()
    
    def simulate(self, demographic_profile: DemographicProfile, 
                question: SurveyQuestion, verbose: bool = True) -> Optional[List[int]]:
        """
        Simulate responses for a demographic/question combination
        
        Args:
            demographic_profile: DemographicProfile object
            question: SurveyQuestion object
            verbose: Print progress information
        
        Returns:
            List of response counts or None if failed
        """
        if verbose:
            print(f"\nSimulating for: {', '.join(demographic_profile.features)}")
            print(f"Question: {question.question_text}")
        
        user_message = self.prompt_builder.build_prompt(demographic_profile, question)
        
        if verbose:
            print("Calling LLM...")
        
        response = self.llm.call_model(
            system_message=self.prompt_builder.SYSTEM_MESSAGE,
            user_message=user_message
        )
        
        if verbose:
            print(f"LLM Response: {response[:100]}...")
        
        parsed = ResponseParser.parse_response(response, len(question.options))
        
        if parsed is None:
            if verbose:
                print("Failed to parse response")
            return None
        
        if verbose:
            print(f"Parsed response: {parsed}")
            print(f"Sum: {sum(parsed)}, Length: {len(parsed)}")
        
        return parsed


class BatchPromptSimulator:
    """Simulate responses for multiple demographic/question combinations"""

    def __init__(self, model_type: str = "ollama", model_name: str = "mistral",
                 data_file: str = "survey_responses_W50.json", openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None):
        self.llm = LLMInterface(model_type=model_type, model_name=model_name,
                               openai_api_key=openai_api_key, anthropic_api_key=anthropic_api_key)
        self.data_loader = SurveyDataLoader(data_file)
        self.prompt_builder = PromptBuilder()
        self.results = []
    
    def simulate_batch(self, demographic_profiles: List[DemographicProfile],
                      questions: List[SurveyQuestion], 
                      output_file: Optional[str] = None,
                      verbose: bool = True) -> List[Dict]:
        """
        Simulate responses for multiple demographic/question combinations
        
        Args:
            demographic_profiles: List of DemographicProfile objects
            questions: List of SurveyQuestion objects
            output_file: Optional file to save results as JSON
            verbose: Print progress information
        
        Returns:
            List of result dictionaries
        """
        total = len(demographic_profiles) * len(questions)
        count = 0
        
        for demo_profile in demographic_profiles:
            for question in questions:
                count += 1
                if verbose:
                    print(f"\n[{count}/{total}] Processing: {', '.join(demo_profile.features)}")
                    print(f"  Question: {question.question_id}")
                
                user_message = self.prompt_builder.build_prompt(demo_profile, question)
                
                try:
                    response = self.llm.call_model(
                        system_message=self.prompt_builder.SYSTEM_MESSAGE,
                        user_message=user_message
                    )
                    
                    parsed = ResponseParser.parse_response(response, len(question.options))
                    
                    if parsed is None:
                        if verbose:
                            print("  ❌ Failed to parse response")
                        status = "failed"
                        parsed = []
                    else:
                        if verbose:
                            print(f"  ✓ Success: {parsed}")
                        status = "success"
                    
                    result = {
                        "demographics": demo_profile.features,
                        "question_id": question.question_id,
                        "question_text": question.question_text,
                        "options": question.options,
                        "response_distribution": parsed,
                        "status": status,
                        "raw_response": response[:200] if status == "failed" else None
                    }
                    self.results.append(result)
                    
                except Exception as e:
                    if verbose:
                        print(f"  ❌ Error: {str(e)}")
                    
                    result = {
                        "demographics": demo_profile.features,
                        "question_id": question.question_id,
                        "question_text": question.question_text,
                        "options": question.options,
                        "response_distribution": [],
                        "status": "error",
                        "error": str(e)
                    }
                    self.results.append(result)
        
        if output_file:
            self.save_results(output_file)
        
        return self.results
    
    def save_results(self, output_file: str):
        """Save results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        if len(self.results) > 0:
            successful = sum(1 for r in self.results if r["status"] == "success")
            print(f"\nSaved {len(self.results)} results ({successful} successful) to {output_file}")


# Example usage functions

def example_single_prompt():
    """Example: Single demographic/question simulation"""
    print("=" * 70)
    print("SINGLE PROMPT EXAMPLE")
    print("=" * 70)
    
    # Create simulator (change model_type to "openai" for ChatGPT)
    simulator = SinglePromptSimulator(model_type="ollama", model_name="mistral")
    
    # Create demographic profile
    demographic = DemographicProfile(features=["Age 18-24", "Income < $30k"])
    
    # Get a question from the survey data
    question = simulator.data_loader.questions[0]  # First question
    
    # Simulate
    result = simulator.simulate(demographic, question, verbose=True)
    
    if result:
        print(f"\nFinal Result: {result}")
        print(f"Sum: {sum(result)}")


def example_batch_prompt():
    """Example: Batch processing of multiple demographics and questions"""
    print("=" * 70)
    print("BATCH PROMPT EXAMPLE")
    print("=" * 70)
    
    # Create simulator
    simulator = BatchPromptSimulator(model_type="ollama", model_name="mistral")
    
    # Define demographic profiles to test
    demographic_profiles = [
        DemographicProfile(features=["Age 18-24", "Income < $30k"]),
        DemographicProfile(features=["Age 35-49", "Income $75k-$100k"]),
        DemographicProfile(features=["Age 65+", "Income > $100k"]),
    ]
    
    # Get first 3 questions from survey data
    questions = simulator.data_loader.questions[:3]
    
    # Run batch simulation
    results = simulator.simulate_batch(
        demographic_profiles=demographic_profiles,
        questions=questions,
        output_file="llm_simulation_results.json",
        verbose=True
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    successful = sum(1 for r in results if r["status"] == "success")
    print(f"Total: {len(results)}, Successful: {successful}, Failed: {len(results) - successful}")


if __name__ == "__main__":
    # Run example based on command line argument
    if len(sys.argv) > 1:
        if sys.argv[1] == "single":
            example_single_prompt()
        elif sys.argv[1] == "batch":
            example_batch_prompt()
        else:
            print("Usage: python llm_prompt_survey.py [single|batch]")
    else:
        print("Usage: python llm_prompt_survey.py [single|batch]")
        print("\nExamples:")
        print("  Single prompt: python llm_prompt_survey.py single")
        print("  Batch prompt:  python llm_prompt_survey.py batch")
