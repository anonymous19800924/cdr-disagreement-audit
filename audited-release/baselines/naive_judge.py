#!/usr/bin/env python3
"""
Naive Judge Baseline (Lower Bound): Evaluates worst-case performance by presenting
the judge with only the question and possible answers, without any expert arguments
or reasoning traces. This baseline helps identify if the non-expert model has prior
knowledge of the dataset (contamination).

This script:
1. Loads aligned cases from two consultant trace files
2. Uses a judge model to answer questions directly (no reasoning traces provided)
3. Evaluates accuracy to detect potential contamination
4. Saves results with the judge's answers and accuracy
"""

import os
import requests
import json
import re
import random
from dotenv import load_dotenv
from pathlib import Path
from argparse import ArgumentParser
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type, before_sleep
from tqdm import tqdm

# Load environment variables - try multiple locations
script_dir = Path(__file__).parent
project_root = script_dir.parent

# Try loading .env from multiple locations
env_loaded = False
for env_path in [
    script_dir / '.env',
    project_root / '.env',
    project_root / 'backend' / '.env',
]:
    if env_path.exists():
        load_dotenv(env_path)
        env_loaded = True
        break

# Also try default load_dotenv() which searches current and parent directories
if not env_loaded:
    load_dotenv()

# Custom exception for parsing failures
class ParsingError(Exception):
    """Raised when judge response cannot be parsed"""
    pass

# Get API key from environment variables
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    error_msg = (
        "OPENROUTER_API_KEY not found in environment variables.\n"
        "Please set it either:\n"
        "  1. As an environment variable: export OPENROUTER_API_KEY='your-key'\n"
        "  2. In a .env file in the project root or baselines/ directory"
    )
    raise ValueError(error_msg)

# Default judge model to use
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"


def load_prompt():
    """Load prompt from naive_judge.txt file"""
    script_dir = Path(__file__).parent
    prompt_path = script_dir / 'prompts' / 'naive_judge.txt'
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        return prompt_template
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")


def extract_model_name_from_filename(filepath):
    """
    Extract model name from filename.
    Expected formats include consultant trace files named as
    GPQA_Reasoning_Traces_{provider}_{model}_{dataset}.json.
    Returns: {provider}/{model}
    """
    filename = Path(filepath).stem  # Get filename without extension
    # Remove prefix and suffix
    if filename.startswith('GPQA_Reasoning_Traces_'):
        # Handle both _disagreements and _all_198 formats
        if filename.endswith('_disagreements'):
            model_part = filename[len('GPQA_Reasoning_Traces_'):-len('_disagreements')]
        elif filename.endswith('_all_198'):
            model_part = filename[len('GPQA_Reasoning_Traces_'):-len('_all_198')]
        else:
            # Try to extract by removing prefix only
            model_part = filename[len('GPQA_Reasoning_Traces_'):]
            # Remove any trailing pattern like _all_198, _disagreements, etc.
            for suffix in ['_all_198', '_disagreements', '_all']:
                if model_part.endswith(suffix):
                    model_part = model_part[:-len(suffix)]
                    break
        
        # Split by underscore - first part is provider, rest is model
        parts = model_part.split('_', 1)
        if len(parts) == 2:
            provider, model = parts
            return f"{provider}/{model}"
        else:
            # Fallback if format is different
            return model_part.replace('_', '/')
    return None


def load_model_data(model1_path, model2_path):
    """
    Load all cases from both model files.
    
    Args:
        model1_path: Path to first model's JSON file
        model2_path: Path to second model's JSON file
    
    Returns:
        Tuple of (matched_cases, MODEL1_NAME, MODEL2_NAME)
    """
    # Convert string paths to Path objects if needed
    model1_path = Path(model1_path)
    model2_path = Path(model2_path)
    
    # Check if files exist
    if not model1_path.exists():
        raise FileNotFoundError(f"Model 1 file not found: {model1_path}")
    if not model2_path.exists():
        raise FileNotFoundError(f"Model 2 file not found: {model2_path}")
    
    # Extract model names from filenames
    MODEL1_NAME = extract_model_name_from_filename(model1_path)
    MODEL2_NAME = extract_model_name_from_filename(model2_path)
    
    if not MODEL1_NAME or not MODEL2_NAME:
        raise ValueError(f"Could not extract model names from filenames: {model1_path}, {model2_path}")
    
    print(f"Loading model1 data from: {model1_path} (model: {MODEL1_NAME})")
    with open(model1_path, 'r', encoding='utf-8') as f:
        model1_data = json.load(f)
    print(f"Loaded {len(model1_data)} entries from model1")
    
    print(f"Loading model2 data from: {model2_path} (model: {MODEL2_NAME})")
    with open(model2_path, 'r', encoding='utf-8') as f:
        model2_data = json.load(f)
    print(f"Loaded {len(model2_data)} entries from model2")
    
    # Create index mapping for faster lookup
    model1_by_index = {item['index']: item for item in model1_data}
    model2_by_index = {item['index']: item for item in model2_data}
    
    # Match cases by index (all cases, not just disagreements)
    matched_cases = []
    for index in sorted(set(model1_by_index.keys()) & set(model2_by_index.keys())):
        matched_cases.append({
            'index': index,
            'model1': model1_by_index[index],
            'model2': model2_by_index[index]
        })
    
    print(f"Found {len(matched_cases)} matched cases")
    return matched_cases, MODEL1_NAME, MODEL2_NAME


def format_prompt(question, choices):
    """Format the prompt with only question and choices (no reasoning traces)"""
    prompt_template = load_prompt()
    
    # Replace placeholders
    formatted_prompt = prompt_template.replace('{{QUESTION}}', question)
    formatted_prompt = formatted_prompt.replace('{{CHOICES}}', choices)
    
    return formatted_prompt


def is_retryable_exception(e):
    """Check if exception is retryable"""
    if isinstance(e, requests.exceptions.RequestException):
        return True
    if hasattr(e, "response") and e.response is not None:
        # Retry 429 or 5xx errors
        return e.response.status_code in [429, 500, 502, 503]
    return False


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(Exception)
)
def call_judge(prompt, judge_model):
    """Call OpenRouter API to get judge's answer"""
    # Some providers reject developer/system instructions. Minimal workaround:
    # avoid system messages for those models.
    jm = (judge_model or "").strip().lower()
    disable_system = ("gemma-3-4b-it" in jm)

    if disable_system:
        messages = [{"role": "user", "content": "You are a helpful assistant. Answer questions directly and concisely.\n\n" + prompt}]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer questions directly and concisely."},
            {"role": "user", "content": prompt},
        ]

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openrouter_api_key}",
        },
        data=json.dumps({
            "model": judge_model,
            "messages": messages,
            "temperature": 0,
            "n": 1
        }),
    )
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None
    
    response_data = response.json()
    
    if 'choices' in response_data and len(response_data['choices']) > 0:
        message_content = response_data['choices'][0]['message']['content']
        return message_content
    else:
        print("No choices found in response")
        print(f"Full response: {response_data}")
        return None


def parse_judge_response(response_text):
    """Parse the judge's response to extract the answer choice"""
    if not response_text:
        raise ParsingError("Empty response from judge")
    
    answer_patterns = [
        r'Answer:\s*([A-Z])',
        r'answer:\s*([A-Z])',
        r'Choice:\s*([A-Z])',
        r'choice:\s*([A-Z])',
        r'^([A-Z])\.?\s*$',
        r'^([A-Z])\s*$',
        r'\b([A-Z])\b',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response_text.strip(), re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1).upper()
            if answer.isalpha() and len(answer) == 1:
                return answer, response_text
    
    preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
    raise ParsingError(f"Could not parse judge response. Preview: {preview}")


def log_retry_attempt(retry_state):
    """Log retry attempts for parsing errors"""
    if retry_state.outcome and retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        attempt_num = retry_state.attempt_number
        print(f"  [Retry {attempt_num}/3] Parsing failed: {str(exception)[:100]}...")


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(ParsingError),
    before_sleep=log_retry_attempt
)
def call_and_parse_judge(prompt, judge_model):
    """
    Call the judge API and parse the response with automatic retry on parsing errors.
    
    Args:
        prompt: The formatted prompt to send to the judge
        judge_model: The model to use as the judge
        
    Returns:
        Tuple of (answer_choice, full_response) where answer_choice is a single letter (A, B, C, D, etc.)
        
    Raises:
        ParsingError: If response cannot be parsed after all retries
    """
    response_text = call_judge(prompt, judge_model)
    if not response_text:
        raise ParsingError("No response from judge API")
    
    answer_choice, full_response = parse_judge_response(response_text)
    return answer_choice, full_response


def main():
    parser = ArgumentParser(description='Naive Judge Baseline (Lower Bound)')
    parser.add_argument("--model1_file", type=str, required=True, 
                       help="Path to the first consultant trace file")
    parser.add_argument("--model2_file", type=str, required=True,
                       help="Path to the second consultant trace file")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true", help="Debug mode: only process first 5 cases")
    parser.add_argument("--output", type=str, default=None, help="Output filename (optional)")
    parser.add_argument("--judge_model", type=str, default=DEFAULT_JUDGE_MODEL, 
                       help=f"Judge model to use (default: {DEFAULT_JUDGE_MODEL})")
    args = parser.parse_args()
    
    print(f"Using judge model: {args.judge_model}")
    
    random.seed(args.random_seed)
    
    matched_cases, model1_name, model2_name = load_model_data(args.model1_file, args.model2_file)
    
    if args.debug:
        print("DEBUG MODE: Processing only the first 5 cases")
        matched_cases = matched_cases[:5]
    
    script_dir = Path(__file__).parent
    output_dir = script_dir / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output:
        output_filename = output_dir / args.output
    else:
        judge_model_safe = args.judge_model.replace('/', '_')
        model1_file_stem = Path(args.model1_file).stem
        if 'GPQA_Reasoning_Traces_' in model1_file_stem:
            dataset_id = model1_file_stem.split('GPQA_Reasoning_Traces_')[1]
            parts = dataset_id.split('_')
            if len(parts) >= 3:
                dataset_start_idx = 2
                if len(parts) > 3 and not any(part in ['main', 'all', 'diamond'] for part in parts[:3]):
                    dataset_start_idx = 3
                dataset_id = '_'.join(parts[dataset_start_idx:])
            else:
                dataset_id = dataset_id
        else:
            import hashlib
            dataset_id = hashlib.md5(model1_file_stem.encode()).hexdigest()[:8]
        
        output_filename = output_dir / f'naive_judge_results_{judge_model_safe}_{dataset_id}_seed{args.random_seed}.json'
    
    results = []
    if os.path.exists(output_filename):
        print(f"Found existing file: {output_filename}")
        with open(output_filename, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")
    
    processed_indices = {result['index'] for result in results}
    remaining_cases = [case for case in matched_cases if case['index'] not in processed_indices]
    
    if len(remaining_cases) == 0:
        print("All cases already processed!")
        if results:
            total = len(results)
            correct_answers = sum(1 for r in results if r.get('judge_correct', False) == True)
            accuracy = correct_answers / total * 100 if total > 0 else 0
            
            answer_counts = {}
            for r in results:
                answer = r.get('judge_answer', 'Unknown')
                answer_counts[answer] = answer_counts.get(answer, 0) + 1
            
            print(f"\nSUMMARY STATISTICS:")
            print(f"Total cases: {total}")
            print(f"Correct answers: {correct_answers} ({accuracy:.1f}%)")
            print(f"Answer distribution:")
            for answer, count in sorted(answer_counts.items()):
                print(f"  {answer}: {count} ({count/total*100:.1f}%)")
            
            model1_correct_count = sum(1 for r in results if r.get('model1_correct', False) == True)
            model2_correct_count = sum(1 for r in results if r.get('model2_correct', False) == True)
            print(f"\nComparison with expert models:")
            print(f"Model 1 ({model1_name}) accuracy: {model1_correct_count}/{total} ({model1_correct_count/total*100:.1f}%)")
            print(f"Model 2 ({model2_name}) accuracy: {model2_correct_count}/{total} ({model2_correct_count/total*100:.1f}%)")
            print(f"Naive judge accuracy: {correct_answers}/{total} ({accuracy:.1f}%)")
            
            if accuracy > 50:
                print(f"\n⚠️  WARNING: Naive judge accuracy ({accuracy:.1f}%) is above random chance.")
                print(f"This may indicate potential contamination in the judge model's training data.")
        return
    
    print(f"Processing {len(remaining_cases)} remaining cases out of {len(matched_cases)} total")
    
    for case in tqdm(remaining_cases, desc="Processing cases"):
        index = case['index']
        model1_data = case['model1']
        model2_data = case['model2']
        
        question = model1_data.get('Question', '')
        choices = model1_data.get('choices', '')
        correct_choice = model1_data.get('correct_choice', '')
        
        prompt = format_prompt(question, choices)
        
        tqdm.write(f"\nProcessing case {index}...")
        tqdm.write(f"Correct answer: {correct_choice}")
        
        try:
            judge_answer, full_response = call_and_parse_judge(prompt, args.judge_model)
            
            judge_correct = (judge_answer.upper() == correct_choice.upper())
            
            model1_correct = model1_data.get('model_correct', False)
            model2_correct = model2_data.get('model_correct', False)
            
            result = {
                'index': index,
                'question': question,
                'choices': choices,
                'correct_choice': correct_choice,
                'judge_model': args.judge_model,
                'judge_answer': judge_answer,
                'judge_correct': judge_correct,
                'raw_prompt': prompt,
                'judge_response': full_response,
                'model1_name': model1_name,
                'model1_answer': model1_data.get('model_answer'),
                'model1_correct': model1_correct,
                'model2_name': model2_name,
                'model2_answer': model2_data.get('model_answer'),
                'model2_correct': model2_correct,
            }
            
            results.append(result)
            
            tqdm.write(f"Judge answer: {judge_answer}")
            tqdm.write(f"Judge correct: {judge_correct}")
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        except ParsingError as e:
            tqdm.write(f"Failed to parse judge response after 3 retries for case {index}: {e}")
            tqdm.write("Skipping this case - consider checking the judge's response format")
        except Exception as e:
            tqdm.write(f"Error processing case {index}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nProcessing complete! Total results: {len(results)}")
    print(f"Results saved to {output_filename}")
    
    if results:
        total = len(results)
        correct_answers = sum(1 for r in results if r.get('judge_correct', False) == True)
        accuracy = correct_answers / total * 100 if total > 0 else 0
        
        answer_counts = {}
        for r in results:
            answer = r.get('judge_answer', 'Unknown')
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        print(f"\nSUMMARY STATISTICS:")
        print(f"Total cases: {total}")
        print(f"Correct answers: {correct_answers} ({accuracy:.1f}%)")
        print(f"Answer distribution:")
        for answer, count in sorted(answer_counts.items()):
            print(f"  {answer}: {count} ({count/total*100:.1f}%)")
        
        model1_correct_count = sum(1 for r in results if r.get('model1_correct', False) == True)
        model2_correct_count = sum(1 for r in results if r.get('model2_correct', False) == True)
        print(f"\nComparison with expert models:")
        print(f"Model 1 ({model1_name}) accuracy: {model1_correct_count}/{total} ({model1_correct_count/total*100:.1f}%)")
        print(f"Model 2 ({model2_name}) accuracy: {model2_correct_count}/{total} ({model2_correct_count/total*100:.1f}%)")
        print(f"Naive judge accuracy: {correct_answers}/{total} ({accuracy:.1f}%)")
        
        if accuracy > 50:
            print(f"\n⚠️  WARNING: Naive judge accuracy ({accuracy:.1f}%) is above random chance.")
            print(f"This may indicate potential contamination in the judge model's training data.")


if __name__ == "__main__":
    main()
