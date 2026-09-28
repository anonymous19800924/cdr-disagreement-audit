#!/usr/bin/env python3
"""
Double Consultancy Baseline: Use a judge (GPT-4o-mini) to determine which consultant
(model) provides the correct answer when comparing two models.

This script:
1. Loads aligned cases from two consultant trace files
2. Randomizes which model is consultant 1 and consultant 2
3. Uses a judge model (default: GPT-4o-mini) to evaluate which consultant is correct
4. Saves results with the judge's decision
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
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini" #


def load_prompt():
    """Load prompt from double_consultancy.txt file"""
    script_dir = Path(__file__).parent
    prompt_path = script_dir / 'prompts' / 'double_consultancy.txt'
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        return prompt_template
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")


def get_model_name_key(model_name):
    """Convert model name to a safe key name for JSON"""
    # Convert to lowercase and replace special characters with underscores
    key = model_name.lower().replace('-', '_').replace('/', '_').replace(' ', '_')
    return key


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


def format_reasoning_trace(reasoning_list):
    """Format reasoning list as a numbered list"""
    if isinstance(reasoning_list, list):
        return '\n'.join([f"{i+1}. {step}" for i, step in enumerate(reasoning_list)])
    return str(reasoning_list)


def format_prompt(question, choices, model1_data, model2_data):
    """Format the prompt with question, choices, and both consultants' responses"""
    prompt_template = load_prompt()
    
    # Format reasoning traces
    model1_reasoning = format_reasoning_trace(model1_data.get('model_reasoning', []))
    model2_reasoning = format_reasoning_trace(model2_data.get('model_reasoning', []))
    
    # Replace placeholders
    formatted_prompt = prompt_template.replace('{{QUESTION}}', question)
    formatted_prompt = formatted_prompt.replace('{{CHOICES}}', choices)
    formatted_prompt = formatted_prompt.replace('{{MODEL_1_ANSWER}}', model1_data.get('model_answer', 'Unknown'))
    formatted_prompt = formatted_prompt.replace('{{MODEL_1_REASONING}}', model1_reasoning)
    formatted_prompt = formatted_prompt.replace('{{MODEL_2_ANSWER}}', model2_data.get('model_answer', 'Unknown'))
    formatted_prompt = formatted_prompt.replace('{{MODEL_2_REASONING}}', model2_reasoning)
    
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
    """Call OpenRouter API to get judge's decision"""
    # Some providers reject developer/system instructions. Minimal workaround:
    # avoid system messages for those models.
    jm = (judge_model or "").strip().lower()
    disable_system = ("gemma-3-4b-it" in jm)

    if disable_system:
        messages = [{"role": "user", "content": "You are a very intelligent assistant, who follows instructions directly.\n\n" + prompt}]
    else:
        messages = [
            {"role": "system", "content": "You are a very intelligent assistant, who follows instructions directly."},
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
    """Parse the judge's response to extract the winner"""
    if not response_text:
        raise ParsingError("Empty response from judge")
    
    winner_pattern = r'Winner:\s*(Consultant\s*1|Consultant\s*2|Neither)'
    match = re.search(winner_pattern, response_text, re.IGNORECASE)
    
    if match:
        winner = match.group(1).strip()
        if '1' in winner:
            return 'Consultant 1', response_text
        elif '2' in winner:
            return 'Consultant 2', response_text
        else:
            return 'Neither', response_text
    
    patterns = [
        r'winner:\s*(consultant\s*1|consultant\s*2|neither)',
        r'choosing:\s*(consultant\s*1|consultant\s*2|neither)',
        r'decision:\s*(consultant\s*1|consultant\s*2|neither)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            winner = match.group(1).strip()
            if '1' in winner:
                return 'Consultant 1', response_text
            elif '2' in winner:
                return 'Consultant 2', response_text
            else:
                return 'Neither', response_text
    
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
        Tuple of (winner, full_response) where winner is 'Consultant 1', 'Consultant 2', or 'Neither'
        
    Raises:
        ParsingError: If response cannot be parsed after all retries
    """
    response_text = call_judge(prompt, judge_model)
    if not response_text:
        raise ParsingError("No response from judge API")
    
    winner, full_response = parse_judge_response(response_text)
    return winner, full_response


def main():
    parser = ArgumentParser(description='Double Consultancy Baseline')
    parser.add_argument("--model1_file", type=str, required=True, 
                       help="Path to the first consultant trace file")
    parser.add_argument("--model2_file", type=str, required=True,
                       help="Path to the second consultant trace file")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for model assignment")
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
        
        output_filename = output_dir / f'double_consultancy_results_{judge_model_safe}_{dataset_id}_seed{args.random_seed}.json'
    
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
            consultant1_wins = sum(1 for r in results if r.get('judge_winner') == 'Consultant 1')
            consultant2_wins = sum(1 for r in results if r.get('judge_winner') == 'Consultant 2')
            neither_wins = sum(1 for r in results if r.get('judge_winner') == 'Neither')
            correct_decisions = sum(1 for r in results if r.get('winning_correct') == True)
            judge_accuracy = sum(1 for r in results if r.get('judge_correct', False) == True)
            
            print(f"\nSUMMARY STATISTICS:")
            print(f"Total cases: {total}")
            print(f"Consultant 1 wins: {consultant1_wins} ({consultant1_wins/total*100:.1f}%)")
            print(f"Consultant 2 wins: {consultant2_wins} ({consultant2_wins/total*100:.1f}%)")
            print(f"Neither wins: {neither_wins} ({neither_wins/total*100:.1f}%)")
            print(f"Judge selected correct answer: {correct_decisions} ({correct_decisions/total*100:.1f}%)")
            print(f"Judge accuracy: {judge_accuracy}/{total} ({judge_accuracy/total*100:.1f}%)")
        return
    
    print(f"Processing {len(remaining_cases)} remaining cases out of {len(matched_cases)} total")
    
    for case in tqdm(remaining_cases, desc="Processing cases"):
        index = case['index']
        model1_data = case['model1']
        model2_data = case['model2']
        
        if random.random() < 0.5:
            consultant1_data = model1_data
            consultant2_data = model2_data
            consultant1_name = model1_name
            consultant2_name = model2_name
        else:
            consultant1_data = model2_data
            consultant2_data = model1_data
            consultant1_name = model2_name
            consultant2_name = model1_name
        
        question = consultant1_data.get('Question', '')
        choices = consultant1_data.get('choices', '')
        
        prompt = format_prompt(question, choices, consultant1_data, consultant2_data)
        
        tqdm.write(f"\nProcessing case {index}...")
        tqdm.write(f"Consultant 1 ({consultant1_name}): {consultant1_data.get('model_answer')}")
        tqdm.write(f"Consultant 2 ({consultant2_name}): {consultant2_data.get('model_answer')}")
        
        try:
            winner, full_response = call_and_parse_judge(prompt, args.judge_model)
            
            if winner == 'Consultant 1':
                winning_model = consultant1_name
                winning_answer = consultant1_data.get('model_answer')
                winning_correct = consultant1_data.get('model_correct', False)
            elif winner == 'Consultant 2':
                winning_model = consultant2_name
                winning_answer = consultant2_data.get('model_answer')
                winning_correct = consultant2_data.get('model_correct', False)
            else:
                winning_model = 'Neither'
                winning_answer = None
                winning_correct = None
            
            correct_answer = consultant1_data.get('correct_choice')
            model1_correct = model1_data.get('model_correct', False)
            model2_correct = model2_data.get('model_correct', False)
            
            model1_key = f"{get_model_name_key(model1_name)}_correct"
            model2_key = f"{get_model_name_key(model2_name)}_correct"
            
            consultant1_correct = consultant1_data.get('model_correct', False)
            consultant2_correct = consultant2_data.get('model_correct', False)
            
            if winner == 'Neither':
                judge_correct = (not consultant1_correct) and (not consultant2_correct)
            else:
                judge_correct = winning_correct if winning_correct is not None else False
            
            result = {
                'index': index,
                'question': question,
                'correct_choice': correct_answer,
                'judge_model': args.judge_model,
                'consultant1_name': consultant1_name,
                'consultant1_answer': consultant1_data.get('model_answer'),
                'consultant1_correct': consultant1_correct,
                'consultant2_name': consultant2_name,
                'consultant2_answer': consultant2_data.get('model_answer'),
                'consultant2_correct': consultant2_correct,
                'judge_winner': winner,
                'winning_model': winning_model,
                'winning_answer': winning_answer,
                'winning_correct': winning_correct,
                'judge_correct': judge_correct,
                'raw_prompt': prompt,
                'judge_response': full_response,
                model1_key: model1_correct,
                model2_key: model2_correct,
            }
            
            results.append(result)
            
            tqdm.write(f"Judge winner: {winner} ({winning_model})")
            tqdm.write(f"Winning model correct: {winning_correct}")
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
        consultant1_wins = sum(1 for r in results if r['judge_winner'] == 'Consultant 1')
        consultant2_wins = sum(1 for r in results if r['judge_winner'] == 'Consultant 2')
        neither_wins = sum(1 for r in results if r['judge_winner'] == 'Neither')
        correct_decisions = sum(1 for r in results if r['winning_correct'] == True)
        judge_accuracy = sum(1 for r in results if r.get('judge_correct', False) == True)
        
        print(f"\nSUMMARY STATISTICS:")
        print(f"Total cases: {total}")
        print(f"Consultant 1 wins: {consultant1_wins} ({consultant1_wins/total*100:.1f}%)")
        print(f"Consultant 2 wins: {consultant2_wins} ({consultant2_wins/total*100:.1f}%)")
        print(f"Neither wins: {neither_wins} ({neither_wins/total*100:.1f}%)")
        print(f"Judge selected correct answer: {correct_decisions} ({correct_decisions/total*100:.1f}%)")
        print(f"Judge accuracy: {judge_accuracy}/{total} ({judge_accuracy/total*100:.1f}%)")


if __name__ == "__main__":
    main()
