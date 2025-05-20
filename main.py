import os
import json
import patronus
from patronus.evals import RemoteEvaluator
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
import argparse
import sys
from datetime import datetime

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {file_path}: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading {file_path}: {e}")
        sys.exit(1)

def find_message_pairs(messages: List[Dict[str, Any]], intent_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Find message pairs (candidate question and bot response).
    
    Args:
        messages: List of message objects from AssistMessage.json
        intent_filter: Optional intent name to filter by
        
    Returns:
        List of message pairs
    """
    # Group messages by thread ID
    thread_messages = {}
    for msg in messages:
        # Skip if intent doesn't match filter (when provided)
        if intent_filter and msg.get('intent') != intent_filter:
            continue
            
        thread_id = msg.get('botThreadId', {}).get('$oid')
        if thread_id:
            if thread_id not in thread_messages:
                thread_messages[thread_id] = []
            thread_messages[thread_id].append(msg)
    
    # Find pairs in each thread
    pairs = []
    for thread_id, thread_msgs in thread_messages.items():
        # Sort by creation time
        thread_msgs.sort(key=lambda x: x.get('createdAt', {}).get('$date', ''))
        
        # Group candidate messages with their bot responses
        for i, msg in enumerate(thread_msgs):
            if msg.get('sender') == 'candidate' and i + 1 < len(thread_msgs):
                next_msg = thread_msgs[i + 1]
                if next_msg.get('sender') == 'bot':
                    pairs.append({
                        'candidate_message': msg,
                        'bot_response': next_msg,
                        'thread_id': thread_id
                    })
    
    return pairs

def get_intent_context(intent_name: str, intents: List[Dict[str, Any]]) -> List[str]:
    """Get context information from the intent definition."""
    context = []
    for intent in intents:
        if intent.get('slug') == intent_name:
            # Add intent description if available
            if 'description' in intent:
                context.append(f"Intent Description: {intent['description']}")
            
            # Add required contexts
            if 'requiredContexts' in intent:
                required = [k for k, v in intent['requiredContexts'].items() if v]
                if required:
                    context.append(f"Required Contexts: {', '.join(required)}")
            
            # Add other relevant info
            if 'contextSchema' in intent:
                context.append(f"Context Schema Available: Yes")
            
            break
    
    return context

def run_evaluation(evaluator, pair: Dict[str, Any], intents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run evaluation on a message pair."""
    candidate_msg = pair['candidate_message']
    bot_msg = pair['bot_response']
    intent_name = candidate_msg.get('intent', '')
    
    # Build context
    context = [
        f"Intent: {intent_name}",
    ]
    
    # Add intent-specific context if available
    intent_context = get_intent_context(intent_name, intents)
    if intent_context:
        context.extend(intent_context)
    
    # Add citations if available
    if 'citations' in bot_msg and bot_msg['citations']:
        context.append(f"Citations: {', '.join(bot_msg['citations'])}")
    
    # Truncate context items if too long
    max_context_length = 500  # Characters per context item
    context = [
        item[:max_context_length] + '...' if len(item) > max_context_length else item
        for item in context
    ]
    
    result = evaluator.evaluate(
        task_input=candidate_msg.get('content', ''),
        task_context=context,
        task_output=bot_msg.get('content', ''),
        gold_answer="" # No gold answer for this test
    )
    
    return {
        'candidate_message': candidate_msg.get('content', ''),
        'bot_response': bot_msg.get('content', ''),
        'intent': intent_name,
        'evaluation_result': result,
        'thread_id': pair.get('thread_id', '')
    }

def get_available_intents(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract all unique intents from the message data."""
    intents = set()
    for msg in messages:
        if 'intent' in msg and msg['intent']:
            intents.add(msg['intent'])
    return sorted(list(intents))

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate Parallel message pairs using Patronus')
    parser.add_argument('--limit', type=int, default=5, help='Number of message pairs to evaluate')
    parser.add_argument('--save', action='store_true', help='Save results to a JSON file')
    parser.add_argument('--intent', type=str, help='Filter messages by specific intent')
    parser.add_argument('--list-intents', action='store_true', help='List all available intents and exit')
    parser.add_argument('--shuffle', action='store_true', help='Randomly shuffle pairs before selecting')
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if API key is available
    api_key = os.environ.get("PATRONUS_API_KEY")
    if not api_key:
        print("Error: PATRONUS_API_KEY not found in environment variables or .env file")
        print("Please set this key in your .env file")
        sys.exit(1)
    
    # File paths - using os.path.join for cross-platform compatibility
    base_dir = os.path.dirname(__file__)
    assist_messages_path = os.path.join(base_dir, 'Parallel-Prod.AssistMessage.json')
    bot_intent_path = os.path.join(base_dir, 'Parallel-Prod.BotIntent.json')
    
    # Load message data
    print("Loading message data...")
    messages = load_json_file(assist_messages_path)
    print(f"Loaded {len(messages)} messages from AssistMessage.json")
    
    # List intents if requested
    if args.list_intents:
        intents = get_available_intents(messages)
        print("\nAvailable intents:")
        for i, intent in enumerate(intents, 1):
            print(f"{i}. {intent}")
        sys.exit(0)
    
    # Load intent data
    print("Loading intent data...")
    intents = load_json_file(bot_intent_path)
    print(f"Loaded {len(intents)} intents from BotIntent.json")
    
    # Initialize patronus with API key from environment
    print("Initializing Patronus...")
    patronus.init(api_key=api_key)
    
    # Create evaluator
    patronus_evaluator = RemoteEvaluator("lynx", "patronus:hallucination")
    
    # Find message pairs (candidate questions and bot responses)
    print("Finding message pairs...")
    pairs = find_message_pairs(messages, args.intent)
    if not pairs:
        print(f"No message pairs found{f' for intent: {args.intent}' if args.intent else ''}.")
        sys.exit(1)
    print(f"Found {len(pairs)} message pairs for evaluation")
    
    # Shuffle pairs if requested
    if args.shuffle:
        import random
        random.shuffle(pairs)
        print("Shuffled message pairs")
    
    # For testing, limit to specified number of pairs
    test_limit = min(args.limit, len(pairs))
    test_pairs = pairs[:test_limit]
    
    # Run evaluation on each pair
    print(f"\nRunning evaluation on {test_limit} message pairs...")
    results = []
    for i, pair in enumerate(test_pairs):
        print(f"\nEvaluating pair {i+1}/{len(test_pairs)}:")
        candidate_content = pair['candidate_message'].get('content', '')
        bot_content = pair['bot_response'].get('content', '')
        intent = pair['candidate_message'].get('intent', 'unknown')
        
        # Truncate long messages for display
        max_display_length = 100
        candidate_display = (candidate_content[:max_display_length] + '...' 
                            if len(candidate_content) > max_display_length 
                            else candidate_content)
        bot_display = (bot_content[:max_display_length] + '...' 
                      if len(bot_content) > max_display_length 
                      else bot_content)
        
        print(f"Candidate ({intent}): {candidate_display}")
        print(f"Bot Response: {bot_display}")
        
        try:
            result = run_evaluation(patronus_evaluator, pair, intents)
            results.append(result)
            print(f"Evaluation result: {result['evaluation_result']}")
        except Exception as e:
            print(f"Error evaluating pair: {e}")
            continue
    
    # Summary
    print(f"\nEvaluation complete for {len(results)} message pairs")
    
    # Save results if requested
    if args.save and results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intent_suffix = f"_{args.intent}" if args.intent else ""
        output_path = os.path.join(base_dir, f'patronus_results{intent_suffix}_{timestamp}.json')
        with open(output_path, 'w') as f:
            json.dump({
                "metadata": {
                    "date": datetime.now().isoformat(),
                    "intent_filter": args.intent,
                    "total_pairs_found": len(pairs),
                    "pairs_evaluated": len(results)
                },
                "results": results
            }, f, indent=2)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()