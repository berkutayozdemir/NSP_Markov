from markov_chain import MarkovChain
import sys

try:
    print("Loading model...")
    chain = MarkovChain("NUTUK_1.txt")
    print("Model loaded.")
    
    print("Generating text...")
    text = chain.generate_text("Millet", length=10)
    print(f"Generated: {text}")
    
    print("Checking visualization data...")
    transitions = chain.get_top_transitions("Millet", top_n=5)
    print(f"Transitions for 'Millet': {transitions}")
    
    if text and transitions:
        print("Verification SUCCESS")
    else:
        print("Verification FAILED")

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
