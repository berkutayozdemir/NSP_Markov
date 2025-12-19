from markov_chain import MarkovChain
import sys

try:
    print("Loading model with Order 2...")
    chain = MarkovChain("NUTUK_1.txt", max_order=2)
    print("Model loaded.")
    
    print("Generating text with start 'Millet ve'...")
    text = chain.generate_text("Millet ve", length=10)
    print(f"Generated: {text}")
    
    print("Checking visualization data for 'Millet ve'...")
    transitions = chain.get_top_transitions("Millet ve", top_n=5)
    print(f"Transitions for 'Millet ve': {transitions}")
    
    if text and transitions:
        print("Verification SUCCESS")
    else:
        print("Verification FAILED")

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
