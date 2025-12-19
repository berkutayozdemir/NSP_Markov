import numpy as np
import re
import os

class MarkovChain:
    def __init__(self, file_path="NUTUK_1.txt", max_order=3):
        self.file_path = file_path
        self.max_order = max_order
        self.words = []
        self.vocab = []
        self.word_to_index = {}
        self.index_to_word = {}
        # models[n] stores transition counts for order n
        # Key: tuple of n words, Value: dict of next_word: count
        self.models = {}
        self.vocab_size = 0
        
        self._load_and_process_data()
        self._build_model()

    def _load_and_process_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        start_line_index = 282
        nutuk = ""
        if len(lines) > start_line_index:
            nutuk = "".join(lines[start_line_index:])
        else:
            nutuk = "".join(lines)

        nutuk_cleaned = nutuk.lower()
        # Add space around punctuation to preserve them as tokens
        nutuk_cleaned = re.sub(r'([.,!?;])', r' \1 ', nutuk_cleaned)
        # Remove anything that is not a word character, whitespace, or punctuation
        nutuk_cleaned = re.sub(r'[^\w\s.,!?;]', '', nutuk_cleaned)
        nutuk_cleaned = nutuk_cleaned.replace('\n', ' ')
        self.words = nutuk_cleaned.split()

    def _build_model(self):
        self.vocab = sorted(list(set(self.words)))
        self.vocab_size = len(self.vocab)
        self.word_to_index = {word: i for i, word in enumerate(self.vocab)}
        self.index_to_word = {i: word for i, word in enumerate(self.vocab)}

        # Build models for orders 1 to max_order
        for n in range(1, self.max_order + 1):
            self.models[n] = {}
            for i in range(len(self.words) - n):
                state = tuple(self.words[i:i+n])
                next_word = self.words[i+n]
                
                if state not in self.models[n]:
                    self.models[n][state] = {}
                
                if next_word not in self.models[n][state]:
                    self.models[n][state][next_word] = 0
                    
                self.models[n][state][next_word] += 1

    def _get_transition_probs(self, history, alpha=0.001):
        # Backoff strategy: Try largest n that matches history suffix
        found_state = None
        found_transitions = None
        
        # Try orders from max_order down to 1
        # history needs at least n words to use order n
        for n in range(min(len(history), self.max_order), 0, -1):
            state = tuple(history[-n:])
            if state in self.models[n]:
                found_state = state
                found_transitions = self.models[n][state]
                break
        
        # Initialize probabilities efficiently
        # Instead of np.ones * alpha which creates a full array, we only need non-zero values if alpha is small
        # But we still need a base probability array to sample from. 
        # Optimized approach:
        
        if found_transitions is None:
             # Uniform distribution if absolutely no context found
             probs = np.ones(self.vocab_size) / self.vocab_size
             return probs, None

        # Calculate probs with smoothing
        # Initialize with alpha directly
        probs = np.full(self.vocab_size, alpha)
        
        for next_word, count in found_transitions.items():
            if next_word in self.word_to_index:
                next_word_idx = self.word_to_index[next_word]
                probs[next_word_idx] += count
        
        return probs / np.sum(probs), found_transitions

    def get_next_word(self, history, temperature=1.0, alpha=0.001):
        probs, _ = self._get_transition_probs(history, alpha)
        
        # Handle very low temperature (deterministic / argmax) to avoid numerical instability
        if temperature < 0.05:
            next_word_idx = np.argmax(probs)
            return self.index_to_word[next_word_idx]

        if temperature != 1.0:
            probs = np.power(probs, 1.0 / temperature)
            probs = probs / np.sum(probs)
            
        next_word_idx = np.random.choice(range(self.vocab_size), p=probs)
        return self.index_to_word[next_word_idx]

    def generate_text(self, start_text, length=20, temperature=1.0, alpha=0.001):
        # Process start text
        current_history = start_text.lower().split()
        
        result = list(current_history)
        
        for _ in range(length):
            next_word = self.get_next_word(current_history, temperature, alpha)
            result.append(next_word)
            current_history.append(next_word)
            
        return " ".join(result)

    def get_top_transitions(self, history, top_n=10, alpha=0.001):
        if isinstance(history, str):
            history = history.lower().split()
            
        probs, _ = self._get_transition_probs(history, alpha)
        
        top_indices = np.argsort(probs)[-top_n:][::-1]
        
        transitions = {}
        for idx in top_indices:
            word = self.index_to_word[idx]
            prob = probs[idx]
            transitions[word] = prob
            
        return transitions
