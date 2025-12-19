import numpy as np
import re
import os

class MarkovChain:
    def __init__(self, file_path="NUTUK_1.txt"):
        self.file_path = file_path
        self.words = []
        self.vocab = []
        self.word_to_index = {}
        self.index_to_word = {}
        self.transition_counts = {}
        self.vocab_size = 0
        
        self._load_and_process_data()
        self._build_model()

    def _load_and_process_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        # Read the file and use content starting from line 283 (logic from original notebook)
        with open(self.file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        start_line_index = 282
        nutuk = ""
        if len(lines) > start_line_index:
            nutuk = "".join(lines[start_line_index:])
        else:
            # Fallback if file is short
            nutuk = "".join(lines)

        # Cleaning
        nutuk_cleaned = nutuk.lower()
        nutuk_cleaned = re.sub(r'[^\w\s]', '', nutuk_cleaned)
        nutuk_cleaned = nutuk_cleaned.replace('\n', ' ')
        self.words = nutuk_cleaned.split()

    def _build_model(self):
        self.vocab = sorted(list(set(self.words)))
        self.vocab_size = len(self.vocab)
        self.word_to_index = {word: i for i, word in enumerate(self.vocab)}
        self.index_to_word = {i: word for i, word in enumerate(self.vocab)}

        for i in range(len(self.words) - 1):
            current_word = self.words[i]
            next_word = self.words[i+1]
            
            if current_word not in self.transition_counts:
                self.transition_counts[current_word] = {}
                
            if next_word not in self.transition_counts[current_word]:
                self.transition_counts[current_word][next_word] = 0
                
            self.transition_counts[current_word][next_word] += 1

    def get_next_word(self, current_word, temperature=1.0, alpha=1.0):
        # Initialize probabilities with smoothing (alpha)
        probs = np.ones(self.vocab_size) * alpha
        
        # Add observed counts
        if current_word in self.transition_counts:
            for next_word, count in self.transition_counts[current_word].items():
                next_word_idx = self.word_to_index[next_word]
                probs[next_word_idx] += count
                
        # Normalize
        probs = probs / np.sum(probs)
        
        # Apply Temperature
        if temperature != 1.0:
            probs = np.power(probs, 1.0 / temperature)
            probs = probs / np.sum(probs)
            
        next_word_idx = np.random.choice(range(self.vocab_size), p=probs)
        return self.index_to_word[next_word_idx]

    def generate_text(self, start_word, length=20, temperature=1.0, alpha=1.0):
        current = start_word.lower()
        if current not in self.word_to_index:
            # If word not in vocab, pick a random one or handle gracefully
            # For this app, simply picking a random start might be better, or just adding it to result and hoping next step works (it won't if not in counts).
            # Let's fallback to random word if unknown
            if self.vocab:
                current = np.random.choice(self.vocab)
        
        result = [current]
        
        for _ in range(length):
            next_word = self.get_next_word(current, temperature, alpha)
            result.append(next_word)
            current = next_word
            
        return " ".join(result)

    def get_top_transitions(self, current_word, top_n=10, alpha=1.0):
        current_word = current_word.lower()
        
        # Similar logic to get_next_word but returns distribution
        probs = np.ones(self.vocab_size) * alpha
        
        if current_word in self.transition_counts:
            for next_word, count in self.transition_counts[current_word].items():
                next_word_idx = self.word_to_index[next_word]
                probs[next_word_idx] += count
        
        probs = probs / np.sum(probs)
        
        # Get top N indices
        top_indices = np.argsort(probs)[-top_n:][::-1]
        
        transitions = {}
        for idx in top_indices:
            word = self.index_to_word[idx]
            prob = probs[idx]
            transitions[word] = prob
            
        return transitions
