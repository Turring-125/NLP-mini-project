"""
shallow_parser.py - Shallow Parsing & Chunking
Module 2.3 / Module 3: Structures and Parsing
Performs Part-of-Speech tagging and Regular Expression Chunking to extract
Noun Phrases (NP), Verb Phrases (VP), and Prepositional Phrases (PP).
"""

import nltk
from typing import List, Tuple, Union

# Robust Chunk Grammar for agricultural queries
CHUNK_GRAMMAR = r"""
  NP: {<DT|PRP\$|POS>?<JJ.*|CD>*<NN.*>+}   # Noun Phrase: optional determiner, adjectives, nouns
  VP: {<MD>?<VB.*>+(<RB.*>)?}              # Verb Phrase: optional modal, verbs, adverbs
  PP: {<IN>+<NP>}                          # Prepositional Phrase: preposition followed by NP
"""

class ShallowParser:
    """NLTK-based POS tagger and Regexp Chunk Parser."""

    def __init__(self, grammar: str = CHUNK_GRAMMAR):
        self.grammar = grammar
        self.parser = nltk.RegexpParser(grammar)

    def parse(self, sentence: str) -> nltk.Tree:
        """Tokenize, POS-tag, and parse sentence into a chunk tree."""
        if not sentence or not isinstance(sentence, str):
            return nltk.Tree("S", [])
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        return self.parser.parse(tagged)

    def format_bracketed(self, tree: nltk.Tree) -> str:
        """Convert chunk tree into human-readable bracketed format [Label Words]."""
        formatted = []
        for item in tree:
            if isinstance(item, nltk.Tree):
                words = " ".join(leaf[0] for leaf in item.leaves())
                formatted.append(f"[{item.label()} {words}]")
            else:
                formatted.append(item[0])
        return " ".join(formatted)

    def extract_chunks(self, sentence: str) -> List[Tuple[str, str]]:
        """Return list of (chunk_label, chunk_text) tuples."""
        tree = self.parse(sentence)
        chunks = []
        for item in tree:
            if isinstance(item, nltk.Tree):
                words = " ".join(leaf[0] for leaf in item.leaves())
                chunks.append((item.label(), words))
        return chunks

# Default singleton instance
_PARSER = ShallowParser()

def parse_chunks(sentence: str) -> str:
    """Convenience helper returning the human-readable bracketed chunk string."""
    try:
        tree = _PARSER.parse(sentence)
        return _PARSER.format_bracketed(tree)
    except Exception as e:
        return f"[ERROR parsing sentence: {e}]"

def extract_chunk_list(sentence: str) -> List[Tuple[str, str]]:
    """Convenience helper returning structured chunk tuples."""
    try:
        return _PARSER.extract_chunks(sentence)
    except Exception:
        return []

if __name__ == "__main__":
    demo_sentences = [
        "my sugarcane leaves are curling because of water stress",
        "farmer applied urea fertilizer in the morning",
        "red rot affects the standing crop severely"
    ]
    for s in demo_sentences:
        print(f"\nSentence: '{s}'")
        print("Chunks:  ", parse_chunks(s))
        print("List:    ", extract_chunk_list(s))
