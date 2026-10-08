"""
wsd.py - Word Sense Disambiguation (WSD)
Module 4.3: Word Sense Disambiguation
Disambiguates polysemous agricultural terms (e.g., 'plant', 'field', 'rot', 'yield', 'spray')
using Part-of-Speech context constraints, WordNet synsets, and Lesk algorithm overlap.
"""

import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from typing import List, Dict, Any

# Target polysemous words common in farm queries and their domain sense mappings
AMBIGUOUS_TARGETS = {"plant", "field", "rot", "yield", "spray"}

def is_verb_context(idx: int, tokens: List[str], pos_tag: str) -> bool:
    """Detect if word functions as a verb or imperative action in context."""
    if pos_tag.startswith("V"):
        return True
    if idx == 0 and len(tokens) > 1:
        # Sentence-initial imperative command (e.g., 'Plant 50 kg...', 'Spray 2 litres...')
        return True
    if idx > 0 and tokens[idx - 1].lower() in ["to", "should", "can", "will", "must", "please", "i", "we"]:
        return True
    return False

def is_noun_context(idx: int, tokens: List[str], pos_tag: str) -> bool:
    """Detect if word functions as a noun (e.g., preceded by determiner or adjective)."""
    if idx > 0 and tokens[idx - 1].lower() in ["the", "a", "an", "this", "that", "my", "our", "its", "each", "every"]:
        return True
    if pos_tag.startswith("N"):
        return True
    return False

def disambiguate_word(word: str, pos_tag: str, sentence: str, tokens: List[str], idx: int = 0) -> Dict[str, Any]:
    """Disambiguate a single target word in its sentence context."""
    w_lower = word.lower()
    
    # Check syntactic context (imperative vs nominal)
    has_verb_ctx = is_verb_context(idx, tokens, pos_tag)
    has_noun_ctx = is_noun_context(idx, tokens, pos_tag) and idx > 0
    
    wn_pos = wn.VERB if has_verb_ctx and not has_noun_ctx else wn.NOUN
    syn = lesk(tokens, w_lower, pos=wn_pos)
    sent_lower = sentence.lower()

    if w_lower == "plant":
        if has_verb_ctx and not has_noun_ctx:
            sense_label = "planting / agricultural action (sowing seeds/saplings into soil)"
            synset_name = syn.name() if syn and syn.pos() == "v" else "plant.v.01"
            definition = "put or set (seeds or seedlings) in the ground to grow"
        else:
            sense_label = "crop / botanical organism (living photosynthetic plant)"
            synset_name = syn.name() if syn and syn.pos() == "n" else "plant.n.02"
            definition = "a living botanical organism lacking the power of locomotion"

    elif w_lower == "field":
        agri_clues = ["crop", "sugarcane", "soil", "acre", "sun", "irrigate", "water", "farm", "paddy", "plot", "nursery"]
        if any(clue in sent_lower for clue in agri_clues):
            sense_label = "farm plot / agricultural land (cultivated parcel of ground)"
            synset_name = "field.n.01"
            definition = "a piece of land cleared of trees and usually enclosed"
        else:
            sense_label = "domain / discipline of study (branch of academic knowledge)"
            synset_name = "field.n.04"
            definition = "a branch of knowledge or area of specialized study"

    elif w_lower == "rot":
        if has_verb_ctx and not has_noun_ctx and "red rot" not in sent_lower:
            sense_label = "decomposition / decay action (break down organically)"
            synset_name = syn.name() if syn and syn.pos() == "v" else "rot.v.01"
            definition = "break down or cause to break down by decay"
        else:
            sense_label = "plant disease / fungal decay symptom (pathological condition)"
            synset_name = syn.name() if syn and syn.pos() == "n" else "rot.n.01"
            definition = "a plant disease that causes decay of vegetable tissues"

    elif w_lower == "yield":
        if has_verb_ctx and not has_noun_ctx:
            sense_label = "produce / bear harvest (give forth agricultural output)"
            synset_name = syn.name() if syn and syn.pos() == "v" else "yield.v.01"
            definition = "be productive or give forth agricultural produce"
        else:
            sense_label = "harvest output / production quantity (total crop volume)"
            synset_name = syn.name() if syn and syn.pos() == "n" else "yield.n.01"
            definition = "an amount produced or harvested"

    elif w_lower == "spray":
        if has_verb_ctx and not has_noun_ctx:
            sense_label = "chemical application action (spritzing liquid onto foliage)"
            synset_name = syn.name() if syn and syn.pos() == "v" else "spray.v.01"
            definition = "apply a liquid in fine dispersion or droplets"
        else:
            sense_label = "liquid chemical mixture / formulation (liquid pesticide)"
            synset_name = syn.name() if syn and syn.pos() == "n" else "spray.n.01"
            definition = "a liquid preparation to be discharged in a fine mist"
            
    else:
        sense_label = syn.definition() if syn else "general vocabulary meaning"
        synset_name = syn.name() if syn else "none"
        definition = syn.definition() if syn else "no definition found"

    return {
        "word": word,
        "pos_tag": pos_tag,
        "sense_label": sense_label,
        "synset": synset_name,
        "definition": definition
    }

def disambiguate_sentence(sentence: str) -> List[Dict[str, Any]]:
    """Scan query for ambiguous agricultural words and resolve their contextual sense."""
    if not sentence or not isinstance(sentence, str):
        return []
    try:
        tokens = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokens)
        results = []
        for idx, (word, tag) in enumerate(pos_tags):
            if word.lower() in AMBIGUOUS_TARGETS:
                res = disambiguate_word(word, tag, sentence, tokens, idx=idx)
                results.append(res)
        return results
    except Exception:
        return []

if __name__ == "__main__":
    demo_pairs = [
        ("Plant the sugarcane seedlings tomorrow.", "The plant is affected by red rot."),
        ("Working in the sugarcane field.", "He is an expert in the field of agronomy."),
        ("The stalks have severe red rot.", "Cane will rot without proper drainage.")
    ]
    for s1, s2 in demo_pairs:
        print(f"\n[A] '{s1}' -> {disambiguate_sentence(s1)}")
        print(f"[B] '{s2}' -> {disambiguate_sentence(s2)}")
