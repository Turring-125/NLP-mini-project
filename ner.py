"""
ner.py - Agricultural Named Entity Recognition (NER)
Module 2.3: Named Entities
Extracts crops, diseases, pests, fertilizers, chemicals, quantities, units,
locations, time expressions, irrigation terms, and environmental conditions.
"""

import re
from typing import List, Dict, Any

# Domain gazetteers for agricultural entity extraction
AGRICULTURAL_GAZETTEERS = {
    "CROP": [
        "sugarcane", "cane", "paddy", "rice", "wheat", "cotton", "maize", "corn",
        "coconut", "banana", "tomato", "potato", "onion", "soybean", "mustard",
        "groundnut", "chilli", "turmeric", "ginger", "gram", "chickpea", "sorghum"
    ],
    "DISEASE": [
        "red rot", "smut", "rust", "blast", "blight", "wilt", "leaf spot", "leaf curl",
        "powdery mildew", "downy mildew", "mosaic", "canker", "damping off", "grassy shoot"
    ],
    "PEST": [
        "stem borer", "top borer", "early shoot borer", "internode borer", "pyrilla",
        "whitefly", "aphid", "thrips", "armyworm", "bollworm", "mealybug", "caterpillar",
        "scale insect", "termite", "root grub", "hopper"
    ],
    "FERTILIZER": [
        "urea", "dap", "diammonium phosphate", "mop", "muriate of potash", "ssp",
        "single super phosphate", "npk", "zinc sulphate", "borax", "compost", "manure",
        "farmyard manure", "vermicompost", "biofertilizer", "potash", "micronutrient"
    ],
    "CHEMICAL": [
        "chlorpyrifos", "carbendazim", "bavistin", "malathion", "atrazine", "glyphosate",
        "mancozeb", "imidacloprid", "monocrotophos", "copper oxychloride", "endosulfan",
        "furadon", "carbofuran", "cartap hydrochloride", "chlorantraniliprole", "weedicide", "pesticide"
    ],
    "IRRIGATION": [
        "drip irrigation", "flood irrigation", "furrow irrigation", "sprinkler irrigation",
        "drip lateral", "drip system", "fertigation", "subsurface drip", "tube well",
        "canal water", "watering interval", "drip"
    ]
}

# Regex patterns for numerical, condition, temporal, and spatial entities
PATTERNS = {
    "CONDITION": r"\b(?:soil\s+is\s+(?:very\s+)?dry|moisture\s+stress|water\s+stress|yellowing\s+leaves|cracked\s+soil|curling\s+leaves|waterlogging|dry\s+soil|wilting)\b",
    "QTY_UNIT": r"\b(\d+(?:\.\d+)?|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b)\s*(acres?|hectares?|kg|kilograms?|grams?|g|litres?|liters?|l|ml|bags?|quintals?|tons?)\b",
    "TIME": r"\b(yesterday|today|tomorrow|morning|evening|last\s+week|next\s+week|summer|winter|monsoon|kharif|rabi|june|july|august|september|october|november|december|january|february|march|april|may)\b",
    "LOCATION": r"\b(maharashtra|uttar pradesh|tamil nadu|punjab|haryana|karnataka|andhra pradesh|bihar|gujarat|nursery|field|plot|farm)\b"
}

def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Extract named entities from an agricultural query string."""
    if not text or not isinstance(text, str):
        return []
    
    text_lower = text.lower()
    entities: List[Dict[str, Any]] = []

    # 1. Condition expressions (e.g., 'soil is very dry', 'moisture stress')
    for match in re.finditer(PATTERNS["CONDITION"], text_lower):
        entities.append({"text": match.group(), "label": "CONDITION", "start": match.start(), "end": match.end()})

    # 2. Quantities and units (e.g., '2 acres', '50 kg')
    for match in re.finditer(PATTERNS["QTY_UNIT"], text_lower):
        entities.append({"text": match.group(1), "label": "QUANTITY", "start": match.start(1), "end": match.end(1)})
        entities.append({"text": match.group(2), "label": "UNIT", "start": match.start(2), "end": match.end(2)})

    # 3. Temporal expressions (e.g., 'yesterday', 'summer', 'june')
    for match in re.finditer(PATTERNS["TIME"], text_lower):
        entities.append({"text": match.group(1), "label": "TIME", "start": match.start(1), "end": match.end(1)})

    # 4. Location expressions (e.g., 'maharashtra', 'field')
    for match in re.finditer(PATTERNS["LOCATION"], text_lower):
        entities.append({"text": match.group(1), "label": "LOCATION", "start": match.start(1), "end": match.end(1)})

    # 5. Domain Gazetteers (longest match prioritized to prevent substring collisions)
    for label, terms in AGRICULTURAL_GAZETTEERS.items():
        terms_sorted = sorted(terms, key=len, reverse=True)
        for term in terms_sorted:
            pattern = r"\b" + re.escape(term) + r"\b"
            for match in re.finditer(pattern, text_lower):
                # Avoid overlapping with already extracted CONDITION spans
                overlap = any(e["start"] <= match.start() and match.end() <= e["end"] for e in entities if e["label"] == "CONDITION")
                if not overlap:
                    entities.append({"text": match.group(), "label": label, "start": match.start(), "end": match.end()})

    # Sort entities chronologically by span start index
    entities = sorted(entities, key=lambda x: x["start"])
    return entities

def format_entities_grouped(entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Group extracted entities by label for clean presentation in UI and pipelines."""
    grouped: Dict[str, List[str]] = {}
    for e in entities:
        label = e["label"]
        if label not in grouped:
            grouped[label] = []
        if e["text"] not in grouped[label]:
            grouped[label].append(e["text"])
    return grouped

if __name__ == "__main__":
    test_queries = [
        "My sugarcane crop has red rot and I applied urea yesterday.",
        "I have 2 acres of sugarcane and the soil is very dry.",
        "How to control stem borer in cotton using chlorpyrifos in summer?"
    ]
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        ents = extract_entities(q)
        print("Entities:", [(e["label"], e["text"]) for e in ents])
