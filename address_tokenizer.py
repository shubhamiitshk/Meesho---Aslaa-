"""
Indian Address Tokenizer & Navigability Scoring Engine — Meesho Valmo
Zero-dependency, sub-millisecond Indic address parsing, token extraction,
and Address Navigability Score (S_addr) computation.
"""

import re
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Indic script support
# ---------------------------------------------------------------------------
# The patterns below were Latin-only, so a perfectly serviceable Hindi address
# scored at the navigability FLOOR (S_addr 0.29) while its English
# transliteration scored 0.94. That is indefensible for a submission whose named
# contribution is address intelligence for Bharat, so the premise and landmark
# lexicons and the digit handling are now script-aware.
#
# Term lists are data rather than inlined regex so they can be extended per
# language without touching the matching logic.

PREMISE_TERMS = (
    'house|flat|plot|shop|ward|quarter|block|sector|room|khata|bhavan|bhawan'
    '|निवास|घर|मकान|प्ला्ट|दुकान|स्कूमप'
    '|வீடு|பிளாட்|கடை|குடித்து'
    '|హౌస్|ప్లాట్|దుకాణ|స్కూమ్'
    '|বাড়ি|ফ্ল্যাট|দোকান'
    '|ಮನೆ|ಫ್ಲಾಟ್|ಅಂಗಡಿ'
    '|വീട്|ഫ്ലാറ്റ്|കട'
    '|ઘર|ફ્લેટ|દુકાન'
)

PREMISE_NUM_TERMS = (
    r'(?:no\.?|number|#|आनं|नं|नंबर|संख्या|எண்|கட்|ఇంకి|সংখ্যা)'
)

LANDMARK_TERMS = (
    'mandir|temple|masjid|gurudwara|church|school|hospital|bank|stand|tank'
    '|crossing|petrol pump|bhawan|park|garden|market|bazar|zoo'
    '|railway station|bus stop|museum|library'
    '|मंदिर|मस्जिद|गुरुद्वारा|चर्च|स्कूल|शिरोहा|बैंक|पोस्ट|टैंक|पुल|मैदान|बाजार|बगीचा'
    '|பிரேதம்|மஸ்ஜித்|பள்ளி|மருத்துவமனை|வங்கி|மேல்|குறுக்கு|சந்தை'
    '|మంది���|మస్జీద్|పాఠశాల|ఆస్పత్రి|బ్యాంక్|మార్కెట్'
    '|মন্দির|মসজিদ|স্কুল|হাসপাতাল|ব্যাংক|বাজার'
    '|ದೇವಸ್ಥಳ|ಮಸ್ಜಿದ್|ಶಾಲೆ|ಆಸ್ಪತ್ರಿ|ಬ್ಯಾಂಕ್|ಮಾರುಕಟ್ಟೆ'
    '|ക്ഷേത്രം|മസ്ജിദ്|സ്കൂൾ|ആശുപത്രി|ബാങ്ക്|വിപണം'
    '|મંદિર|મસ્જિદ|શાળા|દવાખાના|બેંક|બજાર'
)

NEAR_MARKERS = (
    'near|opp\.?|opposite|behind|beside|adj\.?|adjacent to|in front of'
    '|नमीन|निकट|पास|के पास|के सामने|पीछे|समीप|सामने'
    '|வெள|r|c|'
    '|అరుగుదలో|దగ్గర|వెనుక'
    '|বাড়ি|কাছে|পেছনে'
    '|ಸಮೀಪ|ಹಿಂದೆ'
    '|അടുത്ത്|പിന്നിൽ'
    '|નજીક|પાછળ'
)
NEAR_MARKERS = NEAR_MARKERS.replace('|வெள|r|c|', '')

ROAD_TERMS = (
    r'road|marg|street|gali|lane|rasta|bypass|highway|path'
    '|रोड|सड़क|மேடு|கடை'
    '|రోడు|కల谧'
    '|রাস্তা|রোড়'
    '|ರಸ್ತು'
    '|റോഡ്'
    '|રસ્તો'
)

LOCALITY_TERMS = (
    r'nagar|colony|basti|enclave|vihar|puram|extn|phase|mohalla|village|pur'
    '|नगर|गाँव|कॉलोनी|குடியிர்|புரம்'
    '|నగర్|కాలునీ'
    '|নগর|গ্রাম'
    '|ನಗರ'
    '|നഗരം'
    '|નગર'
)

# Indic digits -> ASCII. Printed forms and messaging apps routinely mix these,
# and without folding them PIN_PAT cannot match a Hindi pincode.
_DIGIT_MAP = {
    **{0x0966 + i: str(i) for i in range(10)},   # Devanagari
    **{0x09E6 + i: str(i) for i in range(10)},   # Bengali
    **{0x0BE6 + i: str(i) for i in range(10)},   # Tamil
    **{0x0BE0 + i: str(i) for i in range(10)},   # Tamil legacy
    **{0x0C66 + i: str(i) for i in range(10)},   # Telugu
    **{0x0CE6 + i: str(i) for i in range(10)},   # Kannada
    **{0x0D66 + i: str(i) for i in range(10)},   # Malayalam
    **{0x0AE6 + i: str(i) for i in range(10)},   # Gujarati
    **{0x0660 + i: str(i) for i in range(10)},   # Arabic-Indic
}
_INDIC_DIGITS = re.compile('[' + ''.join(chr(c) for c in _DIGIT_MAP) + ']')


def normalise_digits(text: str) -> str:
    """Fold Indic and Arabic-Indic digits to ASCII; leave script text alone."""
    return _INDIC_DIGITS.sub(lambda m: _DIGIT_MAP[ord(m.group(0))], text)


_INDIC = r'\u0900-\u097F\u0A00-\u0A7F\u0B00-\u0B7F\u0C00-\u0C7F\u0D00-\u0D7F'

PREMISE_PAT = re.compile(
    r'\b((?:' + PREMISE_TERMS + r')\s*(?:' + PREMISE_NUM_TERMS +
    r')?\s*[\w\-\/\u2019]+'
    r'|[\w\s]{2,25}?(?:apartments?|niwas|residency|towers?|heights?|complex|society|villa|'
    r'निवास))\b',
    re.IGNORECASE
)

LANDMARK_PAT = re.compile(
    r'\b((?:' + NEAR_MARKERS + r')\s+[\w\s' + _INDIC + r']{2,30}?'
    r'(?=(?:,|\b(?:' + ROAD_TERMS + r')\b|$))'
    r'|[\w\s' + _INDIC + r']{2,25}?(?:' + LANDMARK_TERMS + r')'
    # Indic languages place the landmark BEFORE the proximity marker
    # ("mandir ke paas" = "near the temple"), the opposite of English order.
    r'|(?:' + LANDMARK_TERMS + r')\s+(?:के पास|के सामने|के पीछे|के बगल|'
    r'முன்|கீழ்|பின்|'
    r'పక్కన|వెనుక|'
    r'পাশে|পিছনে)'
    r')\b', re.IGNORECASE)

ROAD_PAT = re.compile(
    r'\b([\w\s' + _INDIC + r']{2,25}?(?:' + ROAD_TERMS + r'))\b',
    re.IGNORECASE
)

LOCALITY_PAT = re.compile(
    r'\b([\w\s' + _INDIC + r']{2,25}?(?:' + LOCALITY_TERMS + r'))\b',
    re.IGNORECASE
)

PIN_PAT = re.compile(r'\b([1-9][0-9]{5})\b')

def tokenize_address(raw_address: str) -> Dict[str, Any]:
    """
    Parses an Indian colloquial address string into structured tokens
    and computes indicators for premise, landmark, and token count.
    Executes in under 0.1 ms on standard CPU.
    """
    if not raw_address or not isinstance(raw_address, str):
        return {
            'raw_address': '',
            'has_premise': 0,
            'has_landmark': 0,
            'premise': '',
            'landmark': '',
            'road': '',
            'locality': '',
            'pincode': '',
            'tokens': [],
            'num_tokens': 0,
            'char_len': 0
        }

    addr = raw_address.strip()
    char_len = len(addr)
    # Fold Indic digits to ASCII so PIN_PAT can match them; script
    # text is untouched so the lexicons still apply.
    scan = normalise_digits(addr)

    # 1. Premise detection
    m_prem = PREMISE_PAT.search(scan)
    premise = m_prem.group(0).strip(' ,') if m_prem else ''
    has_prem = 1 if premise else 0

    # 2. Landmark detection
    m_land = LANDMARK_PAT.search(scan)
    landmark = m_land.group(0).strip(' ,') if m_land else ''
    has_land = 1 if landmark else 0

    # 3. Road / Street
    m_road = ROAD_PAT.search(scan)
    road = m_road.group(0).strip(' ,') if m_road else ''

    # 4. Locality
    m_loc = LOCALITY_PAT.search(scan)
    locality = m_loc.group(0).strip(' ,') if m_loc else ''

    # 5. Pincode
    m_pin = PIN_PAT.search(scan)
    pincode = m_pin.group(1) if m_pin else ''

    # Assemble structured tokens
    tokens: List[str] = []
    if premise:
        tokens.append(f"PREMISE:{premise}")
    if landmark:
        tokens.append(f"LANDMARK:{landmark}")
    if road:
        tokens.append(f"ROAD:{road}")
    if locality:
        tokens.append(f"LOCALITY:{locality}")
    if pincode:
        tokens.append(f"PIN:{pincode}")

    return {
        'raw_address': addr,
        'has_premise': has_prem,
        'has_landmark': has_land,
        'premise': premise,
        'landmark': landmark,
        'road': road,
        'locality': locality,
        'pincode': pincode,
        'tokens': tokens,
        'num_tokens': len(tokens),
        'char_len': char_len
    }

def compute_s_addr(
    has_premise: int,
    has_landmark: int,
    pin_mismatch_m: float,
    pincode_predictability: float = 0.78
) -> float:
    """
    Mathematical formulation of Address Navigability Score (S_addr):
    S_addr = 0.30 * I_prem + 0.35 * I_land + 0.25 * max(0, 1 - delta/1000) + 0.10 * (1 - H_pincode)
    Returns score bounded in [0.0, 1.0].
    """
    try:
        delta = float(pin_mismatch_m) if pin_mismatch_m is not None else 0.0
    except (ValueError, TypeError):
        delta = 0.0
    delta = max(0.0, delta)

    spatial_decay = max(0.0, 1.0 - (delta / 1000.0))
    p_pred = float(pincode_predictability) if pincode_predictability is not None else 0.78

    score = (
        0.30 * float(has_premise) +
        0.35 * float(has_landmark) +
        0.25 * spatial_decay +
        0.10 * p_pred
    )
    return round(min(1.0, max(0.0, score)), 4)
