PROMPT_SWITCH = {
    "AS": "Given the text: {text}, what are the aspect terms and their sentiments?",
    "ALSC": """Given the target terms and the aspect terms are enclosed by "<" and ">" and the text {text}, what are the opinion terms and their sentiments?""",
    "ASTE": "Given the text {text}, what are the aspect terms, opinion terms and their sentiments?",
    "ASQP": "Given the text {text}, what are the target terms, aspect terms, opinion terms and their sentiments?",
    "FREEFORM": """Given the text {text}, what are the sentiment tuples inside?
    For target term, aspect term and opinion term, please extract the text spans from the text. For target_category,
    please choose from the below list: {target_categories}. For aspect_category, please choose from the below list:
    {aspect_categories}. For sentiment, please choose from the below list: {sentiment_polarities}. Please output sentiment tuples
    in the form of (target term, target_category, aspect term, aspect_category, opinion term, sentiment). Use newline marker \n to separate different tuples.""",
}
ASPECT_CATEGORIES = [
    "Battery",
    "Call Experience",
    "Device Temperature (Thermal)",
    "SystemUI",
    "Performance",
    "Data Connectivity",
    "Authentication(UDFPS, Face authentication)",
    "App Experience",
    "Camera",
    "Stability",
    "Others",
    "Price",
    "Screen",
]

TARGET_CATEORIES = [
    "Pixel 7 Pro",
    "Pixel 7",
    "Pixel 6 Pro",
    "Pixel 6",
    "Pixel Watch",
    "Pixel 5",
    "Others",  # With hint "Not in the above or cannot be determined"
]


SENTIMENT_POLARITIES = ["Positive", "Negative", "Neutral"]
