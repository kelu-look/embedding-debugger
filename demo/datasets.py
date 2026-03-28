"""
Built-in demo datasets for Embedding Debugger.

Each dataset is a function returning (texts, metadata_dict).
No internet download needed — all data is embedded in this file.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


# ------------------------------------------------------------------
# Type alias
# ------------------------------------------------------------------
Dataset = Tuple[List[str], Dict]


# ------------------------------------------------------------------
# 1. FAQ Q&A  — order-sensitivity showcase
# ------------------------------------------------------------------

FAQ_QUESTIONS = [
    "How do I reset my password?",
    "What are your business hours?",
    "How can I contact customer support?",
    "Where is my order?",
    "Can I return a product after 30 days?",
    "How do I cancel my subscription?",
    "What payment methods do you accept?",
    "Is my personal data stored securely?",
    "How do I update my billing address?",
    "What is your refund policy?",
    "How do I download my invoice?",
    "Can I transfer my account to someone else?",
    "How do I change my email address?",
    "Why was my card declined?",
    "Do you offer student discounts?",
    "How long does shipping take?",
    "What happens if my item arrives damaged?",
    "Do you ship internationally?",
    "How do I track my shipment?",
    "Can I change my order after placing it?",
]

FAQ_ANSWERS = [
    "To reset your password, click 'Forgot Password' on the login page.",
    "Our business hours are Monday to Friday, 9 AM to 6 PM EST.",
    "You can reach support via email at support@example.com or call 1-800-555-0100.",
    "Log in to your account and visit 'My Orders' to track your order status.",
    "We only accept returns within 30 days of purchase with original packaging.",
    "To cancel your subscription, go to Account Settings and click 'Cancel Plan'.",
    "We accept Visa, Mastercard, PayPal, and bank transfers.",
    "All data is encrypted using AES-256 and stored on certified secure servers.",
    "Update billing info under Account > Billing > Edit Address.",
    "Refunds are processed within 5-7 business days to your original payment method.",
    "Download invoices from Account > Billing > Invoice History.",
    "Account transfers are not permitted under our Terms of Service.",
    "Change your email under Account > Profile > Contact Information.",
    "Cards may be declined due to insufficient funds, wrong details, or bank restrictions.",
    "We offer a 15% student discount with a valid .edu email address.",
    "Standard shipping takes 3-5 business days; express takes 1-2 days.",
    "Contact support immediately with photos of the damage for a replacement.",
    "We ship to over 50 countries. Shipping rates and times vary by destination.",
    "Use the tracking number in your confirmation email on our tracking page.",
    "Orders can be modified within 1 hour of placement by contacting support.",
]


def load_faq() -> Dataset:
    texts = FAQ_QUESTIONS + FAQ_ANSWERS
    meta = {
        "questions": FAQ_QUESTIONS,
        "answers": FAQ_ANSWERS,
        "n_questions": len(FAQ_QUESTIONS),
        "pairs": list(zip(FAQ_QUESTIONS, FAQ_ANSWERS)),
        "description": "FAQ Q&A pairs — tests semantic matching of questions to answers",
    }
    return texts, meta


# ------------------------------------------------------------------
# 2. Order-blind showcase sentences
#    These sentence pairs have different meaning but similar vocabulary,
#    or same meaning but different word order.
# ------------------------------------------------------------------

ORDER_BLIND_PAIRS = [
    # (original, order-perturbed)
    (
        "The dog bit the man.",
        "The man bit the dog.",
    ),
    (
        "Alice loves Bob.",
        "Bob loves Alice.",
    ),
    (
        "The company acquired the startup.",
        "The startup acquired the company.",
    ),
    (
        "Python is faster than Java for data science.",
        "Java is faster than Python for data science.",
    ),
    (
        "Inflation causes interest rates to rise.",
        "Interest rates cause inflation to rise.",
    ),
    (
        "The patient sued the doctor.",
        "The doctor sued the patient.",
    ),
    (
        "Revenue exceeded expenses last quarter.",
        "Expenses exceeded revenue last quarter.",
    ),
    (
        "Machine learning models learn from data.",
        "Data learns from machine learning models.",
    ),
    (
        "The vaccine prevents the disease.",
        "The disease prevents the vaccine.",
    ),
    (
        "Cats eat mice.",
        "Mice eat cats.",
    ),
]


def load_order_blind_pairs() -> Dataset:
    originals = [p[0] for p in ORDER_BLIND_PAIRS]
    perturbed = [p[1] for p in ORDER_BLIND_PAIRS]
    texts = originals + perturbed
    meta = {
        "originals": originals,
        "perturbed": perturbed,
        "pairs": ORDER_BLIND_PAIRS,
        "description": "Pairs where word-order reversal changes meaning — exposes order-blind models",
    }
    return texts, meta


# ------------------------------------------------------------------
# 3. News headlines — clustering showcase
# ------------------------------------------------------------------

NEWS_HEADLINES = [
    # Technology
    "Apple releases new iPhone with upgraded AI features",
    "Google unveils next-generation search powered by Gemini",
    "Microsoft Azure expands its machine learning infrastructure",
    "OpenAI launches GPT-5 with improved reasoning capabilities",
    "Meta announces AR glasses for mainstream consumers",
    "Nvidia reports record revenue driven by AI chip demand",
    "Amazon Web Services cuts prices on cloud storage",
    # Finance
    "Federal Reserve holds interest rates steady amid inflation concerns",
    "Stock market rallies as inflation data comes in below forecast",
    "Bitcoin surpasses $80,000 as institutional adoption grows",
    "Goldman Sachs upgrades outlook for emerging market equities",
    "US Treasury yields fall after weak jobs report",
    "Private equity fundraising slows in a high-rate environment",
    "Hedge funds increase exposure to energy sector ETFs",
    # Health
    "FDA approves new cancer immunotherapy with 60% response rate",
    "COVID-19 variant XEC spreads across Europe and Asia",
    "Clinical trial shows Alzheimer's drug slows cognitive decline",
    "Obesity drug Wegovy approved for cardiovascular risk reduction",
    "Global measles cases rise as vaccination rates decline",
    "New blood test detects 50 types of cancer with high accuracy",
    "Antibiotic resistance deaths projected to hit 10 million by 2050",
    # Climate
    "Record Arctic temperatures accelerate ice sheet melt",
    "UN climate summit agrees on new emissions reduction targets",
    "Solar power now cheapest energy source in history",
    "Extreme heat events triple in frequency over past decade",
    "Carbon capture plant reaches 1 million ton milestone",
    "Electric vehicle sales surpass 20% market share globally",
    "Ocean temperatures hit all-time high for third consecutive year",
    # Sports
    "Real Madrid wins Champions League in dramatic penalty shootout",
    "LeBron James announces one more NBA season at age 40",
    "Serena Williams inducted into Tennis Hall of Fame",
    "World Cup 2026 ticket sales open amid massive demand",
    "Tiger Woods completes comeback with PGA Tour victory",
]


def load_news() -> Dataset:
    categories = (
        ["Technology"] * 7
        + ["Finance"] * 7
        + ["Health"] * 7
        + ["Climate"] * 7
        + ["Sports"] * 5
    )
    meta = {
        "categories": categories,
        "description": "News headlines across 5 categories — clustering showcase",
    }
    return NEWS_HEADLINES, meta


# ------------------------------------------------------------------
# 4. NLI-style pairs (entailment / contradiction / neutral)
# ------------------------------------------------------------------

NLI_PAIRS = [
    # (premise, hypothesis, label)
    ("A man is playing guitar on the street.", "A musician is performing outdoors.", "entailment"),
    ("A man is playing guitar on the street.", "A woman is cooking dinner.", "contradiction"),
    ("A man is playing guitar on the street.", "The street is busy.", "neutral"),
    ("The cat is sleeping on the couch.", "An animal is resting.", "entailment"),
    ("The cat is sleeping on the couch.", "The cat is running outside.", "contradiction"),
    ("The cat is sleeping on the couch.", "The couch is comfortable.", "neutral"),
    ("Scientists discovered a new species of deep-sea fish.", "A new animal was found by researchers.", "entailment"),
    ("Scientists discovered a new species of deep-sea fish.", "All marine species are already documented.", "contradiction"),
    ("Scientists discovered a new species of deep-sea fish.", "The ocean contains many mysteries.", "neutral"),
    ("The restaurant received a Michelin star.", "The restaurant is well-regarded for its food.", "entailment"),
    ("The restaurant received a Michelin star.", "The restaurant closed due to poor reviews.", "contradiction"),
    ("The restaurant received a Michelin star.", "The chef trained in Paris.", "neutral"),
]


def load_nli() -> Dataset:
    premises = [p[0] for p in NLI_PAIRS]
    hypotheses = [p[1] for p in NLI_PAIRS]
    labels = [p[2] for p in NLI_PAIRS]
    texts = premises + hypotheses
    meta = {
        "premises": premises,
        "hypotheses": hypotheses,
        "labels": labels,
        "all_texts": texts,
        "description": "NLI-style pairs — entailment/contradiction/neutral geometry probe",
    }
    return texts, meta


# ------------------------------------------------------------------
# 5. E-commerce product descriptions
# ------------------------------------------------------------------

PRODUCTS = [
    ("Wireless Noise-Cancelling Headphones", "Over-ear headphones with active noise cancellation, 30-hour battery life, and premium sound."),
    ("True Wireless Earbuds", "Compact in-ear buds with Bluetooth 5.3, 8-hour battery, IPX5 water resistance."),
    ("Wired Gaming Headset", "USB gaming headset with 7.1 surround sound and noise-cancelling microphone."),
    ("Portable Bluetooth Speaker", "360-degree sound speaker, waterproof, 24-hour playtime, built-in carabiner."),
    ("Smart Thermostat", "Wi-Fi enabled thermostat with learning algorithms, energy usage reports, and voice control."),
    ("Smart Door Lock", "Keyless entry lock with fingerprint scanner, PIN code, and smartphone app."),
    ("Home Security Camera", "1080p indoor camera with night vision, two-way audio, and cloud storage."),
    ("Robot Vacuum Cleaner", "Laser mapping robot vacuum with auto-empty base and carpet boost mode."),
    ("Air Purifier", "HEPA + activated carbon air purifier for rooms up to 500 sq ft, whisper quiet."),
    ("Standing Desk Converter", "Adjustable height desk converter for dual monitors, anti-fatigue mat included."),
    ("Ergonomic Office Chair", "Lumbar-adjustable mesh chair with armrests and 5-year warranty."),
    ("Ultra-wide Monitor", "34-inch curved 3440x1440 monitor with 144Hz refresh and USB-C 65W charging."),
    ("Mechanical Keyboard", "Tenkeyless mechanical keyboard with Cherry MX Red switches and RGB backlight."),
    ("Gaming Mouse", "16000 DPI optical sensor mouse with 6 programmable buttons and 70-hour wireless battery."),
    ("4K Webcam", "4K 60fps webcam with AI background removal, ring light, and privacy shutter."),
    ("Laptop Stand", "Adjustable aluminum laptop stand compatible with 10-17 inch laptops."),
    ("USB-C Hub", "7-in-1 USB-C hub with 4K HDMI, 100W PD, SD card reader, and 3 USB-A ports."),
    ("External SSD", "1TB portable SSD with USB 3.2 Gen2, read speed 1050 MB/s."),
    ("Wireless Charging Pad", "15W fast wireless charger compatible with iPhone and Android, dual-coil."),
    ("Smart Watch", "GPS smartwatch with ECG, SpO2, sleep tracking, and 7-day battery life."),
]


def load_products() -> Dataset:
    titles = [p[0] for p in PRODUCTS]
    descriptions = [p[1] for p in PRODUCTS]
    texts = [f"{t}: {d}" for t, d in PRODUCTS]
    meta = {
        "titles": titles,
        "descriptions": descriptions,
        "texts": texts,
        "description": "E-commerce product descriptions — retrieval and clustering showcase",
    }
    return texts, meta


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

DATASETS = {
    "faq": load_faq,
    "order_blind": load_order_blind_pairs,
    "news": load_news,
    "nli": load_nli,
    "products": load_products,
}


def load_dataset(name: str) -> Dataset:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list(DATASETS)}")
    return DATASETS[name]()


def list_datasets() -> pd.DataFrame:
    rows = []
    for name, fn in DATASETS.items():
        texts, meta = fn()
        rows.append(
            {
                "name": name,
                "n_texts": len(texts),
                "description": meta.get("description", ""),
            }
        )
    return pd.DataFrame(rows)
