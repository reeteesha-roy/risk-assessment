import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ================= ENV =================
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# ================= DATA =================
def load_and_clean(path):
    df = pd.read_csv(path)
    df.ffill(inplace=True)
    return df

def extract_json(text):
    """
    Extract first JSON array safely from LLM output
    """
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == -1:
        raise ValueError("No JSON array found in model output")
    return json.loads(text[start:end])

# ================= NORMALIZATION =================
def normalize_score(value):
    """
    Converts LLM outputs like 'High', '3/5', 'Medium', etc. to 1–5 scale
    """
    if isinstance(value, (int, float)):
        return int(value)

    value = str(value).lower().strip()

    mapping = {
        "low": 1,
        "medium": 3,
        "high": 5,
        "very low": 1,
        "very high": 5
    }

    if value in mapping:
        return mapping[value]

    # handle "3/5", "4 out of 5", etc.
    for n in ["1", "2", "3", "4", "5"]:
        if n in value:
            return int(n)

    return 3  # safe default
def normalize_confidence(value):
    """
    Converts confidence outputs like 'High', '0.8', 'Medium', etc. to 0–1 float
    """
    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).lower().strip()

    mapping = {
        "low": 0.3,
        "medium": 0.6,
        "high": 0.9,
        "very low": 0.2,
        "very high": 0.95
    }

    if value in mapping:
        return mapping[value]

    try:
        return float(value)
    except:
        return 0.6  # safe default

# ================= AI CALL =================
def ai_call(system, user):
    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.3,
        max_tokens=900
    )
    return response.choices[0].message.content

# ================= LOAD DATA =================
DATA_PATH = r"C:\Users\R. ROY\OneDrive\Desktop\atv_dataset.csv"
df = load_and_clean(DATA_PATH)

# ================= RISK ANALYSIS =================
risk_prompt = f"""
You are an AI risk assessment engine for an eBAJA ATV manufacturing company.

Scenarios:
- Earthquake
- Pandemic
- War
- Fire

DATA:
{df.to_json(orient="records")}

TASK:
Identify 3–6 major business risks.

Return ONLY a JSON ARRAY with keys:
scenario, risk_name, likelihood, impact, confidence, early_indicators, reason
"""

raw = ai_call(
    "You are a professional enterprise risk analyst.",
    risk_prompt
)

risks = extract_json(raw)

# ================= SCORING =================
for r in risks:
    r["likelihood"] = normalize_score(r["likelihood"])
    r["impact"] = normalize_score(r["impact"])
    r["confidence"] = normalize_confidence(r["confidence"])


    r["score"] = round(
        r["likelihood"] * r["impact"] * r["confidence"], 2
    )

risks = sorted(risks, key=lambda x: x["score"], reverse=True)

# ================= MITIGATION =================
def mitigation(risk):
    prompt = f"""
Risk:
{json.dumps(risk, indent=2)}

Suggest mitigation strategies.

Return JSON with keys:
actions, owner, timeline
"""
    text = ai_call(
        "You assist with enterprise risk mitigation planning.",
        prompt
    )

    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])

for r in risks:
    if r["score"] >= 12:
        r["mitigation"] = mitigation(r)


print("\n=== AI RISK ASSESSMENT OUTPUT ===\n")
print(json.dumps(risks, indent=2))
