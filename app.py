import streamlit as st
import json
import pandas as pd
import os
from openai import OpenAI
import streamlit as st
# ---------------- SETUP ----------------

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

st.set_page_config(
    page_title="AI Risk Assessment – eBAJA",
    layout="wide"
)

# ---------------- HELPERS ----------------
def normalize_score(value):
    if isinstance(value, (int, float)):
        return int(value)
    value = str(value).lower()
    return {"low": 1, "medium": 3, "high": 5}.get(value, 3)

def normalize_confidence(value):
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).lower()
    return {"low": 0.3, "medium": 0.6, "high": 0.9}.get(value, 0.6)

def extract_json(text):
    start = text.find("[")
    end = text.rfind("]") + 1
    return json.loads(text[start:end])

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

# ---------------- UI ----------------
st.title("AI Risk Assessment ")
st.caption("eBAJA ATV Manufacturing | Scenario-Based Risk Intelligence")

uploaded = st.file_uploader("Upload Risk Dataset (CSV)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    df.ffill(inplace=True)

    st.subheader(" Input Dataset")
    st.dataframe(df, use_container_width=True)

    if st.button(" Run AI Risk Assessment"):
        with st.spinner("Analyzing risks using AI..."):
            prompt = f"""
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
                prompt
            )

            risks = extract_json(raw)

            for r in risks:
                r["likelihood"] = normalize_score(r["likelihood"])
                r["impact"] = normalize_score(r["impact"])
                r["confidence"] = normalize_confidence(r["confidence"])
                r["score"] = round(
                    r["likelihood"] * r["impact"] * r["confidence"], 2
                )

            risks = sorted(risks, key=lambda x: x["score"], reverse=True)

        st.success("Risk analysis complete ✅")

        # ---------------- DISPLAY ----------------
        st.subheader(" Prioritized Risks")

        for r in risks:
            with st.container(border=True):
                st.markdown(f"### ⚠️ {r['risk_name']}")
                st.markdown(f"**Scenario:** {r['scenario']}")
                st.markdown(f"**Score:** `{r['score']}`")
                st.markdown(f"**Reason:** {r['reason']}")
                st.markdown(f"**Early Indicators:** {r['early_indicators']}")

        st.subheader("📥 Download Results")
        st.download_button(
            "Download JSON",
            json.dumps(risks, indent=2),
            file_name="risk_assessment_output.json"
        )
