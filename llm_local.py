# llm_local.py — GROQ + LLAMA 3.1 8B (FAST, FREE, WORKS 100%)

import os
import streamlit as st
import requests

# ------------------- STATIC FALLBACKS -------------------
STATIC_FALLBACK = {
    'Cash': "Tip: Consider using a Credit Card for larger purchases to earn rewards and build credit.",
    'Debit Card': "Insight: Debit is practical for daily spending. A Credit Card can offer cashback, rewards, and better buyer protections when used responsibly.",
    'Credit Card': "Excellent Choice! You're optimizing for rewards and protection. Consider paying in full each month to avoid interest.",
    'Digital Wallet': "Smart move! Link your Digital Wallet to a rewards Credit Card to earn points while keeping convenience."
}

# ------------------- PROMPT -------------------
def _prompt_for_customer(summary: dict) -> str:
    summary_str = ", ".join(f"{k}: {v}" for k, v in summary.items())
    return (
        "You are a friendly, concise financial advisor. Write a short, natural recommendation (40–100 words) "
        "encouraging this customer to consider a rewards Credit Card. Highlight 1–2 benefits based on their behavior. "
        "End with a clear call-to-action. Never mention personal data.\n\n"
        f"Customer behavior: {summary_str}\n\n"
        "Recommendation:"
    )

# ------------------- GROQ GENERATION (FAST & FREE) -------------------
def generate_with_groq(customer_summary: dict) -> str:
    prompt = _prompt_for_customer(customer_summary)
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=data, timeout=20)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    return None

# ------------------- SAFE PUBLIC FUNCTION -------------------
@st.cache_data(show_spinner=False, ttl=60*60)
def safe_generate_tip(customer_summary: dict, fallback_label: str) -> str:
    try:
        tip = generate_with_groq(customer_summary)
        if tip and len(tip) > 20:
            return tip
    except Exception as e:
        st.warning(f"LLM failed → using fallback ({e})")

    return STATIC_FALLBACK.get(fallback_label, "Consider a rewards Credit Card for extra benefits!")
