# llm_local.py — FREE MLVOCA API (No HF Drama, Instant Text Gen)

import os
import streamlit as st
import requests
import json

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

# ------------------- MLVOCA GENERATION CALL (FREE, NO KEY) -------------------
def generate_with_llm(customer_summary: dict, max_tokens: int = 150) -> str:
    prompt = _prompt_for_customer(customer_summary)
    payload = {
        "model": "tinyllama",  # Fast, free model (or "deepseek-r1:1.5b" for better quality)
        "prompt": prompt,
        "stream": False  # Single response, no streaming
    }

    try:
        response = requests.post(
            "https://mlvoca.com/api/generate",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "").strip()
        else:
            raise RuntimeError(f"API error {response.status_code}: {response.text}")
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}")

# ------------------- SAFE PUBLIC FUNCTION -------------------
@st.cache_data(show_spinner=False, ttl=60*60)
def safe_generate_tip(customer_summary: dict, fallback_label: str) -> str:
    try:
        generated = generate_with_llm(customer_summary)
        if generated and len(generated) > 15:
            return generated
    except Exception as e:
        st.warning(f"LLM generation failed → using fallback ({e})")

    return STATIC_FALLBACK.get(fallback_label, "Consider a rewards Credit Card for extra benefits!")
