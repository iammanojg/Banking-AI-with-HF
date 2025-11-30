# llm_local.py — PUTER.JS INTEGRATION (Free, No Key LLM)

import streamlit as st

# ------------------- STATIC FALLBACKS -------------------
STATIC_FALLBACK = {
    'Cash': "Tip: Consider using a Credit Card for larger purchases to earn rewards and build credit.",
    'Debit Card': "Insight: Debit is practical for daily spending. A Credit Card can offer cashback, rewards, and better buyer protections when used responsibly.",
    'Credit Card': "Excellent Choice! You're optimizing for rewards and protection. Consider paying in full each month to avoid interest.",
    'Digital Wallet': "Smart move! Link your Digital Wallet to a rewards Credit Card to earn points while keeping convenience."
}

# ------------------- PROMPT BUILDER -------------------
def _prompt_for_customer(summary: dict) -> str:
    summary_str = ", ".join(f"{k}: {v}" for k, v in summary.items())
    return (
        "You are a friendly, concise financial advisor. Write a short, natural recommendation (40–100 words) "
        "encouraging this customer to consider a rewards Credit Card. Highlight 1–2 benefits based on their behavior. "
        "End with a clear call-to-action. Never mention personal data.\n\n"
        f"Customer behavior: {summary_str}\n\n"
        "Recommendation:"
    )

# ------------------- JS CALLBACK FOR LLM (FREE PUTER.JS) -------------------
def safe_generate_tip(customer_summary: dict, fallback_label: str) -> str:
    prompt = _prompt_for_customer(customer_summary)
    
    # Embed JS component for client-side LLM call
    result_placeholder = st.empty()
    if st.button("Generate AI Tip"):  # Trigger JS on button click
        result_placeholder.components.v1.html(
            f"""
            <script src="https://js.puter.com/v2/puter.js"></script>
            <script>
                puter.ai.chat('{prompt}', 'gpt-5-nano', {{"max_tokens": 150, "temperature": 0.7}}).then(response => {{
                    // Pass result back to Python via session state or alert for now
                    window.generatedTip = response;
                    alert('Tip generated: ' + response);  // Simple alert for testing — replace with callback
                    window.parent.postMessage({{type: 'tip_generated', tip: response}}, '*');
                }});
            </script>
            """,
            height=0
        )
        # For now, return fallback — upgrade to callback for production
        return STATIC_FALLBACK.get(fallback_label, "Consider a rewards Credit Card for extra benefits!")
    else:
        return STATIC_FALLBACK.get(fallback_label, "Click 'Generate AI Tip' to use free LLM!")
