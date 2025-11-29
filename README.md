```markdown
# Smart Spending Advisor — Streamlit app

Files added:
- app.py — Streamlit app and model training/prediction code.
- requirements.txt — Python package list.
- data/spending_patterns_REALISTIC_97percent.csv — your dataset (place in data/ or upload at runtime).

Quick local run (Windows / macOS / Linux)
1. Clone repo (or open the folder where app.py lives).
   git clone https://github.com/iammanojg/Trail-run-for-streamlit.git
   cd Trail-run-for-streamlit

2. Create & activate a virtual environment:
   python -m venv .venv
   # Windows (PowerShell)
   .venv\\Scripts\\Activate.ps1
   # Windows (cmd)
   .venv\\Scripts\\activate
   # macOS / Linux
   source .venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Put your dataset in: data/spending_patterns_REALISTIC_97percent.csv
   (Or use the sidebar file uploader when the app runs.)

5. Run Streamlit:
   streamlit run app.py

Using the app
- In the sidebar you can upload a CSV, change CatBoost iterations and depth, and retrain.
- Enter a Customer ID into the main input and click "Analyze Spending" to get:
  - Analytical bullet points (feature-based),
  - Predicted payment method and confidence,
  - A generative-style recommendation aimed at encouraging cash/debit/digital-wallet users toward credit card use.

Notes
- The app trains a CatBoost model on the dataset on first run and caches it for subsequent interactions.
- If your real dataset is large, training may take longer; increase/decrease iterations in the sidebar.
- Keep any private dataset out of public repositories. For production, train offline, save the model (joblib) and load the pre-trained model in the app to avoid training in the web app.
```
