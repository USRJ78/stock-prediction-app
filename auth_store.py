# auth_store.py
import os
import pandas as pd

PREMIUM_FILE = "premium_users.csv"

def load_premium_users() -> set:
    if not os.path.exists(PREMIUM_FILE):
        return set()
    try:
        df = pd.read_csv(PREMIUM_FILE)
        if "email" not in df.columns:
            return set()
        return set(df["email"].astype(str).str.lower().str.strip().dropna().tolist())
    except Exception:
        return set()

def save_premium_user(email: str) -> None:
    email = str(email).lower().strip()
    if not email:
        return

    if os.path.exists(PREMIUM_FILE):
        try:
            df = pd.read_csv(PREMIUM_FILE)
        except Exception:
            df = pd.DataFrame(columns=["email"])
    else:
        df = pd.DataFrame(columns=["email"])

    if "email" not in df.columns:
        df = pd.DataFrame(columns=["email"])

    existing = set(df["email"].astype(str).str.lower().str.strip().dropna().tolist())
    if email in existing:
        return

    df = pd.concat([df, pd.DataFrame({"email": [email]})], ignore_index=True)
    df.to_csv(PREMIUM_FILE, index=False)
