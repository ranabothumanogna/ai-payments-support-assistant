"""
tools.py — the "Agent" layer.

WHAT THIS FILE DOES:
An "AI Agent" is just an LLM that can decide to call a function (a "tool")
instead of only generating text. Here we simulate a real payments backend
with a fake in-memory database of transactions, and expose one tool:
`check_transaction_status(transaction_id)`.

In a real Razorpay-style system this would call an actual database or
microservice. We mock it so the whole project runs standalone with no
external dependencies.
"""

import re

# --- Fake transaction database ---
_MOCK_TRANSACTIONS = {
    "TXN1001": {"status": "SUCCESS", "amount": 1499.00, "method": "UPI"},
    "TXN1002": {"status": "FAILED", "amount": 899.00, "method": "Card", "reason": "INSUFFICIENT_FUNDS"},
    "TXN1003": {"status": "PENDING", "amount": 2500.00, "method": "Netbanking"},
    "TXN1004": {"status": "FAILED", "amount": 4999.00, "method": "Card", "reason": "GATEWAY_TIMEOUT"},
}


def check_transaction_status(transaction_id: str) -> dict:
    """The actual 'tool' function. Given a transaction ID, return its status."""
    txn = _MOCK_TRANSACTIONS.get(transaction_id.upper())
    if not txn:
        return {"found": False, "message": f"No transaction found with ID {transaction_id}"}
    return {"found": True, "transaction_id": transaction_id.upper(), **txn}


def extract_transaction_id(text: str) -> str | None:
    """Very simple 'intent detection': look for a pattern like TXN1234 in the user's message.

    A production agent would use the LLM's function-calling feature to decide
    this. We use a lightweight regex here so the project runs without needing
    an LLM API key for this specific step — but app/main.py shows where you'd
    swap this for real LLM-based tool-calling.
    """
    match = re.search(r"\bTXN\d{4}\b", text.upper())
    return match.group(0) if match else None
