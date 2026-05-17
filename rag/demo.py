from app import ask, AskRequest

tests = [
    ("intern",   "What is the leave policy?",            "Answer from hr_policy.txt"),
    ("intern",   "What is Priya salary?",                "Refusal, denied > 0"),
    ("finance",  "What was Q3 revenue?",                 "Answer cites finance_q3.txt"),
    ("engineer", "Any high-severity incidents recently?","Answer cites incidents.csv"),
    ("hr",       "Show me audit log entries",            "Refusal - audit.json admin-only"),
]

SEP = "=" * 60

def run():
    print("\nRAG Intelligence — Demo\n")
    for i, (role, question, expected) in enumerate(tests, 1):
        print(SEP)
        print(f"TEST {i} | role={role}")
        print(f"Q: {question}")
        print(f"Expected: {expected}")
        print("-" * 60)
        r = ask(AskRequest(question=question, role=role))
        print(f"ANSWER: {r.answer}")
        print(f"Citations:      {r.citations}")
        print(f"Chunks used:    {r.chunks_used}")
        print(f"Denied sources: {r.denied_sources}")
        passed = r.chunks_used > 0 or r.denied_sources > 0
        print(f">>> {'PASS' if passed else 'FAIL'}")
    print(SEP)
    print("\nAll tests complete. Check audit.log for the full trail.")

if __name__ == "__main__":
    run()
