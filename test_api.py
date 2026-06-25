# test_api.py
import httpx

QUERY = (
    "If Hamon were to prove that deep compatibility correlates only with a set of "
    "universally measurable behavioral metrics (e.g., the capacity for non-violent "
    "conflict resolution; resilience in ambiguity), would the convergence between "
    "MBTI and Vedic astrology become redundant? If yes, which part of your synthesis "
    "- the empirical rigor or the symbolic depth - is doing all the heavy lifting "
    "simply to maintain structural weight?"
)

with httpx.Client(timeout=900) as client:
    with client.stream("POST", "http://localhost:8000/query", json={"query": QUERY}) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                print(line[6:], flush=True)