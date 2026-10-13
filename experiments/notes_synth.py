"""Synthetic personal-notes QA — the app's domain, with exact answers.

Each item = a short note of mixed personal facts (bills, birthdays, meds, warranties) + one question whose
answer is computed from the facts; the other facts are distractors. Train and test differ in everything
surface-level, so a test win means the skill transferred, not the templates:
  - name / bill / item / med pools are disjoint
  - fact phrasings and question phrasings: train uses variants 0-1, test uses variant 2
Question types are shared (holding a type out would test something else: unseen skills).

Usage: python experiments/notes_synth.py   (runs the self-check, prints two examples)
"""
import random
from datetime import date

POOLS = {
    "train": {"names": ["Aisha", "Omar", "Priya", "Daniel", "Mei", "Carlos", "Fatima", "Liam", "Sara", "Kenji"],
              "bills": ["rent", "electricity", "internet", "gym", "phone", "water"],
              "items": ["laptop", "fridge", "headphones", "washing machine", "camera"],
              "meds": ["ibuprofen", "amoxicillin", "vitamin D", "metformin"]},
    "test": {"names": ["Leila", "Arjun", "Hannah", "Diego", "Yuki", "Ibrahim"],
             "bills": ["insurance", "streaming", "parking", "school fees"],
             "items": ["TV", "microwave", "e-bike"],
             "meds": ["paracetamol", "cetirizine", "omeprazole"]},
}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]

FACT = {  # 3 phrasings per fact type: train uses [0, 1], test uses [2]
    "bill": ["{name} {amt} due on the {day}th", "{Name}: {amt}, pay by day {day}", "pay {amt} for {name} on the {day}th of each month"],
    "bday": ["{name} birthday {d} {M}", "{name}'s bday is {M} {d}", "remember {name} turns a year older on {d} {M}"],
    "med": ["{med} {n} tablet(s) {k}x a day", "{Med}: take {n} tab(s), {k} times daily", "doctor says {med}, {n} tablet(s) {k} times per day"],
    "warranty": ["{item} bought {M} {y}, {w}-month warranty", "{Item} purchased {M} {y} ({w} months warranty)", "got the {item} in {M} {y}, covered for {w} months"],
}
QUESTION = {
    "bills_total": ["How much do I pay in bills each month?", "What's my total monthly bill amount?", "Add up everything I pay every month — what's the total?"],
    "bill_first": ["Which bill is due earliest in the month?", "What do I have to pay first each month?", "Of my monthly payments, which one comes due first?"],
    "bill_year": ["How much do I spend on {bill} per year?", "What's my yearly cost for {bill}?", "Over twelve months, how much goes to {bill}?"],
    "bday_days": ["Today is {today}. How many days until {name}'s birthday?", "It's {today}. How many days are left before {name}'s birthday?", "Date today: {today}. In how many days is {name}'s birthday?"],
    "med_week": ["How many {med} tablets do I take per week?", "What's my weekly {med} tablet count?", "In seven days, how many {med} tablets will I have taken?"],
    "warranty_year": ["In what year does the warranty on my {item} run out?", "Which year does my {item}'s warranty end?", "When (which year) is my {item} no longer under warranty?"],
}
STYLES = [lambda f: "\n".join(f"- {x}" for x in f), lambda f: "; ".join(f) + ".", lambda f: "\n".join(f)]


def fmt(d):
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def make(rng, split):
    p, v = POOLS[split], [2] if split == "test" else [0, 1]
    ph = lambda opts: opts[rng.choice(v)]
    # unique due days → "first" is unambiguous
    bills = [(b, rng.choice(range(500, 20001, 100)), d) for b, d in zip(rng.sample(p["bills"], 3), rng.sample(range(1, 29), 3))]
    bdays = [(n, date(2027, rng.randint(1, 12), rng.randint(1, 28))) for n in rng.sample(p["names"], 2)]
    med = (rng.choice(p["meds"]), rng.randint(1, 2), rng.randint(1, 3))
    items = [(it, rng.randint(1, 12), rng.randint(2021, 2026), rng.choice([12, 18, 24, 36])) for it in rng.sample(p["items"], 2)]

    facts = [ph(FACT["bill"]).format(name=b, Name=b.capitalize(), amt=a, day=d) for b, a, d in bills]
    facts += [ph(FACT["bday"]).format(name=n, d=d.day, M=MONTHS[d.month - 1]) for n, d in bdays]
    facts.append(ph(FACT["med"]).format(med=med[0], Med=med[0].capitalize(), n=med[1], k=med[2]))
    facts += [ph(FACT["warranty"]).format(item=i, Item=i.capitalize(), M=MONTHS[m - 1], y=y, w=w) for i, m, y, w in items]
    rng.shuffle(facts)
    note = rng.choice(STYLES)(facts) if split == "train" else STYLES[2](facts)

    qtype = rng.choice(list(QUESTION))
    if qtype == "bills_total":
        ans = sum(a for _, a, _ in bills)
        why = " + ".join(str(a) for _, a, _ in bills) + f" = {ans}"
        q = ph(QUESTION[qtype])
    elif qtype == "bill_first":
        b = min(bills, key=lambda x: x[2])
        ans, why, q = b[0], ", ".join(f"{n} on day {d}" for n, _, d in bills) + f"; earliest is {b[0]}", ph(QUESTION[qtype])
    elif qtype == "bill_year":
        b, a, _ = rng.choice(bills)
        ans, why, q = a * 12, f"{a} × 12 = {a * 12}", ph(QUESTION[qtype]).format(bill=b)
    elif qtype == "bday_days":
        n, bd = rng.choice(bdays)
        today = date(2026, rng.randint(1, 12), rng.randint(1, 28))
        nxt = bd.replace(year=today.year)
        nxt = nxt if nxt >= today else bd.replace(year=today.year + 1)
        ans = (nxt - today).days
        why = f"from {fmt(today)} to {fmt(nxt)} is {ans} days"
        q = ph(QUESTION[qtype]).format(today=fmt(today), name=n)
    elif qtype == "med_week":
        m, n, k = med
        ans, why, q = n * k * 7, f"{n} × {k} per day × 7 days = {n * k * 7}", ph(QUESTION[qtype]).format(med=m)
    else:
        i, m, y, w = rng.choice(items)
        end = (y * 12 + m - 1) + w
        ans = end // 12
        why = f"{MONTHS[m - 1]} {y} + {w} months = {MONTHS[end % 12]} {ans}"
        q = ph(QUESTION[qtype]).format(item=i)
    return {"context": note, "question": q, "answer": str(ans), "rationale": why, "qtype": qtype}


def dataset(n, split, seed=0):
    rng = random.Random(f"{split}-{seed}")
    return [make(rng, split) for _ in range(n)]


def _check():
    tr, te = dataset(300, "train"), dataset(300, "test")
    for r in tr + te:  # every answer is stated by its own rationale
        assert r["rationale"].endswith(r["answer"]) or r["answer"] in r["rationale"], r
    tr_text = " ".join(r["context"] + r["question"] for r in tr)
    for w in POOLS["test"]["names"] + POOLS["test"]["items"] + POOLS["test"]["meds"]:
        assert w not in tr_text, f"test-pool word leaked into train: {w}"
    assert {r["qtype"] for r in te} == set(QUESTION), "every question type appears in test"
    # spot-check the date maths
    rng = random.Random(1)
    for _ in range(200):
        r = make(rng, "train")
        if r["qtype"] == "bday_days":
            assert 0 <= int(r["answer"]) <= 365
    assert dataset(5, "test") == dataset(5, "test"), "deterministic"
    print("notes_synth self-check OK")


if __name__ == "__main__":
    _check()
    for r in dataset(1, "train") + dataset(1, "test"):
        print("---\n" + r["context"] + "\nQ: " + r["question"] + "\nA: " + r["rationale"] + " → " + r["answer"])
