"""Writes crates/sky-infer/tests/fixtures/route_bm25.json: invented slates scored by bm25s (the
`bm25` baseline's library and settings), the fixture that sky_infer::route must reproduce.
Run in the pinned environment: <data>/envs/route/bin/python -I scripts/route/make_bm25_fixture.py"""

import json
import os

import bm25s

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "crates", "sky-infer", "tests", "fixtures", "route_bm25.json")

CASES = [
    ("turn on the lights in the hall", ["lights_on: Turns the lights on in a room.", "weather: Gives the weather of a city.", "play_music: Plays a song."]),
    ("What's the weather like in Paris tomorrow? weather please", ["get_forecast: Weather forecast for a city and a day.", "weather: Current weather.", "news: The latest headlines."]),
    ("convert 5 miles to kilometers", ["Unit Converter: Convert units of measurement from one unit to another", "convert_currency: Convert an amount between currencies.", "tool_17: Converts length units."]),
    ("is it of the and to", ["ab: the and of", "cd: it is to"]),
    ("Réveille-moi à 7h", ["alarm_set: Set an alarm for a given time.", "calendar_set: Create an event."]),
    ("book a flight", ["flight_book: Book a flight.", "flight_book: Book a flight."]),
    ("hello", ["only_tool: Says nothing relevant."]),
    ("get the stock price of AAPL and the stock news", ["getStockPrice: Get the stock price of a company by its ticker.", "stock_news: News about a stock.", "crypto_price: Price of a cryptocurrency.", "weather: Weather.", "send_email: Send an email.", "translate: Translate text.", "calculator: Compute an expression."]),
]


def main():
    cases = []
    for q, docs in CASES:
        r = bm25s.BM25(method="lucene")
        r.index(bm25s.tokenize(docs, show_progress=False), show_progress=False)
        qt = bm25s.tokenize([q], return_ids=False, show_progress=False)[0]
        s = [float(x) for x in (r.get_scores(qt) if qt else [0.0] * len(docs))]
        order = sorted(range(len(s)), key=lambda i: (-s[i], i))
        ranks = [0] * len(s)
        for k, i in enumerate(order):
            ranks[i] = k + 1
        name = lambda o: o.split(": ", 1)[0]
        line = "BM25: " + "; ".join(f"{name(o)} score {x:.2f} rank {k}" for o, x, k in zip(docs, s, ranks))
        cases.append({"request": q, "options": docs, "scores": s, "ranks": ranks, "state": f"User: {q.strip()}\n{line}"})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"about": "s6-route-nano: BM25 (bm25s 0.3.16, Lucene, k1 1.5, b 0.75, English stopwords) of each tool option against the request, over the slate's own options; written by scripts/route/make_bm25_fixture.py",
                   "cases": cases}, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
