# Annotation guide — code-mixed Telugu–English mobile-network complaints

**Task:** assign one sentiment label to a review written in Telugu typed in English letters, mixed with English words
(e.g. *"food chala bagundi but delivery late ayindi"* style, here for mobile-network service).

Labels: `negative`, `positive`, `mixed`.

## Core rule
Label the **decisive content** of the review — what the customer actually reports about the service — not its politeness.
Customers often open with praise or courtesies and state the real complaint at the end.

| Label | When to use | Example |
|---|---|---|
| `negative` | The customer reports problems only. | *signal asalu ledu, customer care vallu phone ethatledu* |
| `positive` | The customer reports a good experience only (including "no problem" statements). | *call drop ledu ippudu, network chala bagundi* |
| `mixed` | The review contains **both** a clearly positive and a clearly negative statement about the service. | *network chala bagundi, kani bill lo extra charges vesaru* |

## Rules for tricky cases
1. **Polite or praising opening, complaint at the end** → label by the complaint (`negative`). *"Mee company ni nammutunna… OTP raavatledu"* is `negative`.
2. **Sarcasm** (positive words, negative meaning: *"Super network sir, thanks a lot, call connect avvatledu"*) → `negative`.
3. **Negation.** Telugu negation is a suffix/particle: *ledu / ledhu / avvatledu / kaadu*.
   *"problem ledu"* (no problem) = `positive`; *"network baaga ledu"* (network is not good) = `negative`.
4. **Praise for one aspect + complaint about another** → `mixed`, whichever comes first.
5. **Neutral background sentences** (where the customer lives, which plan, how long a subscriber) carry no sentiment — ignore them.
6. **Spelling variation** (*chala / chaala*, *bagundi / baagundhi*, typos, ALL CAPS) never changes the label.
7. If you genuinely cannot decide between two labels, mark it in a `note` column and pick the label you would defend; unclear items are reported in the error analysis.

## Procedure for the second annotator (inter-annotator agreement)
1. Open `annotation/second_annotator_sheet.csv` (210 test reviews, **labels hidden**).
2. Fill `label_annotator_b` for every row using only this guide.
3. Run `python src/kappa.py annotation/key_annotator_a.csv annotation/second_annotator_sheet.csv`.
   Interpret κ: <0.4 weak, 0.4–0.6 moderate, 0.6–0.8 substantial, >0.8 near-perfect.
