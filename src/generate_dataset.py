"""Generate the code-mixed (romanised Telugu + English) mobile-network complaint dataset.

IMPORTANT: this corpus is *synthetic*. Reviews are composed from hand-written phrase banks
(see data/DATA_CARD.md). It exists because no labelled Telugu-English mobile-network
complaint corpus is publicly available. Scores on it must not be read as real-world accuracy.

Labels: negative / positive / mixed (a review that praises one aspect and criticises another).
The label is decided by the *decisive* sentence(s), never by the polite opening, as required by
the task statement ("customers write polite opening sentences and state the real complaint
at the end").

Run:  python src/generate_dataset.py          (seeded, deterministic)
"""
import csv
import random
import re
from pathlib import Path

SEED = 42
OUT = Path(__file__).resolve().parents[1] / "data" / "code_mixed_complaints.csv"

# --------------------------------------------------------------------------------------
# Phrase banks. Each (aspect, polarity) list has 10 phrases; the last 3 are HELD OUT and are
# only used (with 50% probability) in test reviews, so the test set measures generalisation
# to unseen wording rather than template memorisation.
# --------------------------------------------------------------------------------------
POS = {
    "signal": [
        "network chala bagundi",
        "signal full ga untundi",
        "mana area lo coverage excellent ga undi",
        "indoor lo kuda signal clear ga vastundi",
        "tower range chala baga pani chestondi",
        "basement lo kuda network dorukutundi",
        "signal strength ippudu super ga undi",
        "ee network tho e problem ledu, coverage great",
        "range vishayam lo mee service number one",
        "hills daggara kuda full bars vachayi",
    ],
    "speed": [
        "net speed super ga undi",
        "5G speed chala fast ga undi",
        "download okka minute lo ayipoindi",
        "YouTube buffering lekunda smooth ga nadustundi",
        "data speed chala bagundi",
        "video streaming HD lo problem lekunda avutundi",
        "internet twaraga load avtundi",
        "browsing speed really impressive ga undi",
        "app updates seconds lo ayipotunnayi",
        "online class lo net apudu aagaledu",
    ],
    "calls": [
        "call quality clear ga undi",
        "voice chala crystal clear ga vastundi",
        "call drop ippudu avvatledu",
        "video call lo voice break avvatledu",
        "calls ventane connect avtunnayi",
        "VoLTE calls chala baga unnayi",
        "phone lo matladutunte echo ledu",
        "long calls kuda disconnect avvaledu",
        "call connect avvadaniki time pattaledu",
        "network busy message ee madhya raaledu",
    ],
    "billing": [
        "recharge easy ga ayipoindi",
        "bill amount correct ga vachindi",
        "plan benefits clear ga unnayi",
        "balance deduct avvakunda recharge success ayindi",
        "extra charges emi veyaledu",
        "validity extend avvadam sulabham ga undi",
        "refund twaraga account lo padindi",
        "app lo recharge cheyyadam chala simple",
        "offers genuine ga unnayi, hidden charges levu",
        "postpaid bill details clear ga ichharu",
    ],
    "care": [
        "customer care vallu chala baga help chesaru",
        "support team ventane respond ayyaru",
        "complaint pettina same day solve chesaru",
        "care executive chala polite ga matladaru",
        "problem 10 minutes lo fix chesaru",
        "store lo staff manchiga guide chesaru",
        "helpline call ventane pick chesaru",
        "technician time ki vachi sort chesaru",
        "chat support lo quick ga reply vachindi",
        "service request status updates correct ga ichharu",
    ],
    "service": [
        "OTP seconds lo vachindi",
        "SIM activation same day ayipoindi",
        "port cheyyadam chala easy ga ayindi",
        "network nundi problem emi raaledu",
        "mee service tho nenu happy ga unna",
        "overall experience chala bagundi",
        "SMS delivery ventane avutundi",
        "roaming lo kuda network baga pani chesindi",
        "eppudu use chesina service stable ga undi",
        "new tower vachaka anni baga unnayi",
    ],
}

NEG = {
    "signal": [
        "signal asalu ledu",
        "network chala worst ga undi",
        "tower problem valla range ledu",
        "intlo signal raavatledu",
        "no network ani vastune undi",
        "coverage mana area lo asalu ledu",
        "signal bars okati kuda ledu",
        "network baaga ledu, ellappudu searching",
        "range ledu, outdoor lo kuda cut avutundi",
        "ee area lo emergency call kuda cheyyalem",
    ],
    "speed": [
        "net speed chala slow ga undi",
        "internet nathiga nadustundi",
        "YouTube buffering avtune undi",
        "4G lo kuda 2G speed vastundi",
        "WhatsApp photos download avvatledu",
        "page load avvadaniki chala time pattundi",
        "data speed baaga ledu",
        "online class lo net aagipotundi",
        "app update ki ganta pattindi",
        "speed test lo 1 mbps kuda raaledu",
    ],
    "calls": [
        "call drop avtune undi",
        "call connect avvatledu",
        "matladutunte voice cut avtundi",
        "okka call kuda clear ga vinipinchadam ledu",
        "video call lo voice break avtundi",
        "outgoing calls fail avutunnayi",
        "calls lo echo vastundi, ardham avvadu",
        "network busy ani message vastundi",
        "incoming calls ravatledu",
        "call quality bagaledu",
    ],
    "billing": [
        "bill lo extra charges vesaru",
        "recharge chesina balance cut ayindi kani plan activate avvaledu",
        "validity mundu ne aipoindi",
        "unnecessary deductions jarugutunnayi",
        "refund ivvaledu, 15 rojulu ayindi",
        "plan benefits emi raavatledu",
        "postpaid bill double ga vachindi",
        "offer ani cheppi hidden charges vesaru",
        "app lo recharge fail ayyi money poyindi",
        "data pack apply avvakunda balance ayipoindi",
    ],
    "care": [
        "customer care vallu phone ethatledu",
        "complaint petti 5 rojulu ayindi, solve avvaledu",
        "support team respond avvatledu",
        "care executive rude ga matladaru",
        "same problem ki 4 saarlu complaint chesanu, result ledu",
        "store lo staff pattinchukovatledu",
        "helpline lo ganta wait chesina evaru raaledu",
        "technician vastanu ani raaledu",
        "chat support lo bot loop lone untundi",
        "ticket close chesesaru, problem alage undi",
    ],
    "service": [
        "OTP raavatledu",
        "SIM activation 3 rojulu pattindi",
        "port chesina tarvata network poyindi",
        "mee service tho nenu chala disappoint ayyanu",
        "SMS delivery avvatledu",
        "roaming lo network asalu pani cheyyaledu",
        "overall experience chala kharabu ga undi",
        "eppudu choosina service unstable ga untundi",
        "new tower ani cheppi ippatiki activate cheyyaledu",
        "ee madhya network quality chala padipoindi",
    ],
}
N_HELD = 3  # last N_HELD phrases of every list are held out

# Negated phrases flip polarity - a composition test for the recurrent layer.
# "problem ledu" (= no problem) is positive; "baaga ledu" (= not good) is negative.
NEGATION_POS = [
    "call drop ledu ippudu",
    "net slow ga ledu, chala fast ga undi",
    "signal problem emi ledu",
    "bill lo extra charges emi ledu",
    "customer care tho ye ibbandi ledu",
    "network gurinchi complaint cheyyadaniki emi ledu",
    "buffering ane problem ledu",
    "recharge fail avvadam ledu",
]
NEGATION_NEG = [
    "network baaga ledu",
    "speed antha manchiga ledu",
    "signal sarigga ledu",
    "customer care help chesinattu ledu",
    "call quality bagundi ani cheppalem",
    "net fast ga undi anadam abaddam, slow ga undi",
    "recharge sulabham ga ledu",
    "coverage okay ga ledu",
]
NEGATION_HELD_POS = NEGATION_POS[-2:]
NEGATION_HELD_NEG = NEGATION_NEG[-2:]

IRONY = [
    "Wah wah, what a service",
    "Great job team, chala baga chesaru",
    "Super network sir, thanks a lot",
    "Excellent, chala santhosham",
    "Brilliant service anukunna",
    "Congrats mee network ki",
    "Wow, 5G ani cheppi 2G ichharu, great",
    "Chala manchi service, nijam ga",
]

PRAISE_OPEN = [
    "Namaskaram sir, mee company ni nenu 5 years nundi nammutunna",
    "First of all mee team ki thanks, chala respect undi",
    "Hello team, meeru manchi work chestunnaru ani telusu",
    "Dear sir/madam, mee brand ante naku chala ishtam",
    "Hi support team, mee seva ki dhanyavadalu",
    "Good morning sir, mee service gurinchi manchiga vinnanu",
]
NEUTRAL_OPEN = [
    "Namaskaram sir",
    "Hello customer care team",
    "Dear sir/madam, naa feedback share cheyyalani anukuntunna",
    "Hi team, kontha time teesukoni ee review rastunna",
    "Sir, meeku oka vishayam cheppali",
    "Good evening, naa mobile number tho oka vishayam undi",
]
FILLER = [
    "Nenu last 3 years nundi mee SIM vadutunna",
    "Maa intlo andaru same network vadataru",
    "Nenu Hyderabad lo Gandipet daggara untanu",
    "Monna Tuesday evening 7 pm time lo ee vishayam gamaninchanu",
    "Naa phone Android 12, anni updates install chesanu",
    "Maa friends ki kuda ide vishayam cheppanu",
    "Nenu work from home chestanu, anduke ee vishayam important",
    "Prepaid plan 299 vadutunna every month",
    "Maa nanna garu kuda ee number use chestaru",
    "Weekend lo ooru velli vachanu, appati nundi gamaninchanu",
    "Phone restart chesanu, SIM marchi chusanu",
    "Maa area lo konni rojula nundi construction jarugutondi",
    "Nenu online class teacher ni, daily 4 hours phone vadatanu",
    "Ee review rayadaniki naku time dorikindi ippude",
]
CLOSERS_NEUTRAL = [
    "Please look into this.",
    "Thank you.",
    "Dhanyavadalu.",
    "Regards.",
    "Pls do the needful.",
    "Meeru chusi cheppandi.",
]

PREFIX = ["", "", "", "actually ", "honestly ", "ee madhya ", "monna nundi ", "daily ", "ippudu ", "nijam ga ", "mana area lo "]
SUFFIX = ["", "", "", " sir", " anna", " andi", " really", " yaar"]
CONNECTORS = ["kani", "but", "however", "ayite", "okati bagundi kani", "inkoka vishayam ante,", "anthe kaani", "though"]

# Romanised-Telugu spelling variation (the "spelling variation" error category)
VARIANTS = {
    "chala": ["chaala", "chalaa", "chaaala", "chala"],
    "bagundi": ["baagundi", "bagundhi", "bagumdi", "baagundhi"],
    "baga": ["baaga", "bagaa", "baagaa"],
    "ledu": ["ledhu", "leduu", "ledu"],
    "undi": ["undhi", "vundi", "vundhi", "undi"],
    "ga": ["gaa", "ga"],
    "avtune": ["avthune", "avtunne", "avutune"],
    "avtundi": ["avthundi", "avutundi", "avtundhi"],
    "network": ["netwrk", "nettwork", "network"],
    "signal": ["signel", "sigal", "signal"],
    "problem": ["problm", "prblm", "problem"],
    "customer": ["custmer", "costumer", "customer"],
    "recharge": ["recharg", "rechrge", "recharge"],
    "slow": ["sloww", "slooow", "slow"],
    "speed": ["spead", "speeed", "speed"],
    "vallu": ["valu", "vaallu", "vallu"],
    "chesaru": ["chesaaru", "chesharu", "chesaru"],
    "raaledu": ["raledu", "rale", "raaledhu"],
    "vastundi": ["vasthundi", "vastundhi", "vastundi"],
    "asalu": ["asalu", "asaluu", "assalu"],
}


def pools(table, held):
    out = {}
    for aspect, items in table.items():
        out[aspect] = items[len(items) - N_HELD:] if held else items[:len(items) - N_HELD]
    return out


def pick_phrase(rng, table, aspect, held):
    pool = pools(table, held)[aspect]
    p = rng.choice(pool)
    return rng.choice(PREFIX) + p + rng.choice(SUFFIX)


def decorate(rng, text):
    if rng.random() < 0.15:
        text = text.upper() if rng.random() < 0.3 else text.capitalize()
    return text


def apply_noise(rng, text):
    """Spelling variation + occasional typos; returns noisy text."""
    out = []
    for w in text.split(" "):
        base = re.sub(r"[^\w]", "", w.lower())
        if base in VARIANTS and rng.random() < 0.6:
            rep = rng.choice(VARIANTS[base])
            w = re.sub(re.escape(base), rep, w, flags=re.I) if base in w.lower() else rep
        elif len(base) >= 6 and rng.random() < 0.05:
            i = rng.randrange(1, len(w) - 1)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]  # swap two letters
        out.append(w)
    return " ".join(out)


def sentence(s):
    s = s.strip()
    return s if s[-1] in ".!?" else s + "."


def build_decisive(rng, label, held, category):
    aspects = list(POS)
    if label == "mixed" and category == "mixed_distant":
        a1, a2 = rng.sample(aspects, 2)
        pos = pick_phrase(rng, POS, a1, held)
        neg = pick_phrase(rng, NEG, a2, held)
        fill = " ".join(sentence(f) for f in rng.sample(FILLER, rng.randint(2, 4)))
        first, second = (pos, neg) if rng.random() < 0.5 else (neg, pos)
        return f"{sentence(first)} {fill} {rng.choice(CONNECTORS).capitalize()} {second}"
    if label == "mixed":
        a1, a2 = rng.sample(aspects, 2)
        pos = pick_phrase(rng, POS, a1, held)
        neg = pick_phrase(rng, NEG, a2, held)
        parts = [pos, neg] if rng.random() < 0.5 else [neg, pos]
        return f"{parts[0]}, {rng.choice(CONNECTORS)} {parts[1]}"
    if category == "negation":
        pool = (NEGATION_HELD_POS if held else NEGATION_POS[:-2]) if label == "positive" else \
               (NEGATION_HELD_NEG if held else NEGATION_NEG[:-2])
        return rng.choice(PREFIX) + rng.choice(pool) + rng.choice(SUFFIX)
    a = rng.choice(aspects)
    if label == "positive":
        return pick_phrase(rng, POS, a, held)
    neg = pick_phrase(rng, NEG, a, held)
    if category == "sarcasm":
        irony = rng.choice(IRONY)
        return f"{irony}, {neg}" if rng.random() < 0.5 else f"{neg}. {irony}"
    return neg


CATEGORY_WEIGHTS = {
    "negative": [("plain", .30), ("polite_opening", .25), ("long_buried", .20), ("sarcasm", .15), ("negation", .10)],
    "positive": [("plain", .35), ("polite_opening", .25), ("long_buried", .20), ("negation", .20)],
    "mixed": [("plain", .25), ("polite_opening", .20), ("long_buried", .25), ("mixed_distant", .30)],
}


def pick_category(rng, label):
    cats, w = zip(*CATEGORY_WEIGHTS[label])
    return rng.choices(cats, weights=w)[0]


def build_review(rng, label, held):
    category = pick_category(rng, label)
    decisive = build_decisive(rng, label, held, category)
    parts = []
    if category == "polite_opening":
        for _ in range(rng.randint(1, 3)):
            parts.append(rng.choice(PRAISE_OPEN + NEUTRAL_OPEN))
    elif category == "long_buried":
        parts.append(rng.choice(NEUTRAL_OPEN))
        parts += rng.sample(FILLER, rng.randint(3, 6))
    parts.append(decisive)
    if category != "plain" and rng.random() < 0.6:
        parts.append(rng.choice(CLOSERS_NEUTRAL))
    text = " ".join(sentence(p) for p in parts)
    noisy = rng.random() < 0.5
    if noisy:
        text = apply_noise(rng, text)
    return decorate(rng, text), category, noisy


def norm(t):
    return re.sub(r"\W+", " ", t.lower()).strip()


def main():
    rng = random.Random(SEED)
    labels = ["negative", "positive", "mixed"]
    plan = {"train": (700, 0.0), "val": (150, 0.5), "test": (150, 0.5)}  # per class, held-out share
    rows, seen_train = [], set()
    for split in ["train", "val", "test"]:
        n, held_share = plan[split]
        for label in labels:
            made = 0
            while made < n:
                held = rng.random() < held_share
                text, cat, noisy = build_review(rng, label, held)
                key = norm(text)
                if split == "train":
                    if key in seen_train:
                        continue
                    seen_train.add(key)
                elif key in seen_train:  # no train/eval text leakage, no intra-split duplicates
                    continue
                else:
                    seen_train.add(key)
                rows.append({"text": text, "label": label, "category": cat, "spelling_noise": int(noisy),
                             "split": split, "phrase_pool": "unseen" if held else "seen"})
                made += 1
    rng.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "label", "category", "spelling_noise", "split", "phrase_pool", "n_tokens"])
        w.writeheader()
        for i, r in enumerate(rows):
            r["id"] = i
            r["n_tokens"] = len(re.findall(r"[a-z0-9']+", r["text"].lower()))
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
