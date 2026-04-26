"""
Chat Subject Classifier
-----------------------
Regex-based classification of chatbot inputs into 6 categories:
tarjeta, transferencia, crédito, saldo, queja, consulta.

Uses accent normalization and per-category pattern matching.
"""

import polars as pl
import re
import unicodedata
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset_50k_anonymized.parquet")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")


CATEGORIES = ["tarjeta", "transferencia", "crédito", "credito_auto", "credito_personal", "credito_solicitud", "credito_linea", "saldo", "queja", "consulta", "despedida", "saludo"]

PATTERNS = {
    "tarjeta": [
        r"tarjeta",
        r"bloquear",
        r"bloqueo",
        r"activar",
        r"pin",
        r"débito",
        r"crédito",
    ],
    "transferencia": [
        r"transferir",
        r"transferencia",
        r"enviar\s+dinero",
        r"monto",
        r"destino",
        r"destinatario",
        r"cuenta",
        r"clabe",
        r"banco",
        r"pago",
    ],
    "crédito": [
        r"crédito",
        r"préstamo",
        r"tasa",
        r"cuota",
        r"aprobación",
    ],
    "credito_auto": [
        r"crédito\s+auto",
        r"auto\s+crédito",
        r"préstamo\s+auto",
        r"crédito\s+automotriz",
        r"enganche",
        r"auto\s+hey",
        r"crédito\s+vehículo",
    ],
    "credito_personal": [
        r"crédito\s+personal",
        r"préstamo\s+personal",
        r"tasa\s+de\s+interés",
        r"cuota\s+mensual",
        r"amortización",
        r"plazo\s+del\s+crédito",
    ],
    "credito_solicitud": [
        r"solicitar\s+crédito",
        r"solicitar\s+préstamo",
        r"aprobación\s+de\s+crédito",
        r"requisitos\s+para\s+crédito",
        r"quiero\s+solicitar\s+crédito",
    ],
    "credito_linea": [
        r"línea\s+de\s+crédito",
        r"crédito\s+rotativo",
        r"aumento\s+de\s+línea",
        r"límite\s+de\s+crédito",
    ],
    "saldo": [
        r"saldo",
        r"balance",
        r"disponible",
        r"consultar",
    ],
    "queja": [
        r"problema",
        r"error",
        r"falló",
        r"fallo",
        r"no\s+funciona",
        r"no\s+llega",
        r"no\s+procesó",
        r"reclamo",
        r"queja",
    ],
    "despedida": [
        r"gracias",
        r"muchas\s+gracias",
        r"adiós",
        r"hasta\s+luego",
        r"nos\s+vemos",
        r"hasta\s+mañana",
        r"te\s+agradezco",
        r"cuídese",
    ],
    "saludo": [
        r"^hola$",
        r"buenos\s+días",
        r"buenas\s+tardes",
        r"buenas\s+noches",
        r"^hey$",
        r"qué\s+tal",
        r"buen\s+día",
        r"^saludos$",
        r"^buenas$",
    ],
    "consulta": [
        r"consultar",
        r"información",
        r"datos",
        r"necesito",
        r"saber",
        r"indicarme",
        r"gustaría",
        r"podría",
        r"puede",
        r"atención",
        r"token",
        r"número",
        r"asesor",
        r"hablar",
        r"opción",
        r"necesito\s+información",
        r"quiero\s+saber",
        r"gustaría\s+saber",
        r"podría\s+indicarme",
        r"dónde\s+está",
        r"cuánto\s+tarda",
        r"cuánto\s+cuesta",
        r"cuál\s+es",
        r"cómo\s+puedo",
        r"cómo\s+hacer",
        r"número\s+de",
        r"token\s+de",
        r"datos\s+de",
        r"consultar\s+mi",
        r"consultar\s+el",
        r"información\s+de",
        r"información\s+sobre",
        r"aclaración\s+de",
    ],
}


def normalize_text(text: str) -> str:
    """Lowercase and remove accents."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text


def compile_patterns() -> dict:
    """Compile patterns with accents removed to match normalized text."""
    compiled = {}
    for category, patterns in PATTERNS.items():
        compiled[category] = [re.compile(normalize_text(p), re.IGNORECASE) for p in patterns]
    return compiled


COMPILED_PATTERNS = compile_patterns()


def compute_scores(text: str) -> tuple:
    """Return (scores dict, matched_patterns list)."""
    scores = defaultdict(int)
    matched_patterns = []
    for category, patterns in COMPILED_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                scores[category] += 1
                matched_patterns.append(pattern.pattern)
    return dict(scores), matched_patterns


def classify_original(scores: dict) -> tuple:
    """Original non-hierarchical classification (max score wins)."""
    if not scores:
        return ("", "no_match")

    max_score = max(scores.values())
    winners = [cat for cat, score in scores.items() if score == max_score]

    if len(winners) > 1:
        return ("", "ambiguous")
    else:
        category = winners[0]
        if max_score == 1:
            confidence = "low"
        else:
            confidence = "high"
        return (category, confidence)


def classify_hierarchical(scores: dict, is_greeting: bool, raw_text: str = "") -> tuple:
    """
    Hierarchical classification with greeting/despedida support:
    1. despedida checked first (takes priority over other categories)
    2. Credit sub-categories checked (auto, personal, solicitud, linea)
    3. Other categories via max score
    4. Fallback to 'crédito' if credit patterns matched
    5. 'saludo' fallback if only greeting and nothing else matched
    """
    if scores.get("despedida", 0) > 0:
        confidence = "high" if scores.get("despedida", 0) >= 2 else "low"
        return ("despedida", confidence)

    sub_categories = ["credito_auto", "credito_linea", "credito_solicitud", "credito_personal"]

    for sub_cat in sub_categories:
        if scores.get(sub_cat, 0) > 0:
            score = scores[sub_cat]
            confidence = "high" if score >= 2 else "low"
            return (sub_cat, confidence)

    if scores.get("crédito", 0) > 0:
        score = scores["crédito"]
        confidence = "high" if score >= 2 else "medium"
        return ("crédito", confidence)

    if not scores:
        return ("", "no_match")

    max_score = max(scores.values())
    winners = [cat for cat, score in scores.items() if score == max_score]

    if len(winners) > 1:
        return ("", "ambiguous")

    category = winners[0]

    if is_greeting and category == "" and len(raw_text) < 15:
        return ("saludo", "high")

    confidence = "high" if max_score >= 2 else "low"
    return (category, confidence)


ITERATION_TAG = "v3_saludo_despedida"

def main(sample_fraction=1.0, seed=42):
    print("Loading dataset...")
    df = pl.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} rows")

    if sample_fraction < 1.0:
        sample_size = int(len(df) * sample_fraction)
        df = df.sample(n=sample_size, seed=seed)
        print(f"Sampled {len(df):,} rows ({sample_fraction*100:.0f}%)")

    results = []
    for idx, row in enumerate(df.iter_rows(named=True)):
        raw_text = row.get("input", "")
        conv_id = row.get("conv_id", "")

        normalized = normalize_text(raw_text)
        scores, matched = compute_scores(normalized)

        is_greeting = scores.get("saludo", 0) > 0
        is_goodbye = scores.get("despedida", 0) > 0

        cat_orig, conf_orig = classify_original(scores)
        cat_hier, conf_hier = classify_hierarchical(scores, is_greeting, raw_text)

        results.append({
            "conv_id": conv_id,
            "input": raw_text,
            "category_original": cat_orig,
            "confidence_original": conf_orig,
            "category_hierarchical": cat_hier,
            "confidence_hierarchical": conf_hier,
            "is_greeting": 1 if is_greeting else 0,
            "is_goodbye": 1 if is_goodbye else 0,
            "matched_patterns": ";".join(matched),
            "score_tarjeta": scores.get("tarjeta", 0),
            "score_transferencia": scores.get("transferencia", 0),
            "score_credito_auto": scores.get("credito_auto", 0),
            "score_credito_personal": scores.get("credito_personal", 0),
            "score_credito_solicitud": scores.get("credito_solicitud", 0),
            "score_credito_linea": scores.get("credito_linea", 0),
            "score_crédito": scores.get("crédito", 0),
            "score_saldo": scores.get("saldo", 0),
            "score_queja": scores.get("queja", 0),
            "score_consulta": scores.get("consulta", 0),
            "score_despedida": scores.get("despedida", 0),
            "score_saludo": scores.get("saludo", 0),
        })

        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1:,} / {len(df):,}...")

    result_df = pl.DataFrame(results)

    output_path_orig = os.path.join(OUTPUT_DIR, f"chat_classifications_{ITERATION_TAG}_original.csv")
    df_orig = result_df.select(["conv_id", "input", "category_original", "confidence_original", "is_greeting", "is_goodbye", "matched_patterns",
                                 "score_tarjeta", "score_transferencia", "score_crédito", "score_saldo", "score_queja", "score_consulta",
                                 "score_despedida", "score_saludo"])
    df_orig.write_csv(output_path_orig)
    print(f"Saved: {output_path_orig}")

    output_path_hier = os.path.join(OUTPUT_DIR, f"chat_classifications_{ITERATION_TAG}_hierarchical.csv")
    result_df.select(["conv_id", "input", "category_hierarchical", "confidence_hierarchical", "is_greeting", "is_goodbye", "matched_patterns",
                      "score_tarjeta", "score_transferencia", "score_credito_auto", "score_credito_personal",
                      "score_credito_solicitud", "score_credito_linea", "score_crédito", "score_saldo", "score_queja", "score_consulta",
                      "score_despedida", "score_saludo"]).write_csv(output_path_hier)
    print(f"Saved: {output_path_hier}")

    print_summary(df_orig, "ORIGINAL")
    print_summary(result_df, "HIERARCHICAL")


def print_summary(df: pl.DataFrame, mode: str = "") -> None:
    """Print classification summary with optional mode label."""
    label = f"=== {mode} Classification Summary ===" if mode else "=== Classification Summary ==="
    print(f"\n{label}")
    print(f"Total inputs: {len(df):,}")

    cat_col = "category_original" if "category_original" in df.columns else "category_hierarchical" if "category_hierarchical" in df.columns else "category"
    conf_col = "confidence_original" if "confidence_original" in df.columns else "confidence_hierarchical" if "confidence_hierarchical" in df.columns else "confidence"

    print("\nCategory distribution:")
    cat_counts = df.group_by(cat_col).len()
    cat_counts = cat_counts.sort("len", descending=True)
    for row in cat_counts.iter_rows(named=True):
        pct = row["len"] / len(df) * 100
        cat_label = row[cat_col] if row[cat_col] else "(unclassified)"
        print(f"  {str(cat_label):<20} {row['len']:>6,} ({pct:5.1f}%)")

    print("\nConfidence breakdown:")
    conf_counts = df.group_by(conf_col).len()
    for row in conf_counts.iter_rows(named=True):
        pct = row["len"] / len(df) * 100
        print(f"  {str(row[conf_col]):<15} {row['len']:>6,} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()