def _normalize_quantity_words(text: str) -> str:
    """Expand colloquial Turkish quantity words (e.g. 'bi' -> 'bir') so the
    extraction model doesn't have to guess slang."""
    return re.sub(r"\bbi\b", "bir", text, flags=re.IGNORECASE)

import json
import re

from src.pricing import calculate_price
from src.rag_pipeline import answer_query, get_chat_client

PRICING_KEYWORDS = (
    "fiyat",
    "hesapla",
    "kestirmek",
    "ne kadar",
    "istiyorum",
    "lazım",
    "kesim",
    "tutar",
)


GREETING_KEYWORDS = (
    "selam", "merhaba", "naber", "günaydın", "iyi günler",
    "iyi akşamlar", "nasılsın", "hey", "selamlar",
)

GREETING_RESPONSE = (
    "Selam! 👋 Ben Melis Lazer'in yapay zeka asistanıyım. "
    "Dokümanlarımızla ilgili sorularını yanıtlayabilir, ya da malzeme, "
    "en, boy ve adet bilgisi vererek fiyat teklifi hesaplayabilirim.\n\n"
    "Örnek: \"50 adet 10x10 pleksi ne kadar tutar?\""
)


def _is_greeting(query: str) -> bool:
    lowered = query.strip().replace("İ", "i").lower()
    return any(lowered == g or lowered.startswith(g + " ") or lowered.startswith(g + "!")
               for g in GREETING_KEYWORDS)
MATERIAL_KEYWORDS = ("pleksi", "mdf", "dekota", "ahşap", "epoksi")
DIMENSION_PATTERN = re.compile(r"\d+\s*x\s*\d+", re.IGNORECASE)

EXTRACTION_SYSTEM_PROMPT = (
    "You are a data extraction assistant for a laser-cutting price calculator. "
    "The user's message may describe one or several order items, in any word "
    "order (material, dimensions, and quantity can appear in any sequence). "
    "Extract the material name, width, height, and quantity for EVERY item "
    "mentioned. "
    "Respond with ONLY a valid JSON object in exactly this shape, with no "
    "extra text, no markdown, and no explanation: "
    '{"items": [{"material": "acacak", "width": 0, "height": 0, "quantity": 10}, '
    '{"material": "organizer", "width": 0, "height": 0, "quantity": 20}]}. '
    "The \"items\" list must contain one object per distinct item, even if "
    "there is only a single item. "
    "Widths and heights are numbers in centimeters. "
    "STRICT: If the user does not explicitly specify the width, height, or "
    "quantity for an item, you MUST set that missing field's value to 0 in "
    "the JSON. NEVER guess, assume, or invent default numbers (like 10x10). "
    "If the data is missing from the text, return 0 for that field. "
    "STRICT: Material names can be compound, made of two or more words, "
    "joined either by '+'/'-' (e.g. \"dekota + uv\", \"ahşap + pleksi\") or "
    "simply written next to each other with no separator (e.g. \"açacak "
    "anahtarlık\", where \"anahtarlık\" is a modifier narrowing which "
    "\"açacak\" variant is meant, not a separate item). Always capture the "
    "FULL material phrase exactly as written, including every word and any "
    "separator — NEVER truncate a compound name down to just its first "
    "word. A longer, more specific compound phrase always takes priority "
    "over treating only its first word as the material name.\n\n"
    "Examples:\n"
    'User: "10x10 pleksi 50 adet ne kadar tutar?"\n'
    'JSON: {"items": [{"material": "pleksi", "width": 10, "height": 10, "quantity": 50}]}\n'
    'User: "pleksi malzemeden 10x10 ölçüsünde 50 adet istiyorum"\n'
    'JSON: {"items": [{"material": "pleksi", "width": 10, "height": 10, "quantity": 50}]}\n'
    'User: "10 adet açacak ve 20 adet organizer istiyorum"\n'
    'JSON: {"items": [{"material": "açacak", "width": 0, "height": 0, "quantity": 10}, '
    '{"material": "organizer", "width": 0, "height": 0, "quantity": 20}]}\n'
    'User: "Açacak anahtarlık istiyorum, 50 adet ne kadar tutar?"\n'
    'JSON: {"items": [{"material": "açacak anahtarlık", "width": 0, "height": 0, "quantity": 50}]}\n'
    'User: "bir adet organizer istiyorum"\n'
    'JSON: {"items": [{"material": "organizer", "width": 0, "height": 0, "quantity": 1}]}\n'
    'User: "Dekota + UV istiyorum"\n'
    'JSON: {"items": [{"material": "dekota + uv", "width": 0, "height": 0, "quantity": 0}]}'
)

def _is_pricing_query(query: str) -> bool:
    # Mirror src.pricing._normalize's dotted-İ handling so Turkish keyword
    # matching doesn't break on the combining-mark artifact from str.lower().
    lowered = query.strip().replace("İ", "i").lower()

    if any(keyword in lowered for keyword in PRICING_KEYWORDS):
        return True
    if any(material in lowered for material in MATERIAL_KEYWORDS):
        return True
    if DIMENSION_PATTERN.search(lowered):
        return True

    return False


def _extract_json_text(raw: str) -> str:
    """Pull a bare JSON object out of a raw LLM completion.

    Phi-3.5-mini sometimes hallucinates markdown code fences (```json ... ```)
    or wraps the JSON in conversational text despite instructions, so this
    strips fences first and then locates the {...} object with a regex.
    """
    text = raw.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not brace_match:
        raise ValueError(f"Model çıktısından JSON nesnesi çıkarılamadı: {raw!r}")

    return brace_match.group(0)


def _extract_pricing_request(query: str) -> dict:
    normalized_query = _normalize_quantity_words(query)
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": normalized_query},
    ]
    completion = get_chat_client().complete_chat(messages)
    raw = completion.choices[0].message.content.strip()

    json_text = _extract_json_text(raw)
    return json.loads(json_text)


def _calculate_line_item(item: dict) -> tuple[str, int, float]:
    """Validate and price a single order item, returning (material, quantity, total)."""
    material = item.get("material")
    width = item.get("width")
    height = item.get("height")
    quantity = item.get("quantity", 1)

    try:
        width = float(width) if width is not None else None
        height = float(height) if height is not None else None
        quantity = int(quantity) if quantity is not None else None
    except (TypeError, ValueError):
        raise ValueError(
            "Lütfen hesaplama yapabilmemiz için en, boy ve adet "
            "bilgilerini tam olarak giriniz."
        )

    resolved_material, total_price = calculate_price(
    str(material) if material else material, width, height, quantity
    )
    assert quantity is not None
    return resolved_material, quantity, total_price

def _format_pricing_response(query: str) -> str:
    try:
        data = _extract_pricing_request(query)
    except json.JSONDecodeError:
        return (
            "Üzgünüm, isteğinizi anlayamadım. Lütfen malzeme, en, boy ve "
            "adet bilgilerini tekrar belirtir misiniz?"
        )

    items = data.get("items")
    if not isinstance(items, list) or not items:
        return (
            "Üzgünüm, isteğinizi anlayamadım. Lütfen malzeme, en, boy ve "
            "adet bilgilerini tekrar belirtir misiniz?"
        )

    lines = []
    grand_total = 0.0
    any_success = False

    for item in items:
        if not isinstance(item, dict):
            lines.append("⚠️ Kalem okunamadı: beklenmeyen veri biçimi.")
            continue

        try:
            material, quantity, total_price = _calculate_line_item(item)
        except ValueError as e:
            label = item.get("material") or "Bilinmeyen kalem"
            print(f"Kalem hatası ({label}): {e}")
            lines.append(f"⚠️ {label}: {e}")
            continue

        any_success = True
        grand_total += total_price
        lines.append(f"- {quantity} adet {material.title()} -> {total_price:,.2f} TL")

    if not any_success:
        return "\n".join(lines)

    return (
        "🛒 **Sipariş Özeti:**\n\n"
        + "\n\n".join(lines)
        + f"\n\n**Genel Toplam: {grand_total:,.2f} TL**"
    )


def process_query(question: str) -> str:
    """Hybrid dispatch: route `question` to greeting, pricing, or the RAG pipeline."""
    if _is_greeting(question):
        return GREETING_RESPONSE

    if _is_pricing_query(question):
        return _format_pricing_response(question)

    return answer_query(question)


def main() -> None:
    print("Local RAG CLI — ask a question about the ingested documents.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            answer = process_query(question)
        except Exception as e:
            print(f"Error: {e}\n")
            continue

        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
