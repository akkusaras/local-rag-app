import difflib
from pathlib import Path
from typing import TypedDict

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
EXCEL_PATH = ROOT_DIR / "MELİS LAZER FİYAT TARİFESİ 2026 firma.xlsx"
HEADER_ROW = 2

NAME_COL = 0
QUANTITY_COL = 2
WIDTH_COL = 3
HEIGHT_COL = 4
PRICE_COL = 6

_UNKNOWN_MATERIAL_MESSAGE = "Maalesef sadece listemizdeki malzemelerin kesimini yapıyoruz."
_MISSING_DATA_MESSAGE = (
    "Lütfen hesaplama yapabilmemiz için en, boy ve adet bilgilerini tam olarak giriniz."
)
_MISSING_AREA_MESSAGE = "Lütfen alan bazlı bu malzeme için en ve boy bilgilerini giriniz."


class FixedMaterial(TypedDict):
    type: str  # "fixed"
    price: float


class AreaMaterial(TypedDict):
    type: str  # "area"
    multiplier: float


_materials: dict[str, FixedMaterial | AreaMaterial] | None = None


def _normalize(text: str) -> str:
    """Lowercase a possibly-Turkish string without the dotted-İ combining-mark artifact."""
    return text.strip().replace("İ", "i").lower()


def _is_area_dimension(value: float) -> bool:
    """A real area dimension: numeric, not NaN, and not the fixed-price placeholder 1."""
    return not pd.isna(value) and value != 1


def _build_materials(excel_path: Path = EXCEL_PATH) -> dict[str, FixedMaterial | AreaMaterial]:
    """Read the price list and classify each material as fixed-price or area-based.

    Rows where width or height is NaN, empty, or 1 are unit-based / fixed-price
    items (e.g. Açacak, Organizer): price = total_price / quantity.

    All other rows are area-based (e.g. Pleksi, MDF):
    multiplier = total_price / (quantity * width * height)

    Rows with a missing material name, or a non-numeric/zero quantity or price,
    are skipped.
    """
    df = pd.read_excel(excel_path, sheet_name=0, header=HEADER_ROW)

    names = df.iloc[:, NAME_COL]
    quantities = pd.to_numeric(df.iloc[:, QUANTITY_COL], errors="coerce")
    widths = pd.to_numeric(df.iloc[:, WIDTH_COL], errors="coerce")
    heights = pd.to_numeric(df.iloc[:, HEIGHT_COL], errors="coerce")
    prices = pd.to_numeric(df.iloc[:, PRICE_COL], errors="coerce")

    materials: dict[str, FixedMaterial | AreaMaterial] = {}
    for name, quantity, width, height, price in zip(names, quantities, widths, heights, prices):
        if not isinstance(name, str) or not name.strip():
            continue
        if pd.isna(quantity) or quantity == 0 or pd.isna(price):
            continue

        if _is_area_dimension(width) and _is_area_dimension(height):
            denominator = quantity * width * height
            if denominator == 0:
                continue
            materials[_normalize(name)] = {"type": "area", "multiplier": price / denominator}
        else:
            materials[_normalize(name)] = {"type": "fixed", "price": price / quantity}

    return materials


def _get_materials() -> dict[str, FixedMaterial | AreaMaterial]:
    global _materials
    if _materials is None:
        _materials = _build_materials()
    return _materials


def _resolve_material(material: str, materials: dict[str, FixedMaterial | AreaMaterial]) -> str | None:
    key = _normalize(material)
    if key in materials:
        return key

    # Fall back to a loose substring match (e.g. "pleksi" -> "ahşap + pleksi").
    for candidate in materials:
        if key in candidate or candidate in key:
            return candidate

    # Fall back to fuzzy matching for typos (e.g. "pleksii" -> "pleksi").
    close_matches = difflib.get_close_matches(key, materials.keys(), n=1, cutoff=0.7)
    if close_matches:
        return close_matches[0]

    return None


def _require_positive_number(value: float | None) -> float:
    if value is None:
        raise ValueError(_MISSING_DATA_MESSAGE)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(_MISSING_DATA_MESSAGE) from None
    if number <= 0:
        raise ValueError(_MISSING_DATA_MESSAGE)
    return number


def _to_float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_price(material: str, width: float, height: float, quantity: int) -> float:
    """Calculate the total price for `material` at the given size and quantity."""
    quantity = _require_positive_number(quantity)

    if not material or not str(material).strip():
        raise ValueError(_MISSING_DATA_MESSAGE)

    materials = _get_materials()
    resolved_key = _resolve_material(material, materials)

    if resolved_key is None:
        raise ValueError(_UNKNOWN_MATERIAL_MESSAGE)

    entry = materials[resolved_key]

    if entry["type"] == "fixed":
        return quantity * entry["price"]

    width_value = _to_float_or_none(width)
    height_value = _to_float_or_none(height)
    if not width_value or not height_value:
        raise ValueError(_MISSING_AREA_MESSAGE)

    return entry["multiplier"] * width_value * height_value * quantity
