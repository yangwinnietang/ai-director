from __future__ import annotations


# Map user-facing Chinese target words to YOLO class names.
# Extend this with project-specific trained classes, e.g. "male_lead", "female_lead".
LABEL_ALIASES: dict[str, list[str]] = {
    "person": ["person"],
    "man": ["person"],
    "woman": ["person"],
    "male": ["person"],
    "female": ["person"],
    "food": ["bowl", "cup", "dining table", "cake", "pizza", "sandwich", "banana", "apple", "orange"],
    "rice": ["bowl"],
    "meal": ["bowl", "dining table"],
    "男主": ["person"],
    "男的": ["person"],
    "男人": ["person"],
    "女主": ["person"],
    "女的": ["person"],
    "女人": ["person"],
    "饭": ["bowl", "dining table"],
    "菜": ["bowl", "dining table"],
    "食物": ["bowl", "dining table"],
}


def resolve_target_classes(target: str) -> list[str]:
    """Return detector class names that can satisfy a target phrase."""
    target = target.strip()
    if not target:
        return []
    return LABEL_ALIASES.get(target, [target])
