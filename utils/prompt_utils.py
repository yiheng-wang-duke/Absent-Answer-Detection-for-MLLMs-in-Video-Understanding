import re


OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def format_mcq_prompt(question: str, options: list[str], instruction: str = None) -> str:
    """Build a labeled MCQ prompt string.

    Args:
        question: The question text.
        options: List of option strings (raw text, no letter labels).
        instruction: Optional instruction appended before the answer line.

    Returns:
        Formatted prompt like:
            Question: ...
            Options:
            A. option1
            B. option2
            ...
            {instruction}
            Answer with the option's letter from the given choices directly.
    """
    lines = [f"Question: {question}", "Options:"]
    for i, opt in enumerate(options):
        lines.append(f"{OPTION_LETTERS[i]}. {opt}")

    if instruction:
        lines.append(instruction)
    else:
        lines.append("Answer with the option's letter from the given choices directly.")

    return "\n".join(lines)


def strip_option_labels(options: list[str]) -> list[str]:
    """Remove letter labels like 'A. ' or 'A) ' from options.

    E.g. ['A. Apples', 'B. Bananas'] -> ['Apples', 'Bananas']
    """
    stripped = []
    for opt in options:
        # Match patterns like "A. ", "A) ", "A: ", "(A) "
        m = re.match(r"^\(?[A-Za-z]\)?[\.\)\:\s]+\s*", opt)
        if m:
            stripped.append(opt[m.end():])
        else:
            stripped.append(opt)
    return stripped


def remove_option_by_index(options: list[str], answer_index: int) -> tuple[list[str], dict]:
    """Remove the option at answer_index and return relabeled options.

    Args:
        options: List of raw option strings (no labels).
        answer_index: 0-based index of the option to remove.

    Returns:
        (new_options, label_mapping) where label_mapping maps new letter -> old letter.
    """
    new_options = [opt for i, opt in enumerate(options) if i != answer_index]
    label_mapping = {}
    old_idx = 0
    new_idx = 0
    for i in range(len(options)):
        if i == answer_index:
            old_idx += 1
            continue
        label_mapping[OPTION_LETTERS[new_idx]] = OPTION_LETTERS[old_idx]
        old_idx += 1
        new_idx += 1
    return new_options, label_mapping


def add_none_option(options: list[str]) -> list[str]:
    """Append 'None of the above' as the last option."""
    return options + ["None of the above"]
