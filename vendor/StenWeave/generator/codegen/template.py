import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_PLACEHOLDER_RE = re.compile(r"<<<(\w+)>>>")


def replace_placeholders(
    input_file: str | Path, output_file: str | Path, replacements: Mapping[str, Any]
):
    input_path = Path(input_file)
    output_path = Path(output_file)

    template_content = input_path.read_text(encoding="utf-8")

    # Find all placeholders in the template.
    placeholders = set(_PLACEHOLDER_RE.findall(template_content))
    keys = set(replacements.keys())

    missing = placeholders - keys
    extra = keys - placeholders

    if missing:
        raise ValueError(f"Missing replacements for placeholders: {', '.join(sorted(missing))}")

    if extra:
        raise ValueError(f"Extra replacements provided: {', '.join(sorted(extra))}")

    def _sub(match: re.Match) -> str:
        name = match.group(1)
        return str(replacements[name])

    result = _PLACEHOLDER_RE.sub(_sub, template_content)
    output_path.write_text(result, encoding="utf-8")
