"""Tiny INI parser matching the subset used by Vangers `world.ini` files.

Port of `src/ini.ts` in the reference TypeScript viewer, which in turn mirrors
the `XBuffer`/`analyzeINI` handling in `src/terra/vmap.cpp`.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

_SECTION_RE = re.compile(r"^\[(.+?)\]$")


class Ini:
    """Ordered ``section -> {key: value}`` mapping preserving file order."""

    def __init__(self) -> None:
        self._sections: dict[str, dict[str, str]] = {"": {}}
        self._order: list[str] = [""]

    # -- reading ------------------------------------------------------------
    @classmethod
    def parse(cls, text: str) -> "Ini":
        ini = cls()
        current = ini._sections[""]
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line[0] in ";#/":
                continue
            m = _SECTION_RE.match(line)
            if m:
                name = m.group(1).strip()
                if name not in ini._sections:
                    ini._sections[name] = {}
                    ini._order.append(name)
                current = ini._sections[name]
                continue
            eq = line.find("=")
            if eq < 0:
                continue
            current[line[:eq].strip()] = line[eq + 1 :].strip()
        return ini

    def get(self, section: str, key: str, default: Optional[str] = None) -> Optional[str]:
        return self._sections.get(section, {}).get(key, default)

    def get_int(self, section: str, key: str, default: int = 0) -> int:
        v = self.get(section, key)
        if not v:
            return default
        try:
            return int(v, 10)
        except ValueError:
            return default

    def get_int_list(self, section: str, key: str, count: int, default: int = 0) -> list[int]:
        return parse_int_list(self.get(section, key), count, default)

    def has_section(self, section: str) -> bool:
        return section in self._sections

    @property
    def sections(self) -> list[str]:
        return list(self._order)

    def items(self, section: str) -> Iterable[tuple[str, str]]:
        return list(self._sections.get(section, {}).items())

    # -- writing ------------------------------------------------------------
    def set(self, section: str, key: str, value) -> None:
        if section not in self._sections:
            self._sections[section] = {}
            self._order.append(section)
        self._sections[section][key] = value if isinstance(value, str) else str(value)

    def copy_from(self, other: "Ini", section: str) -> None:
        for k, v in other.items(section):
            self.set(section, k, v)

    def dumps(self) -> str:
        out: list[str] = []
        for section in self._order:
            body = self._sections[section]
            if section:
                out.append(f"[{section}]")
            for k, v in body.items():
                out.append(f"{k}={v}")
            out.append("")
        return "\n".join(out)


def parse_int_list(value: Optional[str], count: int, default: int = 0) -> list[int]:
    """Parses a whitespace/comma separated list of ints (``XBuffer >=``)."""
    out = [default] * count
    if not value:
        return out
    parts = [p for p in re.split(r"[\s,]+", value) if p]
    for i in range(min(count, len(parts))):
        try:
            out[i] = int(parts[i], 10)
        except ValueError:
            out[i] = default
    return out


def format_int_list(values: Iterable[int]) -> str:
    return " ".join(str(v) for v in values)
