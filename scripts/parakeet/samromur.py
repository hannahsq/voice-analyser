"""Samrómur Children 21.09 (OpenSLR 117, CC BY 4.0; children aged 4-17): utterances by split and age, from
metadata.tsv. Speaker ids stay the corpus's opaque numbers; nothing else about a speaker is used."""
import csv
from pathlib import Path

ROOT = Path(r"C:\github\datasets\samromur_children\samromur_children_21.09")


def age_of(s: str) -> float:
    """'10' -> 10; a band such as '13-17' -> its middle."""
    a = [float(x) for x in s.split("-")]
    return sum(a) / len(a)


def speakers(split: str, keep_age=lambda a: True) -> dict:
    """{speaker: [paths]} of valid utterances in `split` ('train' / 'dev' / 'test') whose age passes keep_age."""
    out = {}
    with open(ROOT / "metadata.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["status"] == split and keep_age(age_of(r["age"])):
                out.setdefault(r["speaker_id"], []).append(str(ROOT / f"data_{split}" / r["speaker_id"] / r["filename"]))
    return out
