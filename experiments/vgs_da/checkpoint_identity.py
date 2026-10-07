"""Checkpoint dependencies are identified by content, independent of mount paths."""
import re

REQUIRED = {"source_cfg", "source_checkpoint", "rpn_checkpoint", "text_embeddings",
            "source_search_checkpoint"}

def fingerprints(sources):
    if not isinstance(sources, dict) or set(sources) != REQUIRED:
        raise ValueError("Incomplete checkpoint source identity")
    result = {}
    for key, value in sources.items():
        if key == "source_search_checkpoint" and value is None:
            result[key] = None
            continue
        digest = value.get("sha256") if isinstance(value, dict) else None
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Invalid source SHA256: " + key)
        result[key] = digest
    return result

def same_sources(saved, current):
    return fingerprints(saved) == fingerprints(current)
