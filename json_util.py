from typing import Dict, List, Any
import json

def _load_json_lines(filepath: str) -> List[Dict[str, Any]]:
    """JSONL 파일을 안전하게 로드한다."""
    lines: List[Dict[str, Any]] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data: Dict[str, Any] = json.loads(line)
                    lines.append(data)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return lines