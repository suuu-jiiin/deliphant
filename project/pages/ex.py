from pathlib import Path
import streamlit as st
import re

def _safe_key(s: str) -> str:
    # 공백/하이픈/언더스코어 통일, 소문자, 영숫자만 남김
    s = str(s).strip().lower()
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s.strip("_")

def find_icon_path(icon_dir: Path, subdir: str, key: str,
                   exts=(".png", ".jpg", ".jpeg", ".webp")) -> Path | None:
    """
    icon_dir/subdir 아래에서 key 에 해당하는 파일을 찾는다.
    1) <safe_key>.<ext> 우선
    2) 글롭으로 유사 항목 탐색
    """
    base = icon_dir / subdir
    if not base.exists():
        st.error(f"[ICON] 디렉토리 없음: {base}")
        return None

    skey = _safe_key(key)

    # 1) 정확 매칭
    for ext in exts:
        cand = base / f"{skey}{ext}"
        if cand.exists():
            return cand

    # 2) 유사 매칭
    candidates = []
    for p in base.glob("*.*"):
        stem = _safe_key(p.stem)
        if stem == skey:
            return p
        if skey in stem:
            candidates.append(p)

    return candidates[0] if candidates else None

def debug_icon_listing(icon_dir: Path, subdir: str, limit: int = 30):
    base = icon_dir / subdir
    st.write(f"[ICON] 탐색 경로: {base} (exists={base.exists()})")
    if base.exists():
        names = [p.name for p in list(base.glob("*.*"))[:limit]]
        st.write(f"[ICON] 샘플 파일({len(names)}개):", names)
