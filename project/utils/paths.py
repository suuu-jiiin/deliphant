import os
from pathlib import Path

def find_project_root(start: Path) -> Path:
    """깃 루트(.git)나 pyproject/requirements가 있는 상위 폴더를 프로젝트 루트로 간주"""
    for p in [start] + list(start.parents):
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / "requirements.txt").exists():
            return p
    return start

# __file__이 없는 환경(노트북/코랩) 대비
try:
    _here = Path(__file__).resolve().parent
except NameError:
    _here = Path.cwd()

PROJECT_ROOT = find_project_root(_here)

if PROJECT_ROOT.name == "utils" and (PROJECT_ROOT.parent / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

# 기본 경로(프로젝트 내 상대경로)
DEFAULT_DATA_PATH = PROJECT_ROOT / "project" / "data" / "feature_importance.csv"
DEFAULT_FONT_PATH = PROJECT_ROOT / "project" / "assets" / "fonts" / "NanumSquareRoundL.ttf"

# 환경변수 우선 → 없으면 기본값
DATA_PATH = Path(os.getenv("DATA_PATH", str(DEFAULT_DATA_PATH)))
FONT_PATH = Path(os.getenv("FONT_PATH", str(DEFAULT_FONT_PATH)))

