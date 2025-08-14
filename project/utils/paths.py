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
DEFAULT_FONT_PATH = PROJECT_ROOT / "project" / "assets" / "fonts" / "NanumSquareRoundL.ttf"
DEFAULT_DATA_PATH = PROJECT_ROOT / "project" / "data" / "feature_importance.csv"
DEFAULT_PREPROCESSED_PATH = PROJECT_ROOT / "project" /  "data" / "preprocessed_data.csv"
DEFAULT_FINAL_PATH = PROJECT_ROOT / "project" /  "data" / "final_merged_df.csv"
DEFAULT_PROB_PATH = PROJECT_ROOT / "project" /  "data" / "prob_distribution.csv"

# 환경변수 우선 → 없으면 기본값
FONT_PATH = Path(os.getenv("FONT_PATH", str(DEFAULT_FONT_PATH)))
DATA_PATH = Path(os.getenv("DATA_PATH", str(DEFAULT_DATA_PATH)))
PREPROCESSED_PATH = Path(os.getenv("PREPROCESSED_PATH", str(DEFAULT_PREPROCESSED_PATH)))
LOCAL_CSV_PATH = Path(os.getenv("LOCAL_CSV_PATH", str(DEFAULT_FINAL_PATH)))
PROB_PATH = Path(os.getenv("PROB_PATH", str(DEFAULT_PROB_PATH)))
