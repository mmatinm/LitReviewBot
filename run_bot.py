import sys
from pathlib import Path

from streamlit.web import bootstrap


def _resolve_app_path() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")) / "app.py"
    return Path(__file__).resolve().parent / "app.py"


def main() -> int:
    app_path = _resolve_app_path()
    if not app_path.exists():
        print(f"[ERROR] Could not find app.py at: {app_path}")
        return 1

    # In PyInstaller onefile mode, Streamlit defaults to development mode because
    # its package path is extracted outside site-packages. Force production mode
    # so browser URL/port are correct for end users.
    flag_options = {
        "global.developmentMode": False,
        "server.port": 8501,
        "browser.serverPort": 8501,
    }
    bootstrap.load_config_options(flag_options)
    bootstrap.run(str(app_path), False, [], flag_options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
