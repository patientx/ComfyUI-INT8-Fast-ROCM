import os
import urllib.request

_UPDATER_RAW_URL = "https://raw.githubusercontent.com/patientx-cfz/comfyui-rocm/main/comfyui-rocm-updater.bat"


def _heal_updater_script() -> None:
    """One-time (and ongoing) safety net: fixes the case where a user's local
    updater.bat is stuck out of sync with its own recorded hash (can happen
    when the updater.bat itself changes - robocopy excludes its own running
    filename, so an old copy can never overwrite itself). This runs from a
    custom node's __init__.py instead, which git-pulls normally and isn't
    subject to that same self-reference restriction, so it can always fix
    the updater.bat directly regardless of what state it's stuck in.
    """
    try:
        install_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        updater_path = os.path.join(install_dir, "comfyui-rocm-updater.bat")
        if not os.path.isdir(install_dir):
            return

        with urllib.request.urlopen(_UPDATER_RAW_URL, timeout=5) as resp:
            latest = resp.read()

        current = b""
        if os.path.exists(updater_path):
            with open(updater_path, "rb") as f:
                current = f.read()

        if latest and latest != current:
            with open(updater_path, "wb") as f:
                f.write(latest)
    except Exception:
        # Never let a failed self-heal check (no network, no permissions,
        # renamed repo, etc.) break ComfyUI startup.
        pass


_heal_updater_script()