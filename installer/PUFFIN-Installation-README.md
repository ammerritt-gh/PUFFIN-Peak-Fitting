# PUFFIN Installation Guide (Windows)

PUFFIN is a desktop app for interactive 1D curve fitting (PySide6 + PyQtGraph + SciPy).
This installer sets up PUFFIN under your Windows user account with its own private Python —
no administrator rights, and nothing else on the machine is changed.

---

## Quick Start

1. **Download** [`WINDOWS-install-PUFFIN.bat`](https://github.com/ammerritt-gh/PUFFIN-Peak-Fitting/blob/v0.3.0-alpha/installer/WINDOWS-install-PUFFIN.bat)
   (open the link, then use the **Download raw file** button).
2. **Double-click** `WINDOWS-install-PUFFIN.bat`.
3. **Wait** a few minutes for it to finish.
4. **Launch** PUFFIN from the new **PUFFIN** shortcut on your desktop.

You only need Windows 10/11 and an internet connection. You do **not** need Python
pre-installed — the installer downloads a private copy.

---

## What the installer does

`WINDOWS-install-PUFFIN.bat` performs these steps automatically:

1. **Downloads uv** — a small, self-contained tool (pinned version, checksum-verified) to
   `%LOCALAPPDATA%\PUFFIN\uv`.
2. **Installs Python 3.12** privately via uv (does not touch any system Python).
3. **Downloads PUFFIN** from GitHub at the pinned release, into `%USERPROFILE%\PUFFIN`.
4. **Creates a virtual environment** (`.venv`) and installs the exact pinned dependencies
   from `requirements.lock.txt`.
5. **Creates a desktop shortcut** ("PUFFIN") pointing at the launcher menu.

### Installed components

| Component | Purpose |
|-----------|---------|
| uv | Python/venv manager (single binary) |
| Python 3.12 | Private runtime (provided by uv) |
| PySide6 | GUI framework |
| PyQtGraph | Interactive plotting |
| NumPy, SciPy, pandas, matplotlib | Numerics / data / export |

---

## Versioning

The installer pins a specific PUFFIN release via the `PUFFIN_VERSION` variable near the top
of the script. Release installers always pin a specific tag (e.g. `v0.3.0-alpha`).

To check which version is installed, open `%USERPROFILE%\PUFFIN\INSTALL_INFO.txt`.

To **update to a newer release**, download and run that release's installer (it reinstalls
into the same location). To **repair the current release**, use **Update / repair** in the
launcher (or run `update-puffin.bat`).

---

## Using PUFFIN

After installation, use the **PUFFIN** desktop shortcut, which opens a small menu:

| Option | Description |
|--------|-------------|
| **[1] Run PUFFIN** | Start the application |
| **[2] Update / repair** | Re-fetch the pinned release and reinstall dependencies |
| **[3] Open PUFFIN folder** | Browse the install directory |
| **[4] Open PUFFIN shell** | Command prompt with the venv activated |
| **[5] Exit** | Close the launcher |

### Scripts in `%USERPROFILE%\PUFFIN`

| Script | Purpose |
|--------|---------|
| `PUFFIN-Launcher.bat` | Menu launcher (recommended entry point) |
| `run-puffin.bat` | Launch PUFFIN directly |
| `update-puffin.bat` | Re-fetch the pinned release and reinstall dependencies |

---

## Troubleshooting

### "Failed to download uv" / "Failed to download PUFFIN source"
Check your internet connection and retry. If you are behind a corporate proxy, configure
Windows proxy settings first.

### "uv checksum mismatch"
The download was corrupted or tampered with. Re-run the installer. If it persists, the pinned
`UV_VERSION` may have been changed without updating `UV_SHA256` — see the design document.

### Import errors on launch (e.g. `No module named 'PySide6'`)
Use **[2] Update / repair** from the launcher (or run `update-puffin.bat`), which reinstalls
the pinned dependencies.

### The app does not start and the window closes immediately
Run `run-puffin.bat` directly from a command prompt so the error stays on screen, and report
the message at the issues link below.

---

## Uninstalling

Run [`WINDOWS-uninstall-PUFFIN.bat`](https://github.com/ammerritt-gh/PUFFIN-Peak-Fitting/blob/v0.3.0-alpha/installer/WINDOWS-uninstall-PUFFIN.bat).
It removes `%USERPROFILE%\PUFFIN`, the bundled uv under `%LOCALAPPDATA%\PUFFIN`, and the
desktop shortcut. It does not remove a system-wide uv or anything else.

---

## Installation paths

| Item | Location |
|------|----------|
| uv binary | `%LOCALAPPDATA%\PUFFIN\uv\uv.exe` |
| uv-managed Python | `%LOCALAPPDATA%\uv\python\...` (shared uv data dir) |
| PUFFIN code | `%USERPROFILE%\PUFFIN\` |
| Virtual environment | `%USERPROFILE%\PUFFIN\.venv\` |
| Desktop shortcut | `%USERPROFILE%\Desktop\PUFFIN.lnk` |

---

## Getting help

- **PUFFIN issues**: https://github.com/ammerritt-gh/PUFFIN-Peak-Fitting/issues
- **uv documentation**: https://docs.astral.sh/uv/
