# PUFFIN Windows Installer / Uninstaller Design Document

_Last updated: 2026-06-17_

This document records the intended design and constraints for the PUFFIN Windows user
installer and uninstaller. PUFFIN is a **pure-Python** application: all dependencies
(PySide6, pyqtgraph, numpy, scipy, pandas, matplotlib) are on PyPI with prebuilt Windows
wheels. There is no conda package, no C compiler requirement, and no system service.

The installer target is a normal Windows user machine, not a developer checkout. It must work
for users who do not already have Python installed and should not need admin rights.

This design is modeled on the sibling TAVI installer but is deliberately much simpler:
TAVI uses **micromamba** because McStas is a conda package needing a C compiler. PUFFIN needs
neither, so it uses **uv** — a single static binary that provisions its own Python and a
standard `.venv`, installing from one pinned lock file.

---

## 1. Scope

### Installer responsibilities (`WINDOWS-install-PUFFIN.bat`)

1. Explain what it will do and ask for confirmation.
2. Download a pinned, checksum-verified copy of `uv`.
3. Install a private Python (via uv) — never touch system Python or PATH.
4. Download the pinned PUFFIN release source (GitHub tag zip).
5. Create a `.venv` and install the exact pinned dependencies from `requirements.lock.txt`.
6. Create a desktop shortcut to the launcher menu.
7. Fail early if a required file is missing after download.

### Uninstaller responsibilities (`WINDOWS-uninstall-PUFFIN.bat`)

1. Ask for confirmation.
2. Remove `%USERPROFILE%\PUFFIN`.
3. Remove the bundled uv at `%LOCALAPPDATA%\PUFFIN\uv`.
4. Remove the desktop shortcut.
5. Never remove a system-wide uv, unrelated tools, or the shared `%LOCALAPPDATA%\uv` data dir.

---

## 2. Non-goals

The installer does not:

- Install or modify a system-wide Python or change PATH.
- Require `git` on the target machine (source is fetched as a release zip).
- Require administrator privileges.
- Bundle a C compiler, conda, micromamba, or McStas (none are needed).

---

## 3. Key paths and names

```bat
set "PUFFIN_VERSION=v0.3.0-alpha"
set "PYTHON_VERSION=3.12"
set "UV_VERSION=0.11.18"
set "UV_SHA256=<sha256 of uv-x86_64-pc-windows-msvc.zip>"
set "REPO_SLUG=ammerritt-gh/PUFFIN-Peak-Fitting"

set "INSTALL_DIR=%USERPROFILE%\PUFFIN"
set "UV_HOME=%LOCALAPPDATA%\PUFFIN\uv"
set "UV_EXE=%UV_HOME%\uv.exe"
set "SHORTCUT=%USERPROFILE%\Desktop\PUFFIN.lnk"
```

Developer vs. user install:

- The **developer** runs `run-puffin-dev.bat` from a Git checkout; the `.venv` lives in the
  checkout. uv may come from PATH.
- The **user installer** installs to `%USERPROFILE%\PUFFIN` with a private uv under
  `%LOCALAPPDATA%\PUFFIN`.

The uninstaller affects only the user install, never a developer checkout.

---

## 4. Pinning

Three things are pinned for reproducibility:

1. **PUFFIN release** — `PUFFIN_VERSION` (a GitHub tag). The installer downloads that tag's
   source zip. The tag **must** point at a commit that contains `requirements.lock.txt` and
   the launcher `.bat` files.
2. **uv** — `UV_VERSION` + `UV_SHA256`. The installer verifies the downloaded zip's SHA256
   before use. **When bumping `UV_VERSION`, update `UV_SHA256`** (from the release's
   `uv-x86_64-pc-windows-msvc.zip.sha256`).
3. **Python dependencies** — a single fully-pinned `requirements.lock.txt`, generated with:
   ```
   uv pip compile requirements.txt -o requirements.lock.txt --python-version 3.12
   ```
   `requirements.txt` (loose ranges) stays the human-edited source of truth; the lock is the
   committed reproducible artifact the venv is built from.

---

## 5. Batch scripting constraints

Conservative Windows batch style (carried over from TAVI's hard-won lessons):

- `@echo off` + `setlocal DisableDelayedExpansion`.
- Quote every path (handles `C:\Program Files (x86)`, spaces, parentheses).
- **Never set and use a variable inside the same parenthesized block** (with delayed
  expansion off, `%VAR%` in a block expands at parse time, before the `set` runs). Use
  sequential lines or `goto` labels instead — see the existing-directory handling in Step 3.
- **Do not generate launchers with large `echo`/redirect blocks.** PUFFIN avoids this entirely:
  the launchers (`run-puffin.bat`, `update-puffin.bat`, `PUFFIN-Launcher.bat`) are committed
  files shipped inside the source zip and use `%~dp0`, so the installer does not generate them.
  The only generated artifact is the `.lnk` shortcut (via PowerShell `WScript.Shell`).

---

## 6. Existing-directory handling

Step 3 must handle a pre-existing `%INSTALL_DIR%`:

- Contains `main.py` -> treat as an existing PUFFIN install; copy new files over it,
  **excluding `.venv`** (`robocopy ... /XD .venv`) so the environment is preserved.
- Exists but no `main.py` -> move it aside to a timestamped backup; never blind-delete.
- Does not exist -> create it.

The GitHub zip extracts to a single top-level folder named with the leading `v` stripped
(e.g. `PUFFIN-Peak-Fitting-0.3.0-alpha`). The installer **detects** this folder with a
`for /d` loop rather than hardcoding the name.

---

## 7. Fail-early validation

After download, verify before proceeding:

```bat
if not exist "%INSTALL_DIR%\main.py" ( ... fail ... )
if not exist "%INSTALL_DIR%\requirements.lock.txt" ( ... fail ... )
```

The second check catches the case where `PUFFIN_VERSION` points at a tag that predates the
pinned installer files.

---

## 8. Launchers

| File | Role | Notes |
|------|------|-------|
| `run-puffin-dev.bat` | Developer launch from a checkout | Builds `.venv` if missing; uv from PATH or bundled |
| `run-puffin.bat` | Installed-copy launch | Runs `.venv\Scripts\python.exe main.py` |
| `update-puffin.bat` | Repair/refresh current pinned release | Re-fetch tag zip (preserve `.venv`) + `uv pip sync` |
| `PUFFIN-Launcher.bat` | Menu (shortcut target) | Run / Update / Open folder / Open shell / Exit |

All use `cd /d "%~dp0"` so they are location-independent.

---

## 9. Safe uninstaller

Removes only: `%USERPROFILE%\PUFFIN`, `%LOCALAPPDATA%\PUFFIN\uv`, and the desktop shortcut.
If `%INSTALL_DIR%` lacks `main.py`, it warns and asks before deleting. It never removes a
system-wide uv or the shared `%LOCALAPPDATA%\uv` data dir (which may hold uv-managed Pythons
used by other uv tools).

---

## 10. Pre-release checklist

- [ ] `requirements.lock.txt` regenerated and committed.
- [ ] `PUFFIN_VERSION` set to the release tag; tag pushed and points at a commit that
      includes the lock file and launchers.
- [ ] `UV_VERSION` + `UV_SHA256` match a real uv release asset.
- [ ] Installer starts with explanation + confirmation.
- [ ] Installer uses `setlocal DisableDelayedExpansion` and quotes all paths.
- [ ] Installer verifies `main.py` and `requirements.lock.txt` after download.
- [ ] Desktop shortcut targets `PUFFIN-Launcher.bat`.
- [ ] Uninstaller scope is limited to the user install + bundled uv + shortcut.
- [ ] Install, run, update, and uninstall tested from a clean profile (ideally a VM with no
      Python).

---

## 11. Line endings

`.bat` files are marked `text eol=crlf` in `.gitattributes` so they check out with Windows
line endings. The scripts are also written to be tolerant of LF-only endings.
