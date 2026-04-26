# -*- mode: python ; coding: utf-8 -*-
import sys as _sys
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ('faster_whisper', 'ctranslate2', 'tokenizers', 'huggingface_hub',
            'soundcard', 'numpy', 'keyring', 'pip'):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Bundle the diarization runner so it can be extracted beside the EXE at runtime.
datas += [('audio2text_diarize.py', '.')]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + [
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
        # Application modules
        'config',
        'diarization',
        'ui',
        # keyring platform backends
        'keyring.backends',
        'keyring.backends.Windows',
        'keyring.backends.SecretService',
        'keyring.backends.macOS',
        'keyring.backends.fail',
        'keyring.backends.null',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'torchaudio', 'tensorflow'],
    noarchive=False,
)

pyz = PYZ(a.pure)

_is_mac = _sys.platform == 'darwin'

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='audio2text-macos' if _is_mac else 'audio2text-windows',
    debug=False,
    strip=_is_mac,
    upx=not _is_mac,
    console=False,
    icon=None,
)

if _is_mac:
    app = BUNDLE(
        exe,
        name='Audio2Text.app',
        bundle_identifier='com.vytvareniher.audio2text',
        info_plist={
            'NSMicrophoneUsageDescription': (
                'Audio2Text needs microphone access for live transcription.'
            ),
            'NSHighResolutionCapable': True,
            'LSBackgroundOnly': False,
        },
    )
