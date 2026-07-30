"""Desktop integration: tray presence and a global hotkey.

Windows-only. Launch it with ``pythonw -m quainex.desktop.tray``.

Deliberately free of imports. Re-exporting ``TrayApplication`` here puts the
module into ``sys.modules`` *before* ``python -m quainex.desktop.tray`` executes
it, which makes ``runpy`` warn about unpredictable behaviour and run the module
body twice. Since running as a module is the entry point, a shorter import path is
not worth that.
"""
