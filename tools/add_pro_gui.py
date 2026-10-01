#!/usr/bin/env python3
"""Window for tools/add_pro.py: paste a YouTube link (or pick a file), fill in
the player, press the button, check the preview.

    python tools/add_pro_gui.py

Uses Tkinter, which ships with Python. (Homebrew Python on a Mac needs
`brew install python-tk`; Debian/Ubuntu need `sudo apt install python3-tk`.)
Everything runs the same code as the command-line tool.
"""

import base64
import os
import queue
import re
import subprocess
import sys
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # pragma: no cover - depends on the Python build
    sys.exit("Tkinter is not available in this Python. On a Mac with Homebrew: brew install python-tk; "
             "on Debian/Ubuntu: sudo apt install python3-tk. Or use tools/add_pro.py from the command line.")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import add_pro  # noqa: E402

ROOT = add_pro.ROOT
SPEEDS = ["Real time", "2x slow motion", "4x slow motion", "8x slow motion", "16x slow motion"]
FF = ["Auto", "1 (every frame)", "2", "3", "4", "6", "8"]
BROWSERS = ["None", "chrome", "firefox", "safari", "edge", "brave", "chromium", "opera"]


def open_path(path):
    """Open a file or folder with the system's default app."""
    path = str(path)
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class QueueWriter:
    """File-like object that forwards writes to the GUI thread."""

    def __init__(self, q):
        self.q = q

    def write(self, s):
        if s:
            self.q.put(("log", s))
        return len(s)

    def flush(self):
        pass


class App:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.running = False
        self.result = None
        self.preview_img = None
        root.title("Swing Match: add a pro swing")
        root.minsize(980, 640)
        self._build()
        self._toggle_source()
        self._toggle_target()
        root.after(80, self._poll)

    # ------------------------------------------------------------------ layout

    def _build(self):
        pad = {"padx": 6, "pady": 3}
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        form = ttk.Frame(main)
        form.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        form.columnconfigure(1, weight=1)
        r = 0

        def section(text):
            nonlocal r
            ttk.Label(form, text=text, font=("TkDefaultFont", 11, "bold")).grid(row=r, column=0, columnspan=3, sticky="w", pady=(10 if r else 0, 2))
            r += 1

        def row(label, widget, hint=None):
            nonlocal r
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w", **pad)
            widget.grid(row=r, column=1, sticky="ew", **pad)
            if hint:
                ttk.Label(form, text=hint, foreground="#777").grid(row=r, column=2, sticky="w")
            r += 1

        # Source
        section("1. The swing")
        self.source = tk.StringVar(value="url")
        src = ttk.Frame(form)
        ttk.Radiobutton(src, text="YouTube link", variable=self.source, value="url", command=self._toggle_source).pack(side="left")
        ttk.Radiobutton(src, text="Video file", variable=self.source, value="file", command=self._toggle_source).pack(side="left", padx=8)
        src.grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        r += 1
        self.url = tk.StringVar()
        self.url_entry = ttk.Entry(form, textvariable=self.url, width=46)
        row("Link", self.url_entry)
        self.file = tk.StringVar()
        fbox = ttk.Frame(form)
        self.file_entry = ttk.Entry(fbox, textvariable=self.file, width=36)
        self.file_entry.pack(side="left", fill="x", expand=True)
        self.browse_btn = ttk.Button(fbox, text="Browse…", command=self._browse)
        self.browse_btn.pack(side="left", padx=(4, 0))
        row("File", fbox)
        self.start = tk.StringVar()
        self.end = tk.StringVar()
        times = ttk.Frame(form)
        ttk.Entry(times, textvariable=self.start, width=9).pack(side="left")
        ttk.Label(times, text=" to ").pack(side="left")
        ttk.Entry(times, textvariable=self.end, width=9).pack(side="left")
        row("Time range", times, "e.g. 1:02.5 to 1:06")

        # Player
        section("2. The player")
        self.name = tk.StringVar()
        row("Name *", ttk.Entry(form, textvariable=self.name))
        self.team = tk.StringVar()
        row("Team", ttk.Entry(form, textvariable=self.team))
        self.bats = tk.StringVar(value="R")
        row("Bats", ttk.Combobox(form, textvariable=self.bats, values=["R", "L", "S"], state="readonly", width=6))
        self.notes = tk.StringVar()
        row("Notes", ttk.Entry(form, textvariable=self.notes))

        # Footage
        section("3. The footage")
        self.speed = tk.StringVar(value=SPEEDS[0])
        row("Speed", ttk.Combobox(form, textvariable=self.speed, values=SPEEDS, width=18), "slow-mo replays are common")
        self.ff = tk.StringVar(value=FF[0])
        row("Fast-forward", ttk.Combobox(form, textvariable=self.ff, values=FF, state="readonly", width=18), "analyze every Nth frame")
        self.pitcher = tk.StringVar(value="Auto")
        row("Pitcher is on the", ttk.Combobox(form, textvariable=self.pitcher, values=["Auto", "left", "right"], state="readonly", width=18))
        self.use_target = tk.BooleanVar(value=False)
        tbox = ttk.Frame(form)
        ttk.Checkbutton(tbox, text="Several people in frame: hitter is at", variable=self.use_target, command=self._toggle_target).pack(anchor="w")
        self.target = tk.DoubleVar(value=0.5)
        self.target_scale = ttk.Scale(tbox, from_=0.0, to=1.0, variable=self.target, command=lambda _v: self._target_label())
        self.target_scale.pack(fill="x")
        self.target_lbl = ttk.Label(tbox, text="", foreground="#777")
        self.target_lbl.pack(anchor="w")
        row("Hitter", tbox)

        # Options
        section("4. Options")
        self.model = tk.StringVar(value="heavy")
        row("Pose model", ttk.Combobox(form, textvariable=self.model, values=["heavy", "full", "lite"], state="readonly", width=18), "heavy = most accurate")
        self.cookies = tk.StringVar(value="None")
        row("Browser cookies", ttk.Combobox(form, textvariable=self.cookies, values=BROWSERS, state="readonly", width=18), "if YouTube asks to sign in")
        self.dry = tk.BooleanVar(value=False)
        self.keep = tk.BooleanVar(value=False)
        obox = ttk.Frame(form)
        ttk.Checkbutton(obox, text="Preview only (don't change the database)", variable=self.dry).pack(anchor="w")
        ttk.Checkbutton(obox, text="Keep the downloaded video", variable=self.keep).pack(anchor="w")
        obox.grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        r += 1

        self.run_btn = ttk.Button(form, text="Analyze and add to database", command=self._run)
        self.run_btn.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(12, 4))
        r += 1
        self.progress = ttk.Progressbar(form, mode="indeterminate")
        self.progress.grid(row=r, column=0, columnspan=3, sticky="ew")
        r += 1
        self.status = ttk.Label(form, text="Ready.", wraplength=380)
        self.status.grid(row=r, column=0, columnspan=3, sticky="w", pady=4)

        # Right side: preview and log
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=3)
        right.rowconfigure(3, weight=2)
        ttk.Label(right, text="Preview of the six phase frames", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, sticky="w")
        self.preview = ttk.Label(right, text="The preview appears here after a run.", anchor="center", relief="groove")
        self.preview.grid(row=1, column=0, sticky="nsew", pady=(2, 6))
        self.preview.bind("<Configure>", lambda _e: self._show_preview())
        btns = ttk.Frame(right)
        btns.grid(row=2, column=0, sticky="w")
        self.open_prev_btn = ttk.Button(btns, text="Open preview", command=lambda: self.result and open_path(self.result["preview"]), state="disabled")
        self.open_prev_btn.pack(side="left")
        ttk.Button(btns, text="Open database folder", command=lambda: open_path(ROOT / "data" / "pros")).pack(side="left", padx=6)
        self.git_btn = ttk.Button(btns, text="Copy git commands", command=self._copy_git, state="disabled")
        self.git_btn.pack(side="left")
        log_frame = ttk.Frame(right)
        log_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=12, wrap="word", font=("TkFixedFont", 10), state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(log_frame, command=self.log.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log["yscrollcommand"] = sb.set

    # ------------------------------------------------------------------ form helpers

    def _toggle_source(self):
        is_url = self.source.get() == "url"
        self.url_entry.state(["!disabled"] if is_url else ["disabled"])
        self.file_entry.state(["disabled"] if is_url else ["!disabled"])
        self.browse_btn.state(["disabled"] if is_url else ["!disabled"])

    def _toggle_target(self):
        self.target_scale.state(["!disabled"] if self.use_target.get() else ["disabled"])
        self._target_label()

    def _target_label(self):
        if not self.use_target.get():
            self.target_lbl.config(text="(off: the most prominent person is used)")
            return
        v = self.target.get()
        where = "left" if v < 0.4 else "right" if v > 0.6 else "middle"
        self.target_lbl.config(text=f"{v:.2f} of the frame width ({where})")

    def _browse(self):
        path = filedialog.askopenfilename(title="Choose a swing video", filetypes=[("Video", "*.mp4 *.mov *.m4v *.webm *.mkv *.avi"), ("All files", "*.*")])
        if path:
            self.file.set(path)
            self.source.set("file")
            self._toggle_source()

    def build_argv(self):
        """Command-line arguments for add_pro.main from the form (raises ValueError)."""
        argv = []
        if self.source.get() == "url":
            url = self.url.get().strip()
            if not url:
                raise ValueError("Paste a YouTube link (or choose a video file).")
            argv.append(url)
        else:
            f = self.file.get().strip()
            if not f:
                raise ValueError("Choose a video file (or paste a YouTube link).")
            argv += ["--file", f]
        for flag, var in (("--start", self.start), ("--end", self.end)):
            v = var.get().strip()
            if v:
                add_pro.parse_time(v)  # validate
                argv += [flag, v]
        name = self.name.get().strip()
        if not name:
            raise ValueError("Enter the player's name.")
        argv += ["--name", name]
        for flag, var in (("--team", self.team), ("--notes", self.notes)):
            if var.get().strip():
                argv += [flag, var.get().strip()]
        argv += ["--bats", self.bats.get()]
        speed = re.match(r"\s*([\d.]+)", self.speed.get())
        if speed and self.speed.get() != SPEEDS[0]:
            argv += ["--speed", speed.group(1)]
        ff = re.match(r"\s*(\d+)", self.ff.get())
        if ff:
            argv += ["--ff", ff.group(1)]
        if self.pitcher.get() in ("left", "right"):
            argv += ["--pitcher", self.pitcher.get()]
        if self.use_target.get():
            argv += ["--target-x", f"{self.target.get():.3f}"]
        argv += ["--model", self.model.get()]
        if self.cookies.get() != "None":
            argv += ["--cookies-from-browser", self.cookies.get()]
        if self.dry.get():
            argv.append("--dry-run")
        if self.keep.get():
            argv.append("--keep-video")
        return argv

    # ------------------------------------------------------------------ running

    def _run(self):
        if self.running:
            return
        try:
            argv = self.build_argv()
        except ValueError as e:
            messagebox.showerror("Missing information", str(e))
            return
        self.running = True
        self.result = None
        self.run_btn.state(["disabled"])
        self.open_prev_btn.state(["disabled"])
        self.git_btn.state(["disabled"])
        self.progress.start(12)
        self.status.config(text="Working… (the first run downloads the pose model)")
        self._append("$ python tools/add_pro.py " + " ".join(f'"{a}"' if " " in a else a for a in argv) + "\n")
        threading.Thread(target=self._worker, args=(argv,), daemon=True).start()

    def _worker(self, argv):
        w = QueueWriter(self.q)
        try:
            with redirect_stdout(w), redirect_stderr(w):
                result = add_pro.main(argv)
            self.q.put(("done", result))
        except SystemExit as e:
            self.q.put(("error", str(e.code) if e.code not in (None, 0) else "Stopped."))
        except Exception as e:  # noqa: BLE001 - show anything to the user
            self.q.put(("log", traceback.format_exc()))
            self.q.put(("error", f"{type(e).__name__}: {e}"))

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._append(payload)
                elif kind == "done":
                    self._finish(payload, None)
                elif kind == "error":
                    self._finish(None, payload)
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def _append(self, text):
        """Append to the log; a carriage return overwrites the current line."""
        self.log.config(state="normal")
        for part in re.split(r"(\r|\n)", text):
            if part == "\r":
                self.log.delete("end-1c linestart", "end-1c")
            elif part:
                self.log.insert("end", part)
        self.log.see("end")
        self.log.config(state="disabled")

    def _finish(self, result, error):
        self.running = False
        self.progress.stop()
        self.progress["value"] = 0
        self.run_btn.state(["!disabled"])
        if error:
            self.status.config(text=f"Failed: {error}")
            self._append(f"\n{error}\n")
            return
        self.result = result
        self.open_prev_btn.state(["!disabled"])
        if result.get("entry"):
            self.git_btn.state(["!disabled"])
            self.status.config(text=f"Added {result['id']}. Check the preview, then commit data/pros (Copy git commands).")
        else:
            self.status.config(text="Preview ready. Nothing was written (preview only).")
        self._show_preview()

    def _show_preview(self):
        path = self.result and self.result.get("preview")
        if not path or not Path(path).exists():
            return
        try:
            import cv2

            img = cv2.imread(str(path))
            w = max(200, self.preview.winfo_width() - 8)
            h = max(150, self.preview.winfo_height() - 8)
            s = min(w / img.shape[1], h / img.shape[0], 1.0)
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            ok, png = cv2.imencode(".png", img)
            if ok:
                self.preview_img = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
                self.preview.config(image=self.preview_img, text="")
        except Exception as e:  # noqa: BLE001
            self.preview.config(text=f"Could not show the preview ({e}); use Open preview.")

    def _copy_git(self):
        name = self.name.get().strip()
        cmd = f'git add data/pros && git commit -m "Add {name} swing" && git push'
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        self.status.config(text=f"Copied: {cmd}")


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam" if sys.platform.startswith("linux") else ttk.Style().theme_use())
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
