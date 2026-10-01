#!/usr/bin/env python3
"""Window for tools/add_pro.py: paste a YouTube link (or pick a file), fill in
the player, press Analyze, check and fix the six swing beats frame by frame,
then add the swing to the database.

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
from swingdb.analysis import PHASE_KEYS  # noqa: E402
from swingdb.entry import slugify, write_entry  # noqa: E402

ROOT = add_pro.ROOT
BROWSERS = ["None", "chrome", "firefox", "safari", "edge", "brave", "chromium", "opera"]
LABELS = add_pro.LABELS
SHORT = {"stance": "S", "load": "L", "footPlant": "FP", "contact": "C", "extension": "E", "finish": "F"}
COLORS = {"stance": "#64748b", "load": "#2563eb", "footPlant": "#059669", "contact": "#dc2626",
          "extension": "#d97706", "finish": "#7c3aed"}
THUMB_HEIGHT = 540  # analyzed frames are kept at this height for the beat review
TEXT_WIDGETS = (tk.Entry, tk.Text, tk.Spinbox, ttk.Entry, ttk.Combobox, ttk.Spinbox)


def open_path(path):
    """Open a file or folder with the system's default app."""
    path = str(path)
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def beat_range(beats, key, n):
    """Frames ``key`` may move to: strictly between its neighbouring beats (as in the web app)."""
    i = PHASE_KEYS.index(key)
    lo = beats[PHASE_KEYS[i - 1]] + 1 if i > 0 else 0
    hi = beats[PHASE_KEYS[i + 1]] - 1 if i < len(PHASE_KEYS) - 1 else n - 1
    return lo, hi


def clamp_beat(beats, key, frame, n):
    """Frame for ``key`` kept strictly between its neighbouring beats."""
    lo, hi = beat_range(beats, key, n)
    if lo > hi:  # no room (neighbours are adjacent): leave it
        return beats[key]
    return min(max(frame, lo), hi)


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
        self.result = None  # add_pro.main() result of the last analysis
        self.beats = None  # beats being reviewed (frame index per key)
        self.saved = None  # (id, beats) of the last write to the database
        self.cur = 0
        self.view_img = None
        self.sheet_img = None
        self.sheet_stale = False
        self._render_pending = False
        root.title("Swing Match: add a pro swing")
        root.minsize(1100, 700)
        self._build()
        self._toggle_source()
        self._toggle_target()
        self._update_review()
        root.bind_all("<Left>", lambda e: self._key_step(e, -1))
        root.bind_all("<Right>", lambda e: self._key_step(e, 1))
        root.bind_all("<Shift-Left>", lambda e: self._key_step(e, -10))
        root.bind_all("<Shift-Right>", lambda e: self._key_step(e, 10))
        root.protocol("WM_DELETE_WINDOW", self._close)
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
        self.url_entry = ttk.Entry(form, textvariable=self.url, width=30)
        row("Link", self.url_entry)
        self.file = tk.StringVar()
        fbox = ttk.Frame(form)
        self.file_entry = ttk.Entry(fbox, textvariable=self.file, width=20)
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
        ttk.Label(form, text="Slow-motion clips work as they are: the swing is timed by its own hand speed.",
                  foreground="#777", wraplength=360).grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        r += 1

        # Options
        section("4. Options")
        self.model = tk.StringVar(value="heavy")
        row("Pose model", ttk.Combobox(form, textvariable=self.model, values=["heavy", "full", "lite"], state="readonly", width=18), "heavy is best")
        self.max_frames = tk.StringVar(value="240")
        row("Max frames", ttk.Spinbox(form, textvariable=self.max_frames, from_=0, to=5000, increment=60, width=8), "0 = every frame")
        self.cookies = tk.StringVar(value="None")
        row("Browser cookies", ttk.Combobox(form, textvariable=self.cookies, values=BROWSERS, state="readonly", width=18), "if asked to sign in")
        self.keep = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="Keep the downloaded video", variable=self.keep).grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        r += 1

        self.run_btn = ttk.Button(form, text="Analyze", command=self._run)
        self.run_btn.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(12, 4))
        r += 1
        self.progress = ttk.Progressbar(form, mode="indeterminate")
        self.progress.grid(row=r, column=0, columnspan=3, sticky="ew")
        r += 1
        self.status = ttk.Label(form, text="Ready.", wraplength=360)
        self.status.grid(row=r, column=0, columnspan=3, sticky="w", pady=4)

        # Right side: beats review, contact sheet, log
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        self.tabs = ttk.Notebook(right)
        self.tabs.grid(row=0, column=0, sticky="nsew")
        self.tabs.bind("<<NotebookTabChanged>>", lambda _e: self._tab_changed())
        self._build_beats_tab()
        self._build_sheet_tab()
        self._build_log_tab()
        btns = ttk.Frame(right)
        btns.grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Button(btns, text="Open database folder", command=lambda: open_path(ROOT / "data" / "pros")).pack(side="left")
        self.git_btn = ttk.Button(btns, text="Copy git commands", command=self._copy_git, state="disabled")
        self.git_btn.pack(side="left", padx=6)

    def _build_beats_tab(self):
        tab = ttk.Frame(self.tabs, padding=6)
        self.beats_tab = tab
        self.tabs.add(tab, text="Beats")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        self.view = ttk.Label(tab, text="Analyze a swing, then check its beats here.", anchor="center", relief="groove")
        self.view.grid(row=0, column=0, sticky="nsew")
        self.view.bind("<Configure>", lambda _e: self._queue_render())

        side = ttk.Frame(tab, padding=(10, 0, 0, 0))
        side.grid(row=0, column=1, sticky="ns")
        ttk.Label(side, text="Swing beats", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(side, text="Find the frame (drag the bar,\n◀ ▶, or ← → with Shift for 10),\nthen press Set. The comparison\nlines up these six moments.", foreground="#777").grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 6))
        self.beat_rows = {}
        for i, k in enumerate(PHASE_KEYS):
            rr = i + 2
            name = tk.Label(side, text=LABELS[k], fg=COLORS[k], font=("TkDefaultFont", 10, "bold"), anchor="w")
            name.grid(row=rr, column=0, sticky="w", pady=2)
            val = ttk.Label(side, text="", width=14)
            val.grid(row=rr, column=1, sticky="w", padx=4)
            go = ttk.Button(side, text="Go", width=4, command=lambda k=k: self._seek(self.beats[k]))
            go.grid(row=rr, column=2, padx=2)
            st = ttk.Button(side, text="Set", width=4, command=lambda k=k: self._set_beat(k))
            st.grid(row=rr, column=3, padx=2)
            self.beat_rows[k] = (name, val, go, st)
        self.reset_btn = ttk.Button(side, text="Reset to automatic", command=self._reset_beats)
        self.reset_btn.grid(row=10, column=0, columnspan=4, sticky="ew", pady=(10, 2))
        self.save_btn = ttk.Button(side, text="Add to database", command=self._save)
        self.save_btn.grid(row=11, column=0, columnspan=4, sticky="ew", pady=2)
        self.review_note = ttk.Label(side, text="", foreground="#777", wraplength=240)
        self.review_note.grid(row=12, column=0, columnspan=4, sticky="w", pady=(4, 0))

        self.scrub = tk.Canvas(tab, height=48, highlightthickness=0, background="#f3f4f6", cursor="sb_h_double_arrow")
        self.scrub.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.scrub.bind("<Configure>", lambda _e: self._draw_scrub())
        self.scrub.bind("<Button-1>", self._scrub_click)
        self.scrub.bind("<B1-Motion>", self._scrub_click)
        nav = ttk.Frame(tab)
        nav.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.prev_btn = ttk.Button(nav, text="◀", width=3, command=lambda: self._seek(self.cur - 1))
        self.prev_btn.pack(side="left")
        self.next_btn = ttk.Button(nav, text="▶", width=3, command=lambda: self._seek(self.cur + 1))
        self.next_btn.pack(side="left", padx=(2, 8))
        self.frame_lbl = ttk.Label(nav, text="")
        self.frame_lbl.pack(side="left")

    def _build_sheet_tab(self):
        tab = ttk.Frame(self.tabs, padding=6)
        self.sheet_tab = tab
        self.tabs.add(tab, text="Contact sheet")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        self.preview = ttk.Label(tab, text="The six beat frames appear here after a run.", anchor="center", relief="groove")
        self.preview.grid(row=0, column=0, sticky="nsew")
        self.preview.bind("<Configure>", lambda _e: self._show_sheet())
        self.open_prev_btn = ttk.Button(tab, text="Open image", state="disabled",
                                        command=lambda: self.result and open_path(self.result["preview"]))
        self.open_prev_btn.grid(row=1, column=0, sticky="w", pady=(6, 0))

    def _build_log_tab(self):
        tab = ttk.Frame(self.tabs, padding=6)
        self.log_tab = tab
        self.tabs.add(tab, text="Log")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        self.log = tk.Text(tab, height=12, wrap="word", font=("TkFixedFont", 10), state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(tab, command=self.log.yview)
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
        """Command-line arguments for add_pro.main from the form (raises ValueError).
        The window always analyzes with --dry-run; it writes the entry itself
        once the beats have been checked."""
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
        if self.pitcher.get() in ("left", "right"):
            argv += ["--pitcher", self.pitcher.get()]
        if self.use_target.get():
            argv += ["--target-x", f"{self.target.get():.3f}"]
        argv += ["--model", self.model.get()]
        mf = self.max_frames.get().strip() or "240"
        if not re.fullmatch(r"\d+", mf):
            raise ValueError("Max frames must be a whole number (0 = analyze every frame).")
        argv += ["--max-frames", mf]
        if self.cookies.get() != "None":
            argv += ["--cookies-from-browser", self.cookies.get()]
        if self.keep.get():
            argv.append("--keep-video")
        argv.append("--dry-run")
        return argv

    # ------------------------------------------------------------------ running

    def _unsaved(self):
        return self.result is not None and (self.saved is None or self.saved[1] != self.beats)

    def _run(self):
        if self.running:
            return
        try:
            argv = self.build_argv()
        except ValueError as e:
            messagebox.showerror("Missing information", str(e))
            return
        if self._unsaved() and not messagebox.askyesno(
                "Discard this swing?", "The swing you are reviewing hasn't been added to the database. Analyze a new one anyway?"):
            return
        self.running = True
        self.result = self.beats = self.saved = None
        self._update_review()
        self.run_btn.state(["disabled"])
        self.open_prev_btn.state(["disabled"])
        self.git_btn.state(["disabled"])
        self.progress.start(12)
        self.status.config(text="Working… (the first run downloads the pose model)")
        self.tabs.select(self.log_tab)
        self._append("$ python tools/add_pro.py " + " ".join(f'"{a}"' if " " in a else a for a in argv) + "\n")
        threading.Thread(target=self._worker, args=(argv,), daemon=True).start()

    def _worker(self, argv):
        w = QueueWriter(self.q)
        try:
            with redirect_stdout(w), redirect_stderr(w):
                result = add_pro.main(argv, thumb_height=THUMB_HEIGHT)
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
        self.beats = dict(result["phases"])
        self.cur = self.beats["contact"]
        self.open_prev_btn.state(["!disabled"])
        n = len(result["frames"])
        self.status.config(text=f"Analyzed {n} frames. Check each beat on the Beats tab (Go, step with ← →, Set), "
                                "then press Add to database.")
        self._update_review()
        self.tabs.select(self.beats_tab)

    # ------------------------------------------------------------------ beat review

    def _n(self):
        return len(self.result["frames"]) if self.result else 0

    def _seek(self, i):
        if not self.result:
            return
        self.cur = max(0, min(self._n() - 1, int(i)))
        self._update_review()

    def _key_step(self, event, d):
        if not self.result or isinstance(event.widget, TEXT_WIDGETS) or self.tabs.select() != str(self.beats_tab):
            return None
        self._seek(self.cur + d)
        return "break"

    def _set_beat(self, key):
        if not self.result:
            return
        v = clamp_beat(self.beats, key, self.cur, self._n())
        self.beats[key] = v
        if v == self.cur:
            self.review_note.config(text=f"{LABELS[key]} set to frame {v}.")
        else:
            i = PHASE_KEYS.index(key)
            before = f"{LABELS[PHASE_KEYS[i - 1]]} (frame {self.beats[PHASE_KEYS[i - 1]]})" if i > 0 else "the start of the clip"
            after = f"{LABELS[PHASE_KEYS[i + 1]]} (frame {self.beats[PHASE_KEYS[i + 1]]})" if i < len(PHASE_KEYS) - 1 else "the end of the clip"
            self.review_note.config(text=f"Beats stay in order: {LABELS[key]} has to come after {before} and before {after}, "
                                         f"so it is at frame {v}. Move the neighbouring beat first to go further.")
        self.sheet_stale = True
        self._update_review()

    def _reset_beats(self):
        if not self.result:
            return
        self.beats = dict(self.result["phases"])
        self.review_note.config(text="Beats reset to the automatic detection.")
        self.sheet_stale = True
        self._update_review()

    def _time(self, i):
        times = self.result.get("times") or []
        return times[i] if i < len(times) else i / self.result["fps"]

    def _update_review(self):
        """Refresh everything on the Beats tab from self.result/beats/cur."""
        has = self.result is not None
        for w in (self.prev_btn, self.next_btn, self.reset_btn):
            w.state(["!disabled"] if has else ["disabled"])
        for k, (name, val, go, st) in self.beat_rows.items():
            go.state(["!disabled"] if has else ["disabled"])
            st.state(["!disabled"] if has else ["disabled"])
            if not has:
                val.config(text="")
                name.config(font=("TkDefaultFont", 10, "bold"))
                continue
            v = self.beats[k]
            auto = self.result["phases"][k]
            val.config(text=f"#{v}  {self._time(v):.2f} s" + ("" if v == auto else "  ✎"))
            name.config(font=("TkDefaultFont", 10, "bold underline") if v == self.cur else ("TkDefaultFont", 10, "bold"))
        if not has:
            self.save_btn.state(["disabled"])
            self.save_btn.config(text="Add to database")
            self.frame_lbl.config(text="")
            self.review_note.config(text="")
            self.view_img = None
            self.view.config(image="", text="Analyze a swing, then check its beats here." if not self.running else "Analyzing…")
            self._draw_scrub()
            return
        if self.saved is None:
            self.save_btn.state(["!disabled"])
            self.save_btn.config(text="Add to database")
        elif self.saved[1] != self.beats:
            self.save_btn.state(["!disabled"])
            self.save_btn.config(text="Save changes to the database")
        else:
            self.save_btn.state(["disabled"])
            self.save_btn.config(text="Saved ✓")
        here = [LABELS[k] for k in PHASE_KEYS if self.beats[k] == self.cur]
        self.frame_lbl.config(text=f"Frame {self.cur} of {self._n() - 1} · {self._time(self.cur):.2f} s"
                                   + (f" · {', '.join(here)}" if here else ""))
        self._draw_scrub()
        self._queue_render()

    def _draw_scrub(self):
        c = self.scrub
        c.delete("all")
        n = self._n()
        w = max(c.winfo_width(), 200)
        pad, y = 14, 34
        c.create_line(pad, y, w - pad, y, fill="#cbd5e1", width=4, capstyle="round")
        if n < 2:
            return
        x = lambda i: pad + i / (n - 1) * (w - 2 * pad)  # noqa: E731
        c.create_line(pad, y, x(self.cur), y, fill="#94a3b8", width=4, capstyle="round")
        for k in PHASE_KEYS:
            bx = x(self.beats[k])
            c.create_line(bx, y - 12, bx, y + 8, fill=COLORS[k], width=3)
            c.create_text(bx, 11, text=SHORT[k], fill=COLORS[k], font=("TkDefaultFont", 9, "bold"))
        cx = x(self.cur)
        c.create_oval(cx - 7, y - 7, cx + 7, y + 7, fill="#111827", outline="white", width=2)

    def _scrub_click(self, event):
        n = self._n()
        if n < 2:
            return
        w = max(self.scrub.winfo_width(), 200)
        pad = 14
        self._seek(round((event.x - pad) / (w - 2 * pad) * (n - 1)))

    def _queue_render(self):
        if not self._render_pending:
            self._render_pending = True
            self.root.after_idle(self._render)

    def _render(self):
        self._render_pending = False
        r = self.result
        if not r or not r.get("thumbs"):
            return
        try:
            import cv2

            from swingdb.preview import decode_thumb, draw_skeleton, scale_frame

            img = decode_thumb(r["thumbs"][self.cur])
            if img is None:
                return
            w = max(160, self.view.winfo_width() - 6)
            h = max(120, self.view.winfo_height() - 6)
            s = min(w / img.shape[1], h / img.shape[0])
            img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))),
                             interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            draw_skeleton(img, scale_frame(r["frames"][self.cur], r["thumbScale"] * s))
            here = [LABELS[k] for k in PHASE_KEYS if self.beats[k] == self.cur]
            if here:
                cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
                cv2.putText(img, " / ".join(here), (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            ok, png = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if ok:
                self.view_img = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
                self.view.config(image=self.view_img, text="")
        except Exception as e:  # noqa: BLE001
            self.view.config(image="", text=f"Could not show the frame ({e}).")

    # ------------------------------------------------------------------ saving

    def _entry(self):
        """Database entry from the analysis, the reviewed beats and the form's player details."""
        name = self.name.get().strip()
        if not name:
            raise ValueError("Enter the player's name.")
        entry_id = slugify(name)
        if not entry_id:
            raise ValueError("Use letters or digits in the player's name (it becomes the file name).")
        entry = dict(self.result["entryData"])
        entry.update(id=entry_id, name=name, team=self.team.get().strip(), bats=self.bats.get(),
                     notes=self.notes.get().strip(), stanceFrame=int(self.beats["stance"]),
                     phases={k: int(self.beats[k]) for k in PHASE_KEYS})
        return entry

    def _save(self):
        if not self.result:
            return
        try:
            entry = self._entry()
        except ValueError as e:
            messagebox.showerror("Missing information", str(e))
            return
        db = Path(self.result["db"])
        target = db / f"{entry['id']}.json"
        if target.exists() and (self.saved is None or self.saved[0] != entry["id"]):
            if not messagebox.askyesno("Replace entry?", f"{target.name} is already in the database. Replace it with this swing?"):
                self.status.config(text="Not saved. Change the player's name to add it as a separate entry.")
                return
        try:
            path = write_entry(entry, db)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Could not save", f"{type(e).__name__}: {e}")
            return
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        verb = "Updated" if self.saved and self.saved[0] == entry["id"] else "Added"
        self.saved = (entry["id"], dict(self.beats))
        self.result["id"] = entry["id"]
        self._append(f"\n{verb} {rel} (beats: " + ", ".join(f"{k} {entry['phases'][k]}" for k in PHASE_KEYS) + ")\n")
        self.git_btn.state(["!disabled"])
        self.status.config(text=f"{verb} {rel}. Commit data/pros to publish it (Copy git commands).")
        self.sheet_stale = True
        self._refresh_sheet()
        self._update_review()

    # ------------------------------------------------------------------ contact sheet

    def _tab_changed(self):
        if self.tabs.select() == str(self.sheet_tab):
            self._refresh_sheet()

    def _refresh_sheet(self):
        """Redraw the contact sheet with the reviewed beats (from the kept frames)."""
        r = self.result
        if not r or not self.sheet_stale or not r.get("thumbs"):
            self._show_sheet()
            return
        try:
            from swingdb.preview import contact_sheet, decode_thumb, scale_frame

            images = {k: decode_thumb(r["thumbs"][self.beats[k]]) for k in PHASE_KEYS}
            s = r["thumbScale"]
            frames = r["frames"]
            scaled = {self.beats[k]: scale_frame(frames[self.beats[k]], s) for k in PHASE_KEYS}
            path = Path(r["preview"]).with_name(f"{r['id']}.jpg")
            contact_sheet(images, [scaled.get(i) for i in range(len(frames))], self.beats, r["fps"], path,
                          title=f"{self.name.get().strip() or r['id']} - {r['title']}")
            r["preview"] = path
            self.sheet_stale = False
        except Exception as e:  # noqa: BLE001
            self._append(f"\nCould not redraw the contact sheet: {e}\n")
        self._show_sheet()

    def _show_sheet(self):
        path = self.result and self.result.get("preview")
        if not path or not Path(path).exists():
            self.sheet_img = None
            self.preview.config(image="", text="The six beat frames appear here after a run.")
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
                self.sheet_img = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
                self.preview.config(image=self.sheet_img, text="")
        except Exception as e:  # noqa: BLE001
            self.preview.config(text=f"Could not show the contact sheet ({e}); use Open image.")

    # ------------------------------------------------------------------ misc

    def _copy_git(self):
        name = self.name.get().strip()
        cmd = f'git add data/pros && git commit -m "Add {name} swing" && git push'
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        self.status.config(text=f"Copied: {cmd}")

    def _close(self):
        if self._unsaved():
            what = "your beat changes haven't been saved" if self.saved else "this swing hasn't been added to the database"
            if not messagebox.askyesno("Close?", f"Close anyway? {what[0].upper() + what[1:]}."):
                return
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam" if sys.platform.startswith("linux") else ttk.Style().theme_use())
    except tk.TclError:
        pass
    root.geometry("1280x800")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
