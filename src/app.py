"""Desktop GUI: live webcam virtual makeup try-on."""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from detection import FaceDetector
from makeup import MakeupSettings, ProductState, apply_makeup
from palette import SWATCHES, rgb_to_bgr, rgb_to_hex

BG = "#161214"
PANEL = "#24181c"
CARD = "#2e1f24"
ACCENT = "#c45c6a"
TEXT = "#f4e8ec"
MUTED = "#b79aa3"
PREVIEW_BG = "#0e0a0c"


class MakeupTryOnApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Virtual Makeup Try-On")
        self.root.configure(bg=BG)
        self.root.minsize(1100, 680)
        self.root.geometry("1280x760")

        self.cap = None
        self.video_detector = FaceDetector(video=True)
        self.current_raw = None
        self.current_processed = None
        self.photo_image = None
        self.mirror = tk.BooleanVar(value=True)
        self.show_original = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Starting camera…")
        self.fps_text = tk.StringVar(value="0 FPS")
        self._last_fps_time = time.perf_counter()
        self._frame_count = 0
        self._running = True
        self.selected_swatch = {"lipstick": 0, "blush": 0, "eyeshadow": 0}
        self.swatch_buttons: dict[str, list[tk.Canvas]] = {}

        self.settings = MakeupSettings(
            lipstick=ProductState(True, rgb_to_bgr(SWATCHES["lipstick"][0][1]), 0.62),
            blush=ProductState(True, rgb_to_bgr(SWATCHES["blush"][0][1]), 0.40),
            eyeshadow=ProductState(True, rgb_to_bgr(SWATCHES["eyeshadow"][0][1]), 0.28),
        )

        self._build_style()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.start_camera()
        self.root.after(20, self._tick)

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 18))
        style.configure("Sub.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI Semibold", 11))
        style.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="white",
            font=("Segoe UI Semibold", 10),
            padding=8,
        )
        style.map("Accent.TButton", background=[("active", "#d46b78")])
        style.configure(
            "Ghost.TButton",
            background=CARD,
            foreground=TEXT,
            font=("Segoe UI", 10),
            padding=8,
        )
        style.configure("TCheckbutton", background=CARD, foreground=TEXT, font=("Segoe UI", 9))
        style.configure("Bar.TCheckbutton", background=BG, foreground=TEXT, font=("Segoe UI", 9))
        style.configure("TScale", background=CARD, troughcolor="#3d2a30")

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, style="TFrame")
        header.pack(fill=tk.X, padx=20, pady=(16, 8))
        ttk.Label(header, text="Virtual Makeup Try-On", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Live face landmarks · lipstick · blush · eyeshadow  ·  runs locally",
            style="Sub.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

        body = ttk.Frame(self.root, style="TFrame")
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=8)

        preview_wrap = tk.Frame(body, bg=PREVIEW_BG, highlightthickness=0)
        preview_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 16))
        self.preview = tk.Label(preview_wrap, bg=PREVIEW_BG, fg=MUTED, text="Opening webcam…")
        self.preview.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        sidebar = tk.Frame(body, bg=PANEL, width=340)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        canvas = tk.Canvas(sidebar, bg=PANEL, highlightthickness=0)
        scroll = ttk.Scrollbar(sidebar, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas, bg=PANEL)
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor=tk.NW, width=320)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0), pady=12)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=12)

        for product, title in (
            ("lipstick", "Lipstick"),
            ("blush", "Blush"),
            ("eyeshadow", "Eyeshadow"),
        ):
            self._product_card(inner, product, title)

        actions = tk.Frame(inner, bg=PANEL)
        actions.pack(fill=tk.X, pady=(8, 4))
        ttk.Button(actions, text="Use webcam", style="Accent.TButton", command=self.start_camera).pack(
            fill=tk.X, pady=4
        )
        ttk.Button(actions, text="Save result", style="Ghost.TButton", command=self.save_result).pack(
            fill=tk.X, pady=4
        )

        footer = ttk.Frame(self.root, style="TFrame")
        footer.pack(fill=tk.X, padx=20, pady=(0, 12))
        ttk.Checkbutton(footer, text="Mirror camera", variable=self.mirror, style="Bar.TCheckbutton").pack(
            side=tk.LEFT, padx=(0, 16)
        )
        ttk.Checkbutton(
            footer, text="Show original (before)", variable=self.show_original, style="Bar.TCheckbutton"
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(footer, textvariable=self.fps_text, style="Status.TLabel").pack(side=tk.RIGHT)
        ttk.Label(footer, textvariable=self.status, style="Status.TLabel").pack(side=tk.RIGHT, padx=16)

    def _product_card(self, parent: tk.Frame, key: str, title: str) -> None:
        card = tk.Frame(parent, bg=CARD)
        card.pack(fill=tk.X, pady=6, padx=4)

        top = tk.Frame(card, bg=CARD)
        top.pack(fill=tk.X, padx=10, pady=(10, 4))
        enabled = tk.BooleanVar(value=True)

        def toggle() -> None:
            getattr(self.settings, key).enabled = enabled.get()

        ttk.Checkbutton(top, text=title, variable=enabled, command=toggle, style="TCheckbutton").pack(
            side=tk.LEFT
        )

        swatch_row = tk.Frame(card, bg=CARD)
        swatch_row.pack(fill=tk.X, padx=10, pady=6)
        self.swatch_buttons[key] = []
        for i, (name, rgb) in enumerate(SWATCHES[key]):
            btn = tk.Canvas(
                swatch_row,
                width=28,
                height=28,
                bg=rgb_to_hex(rgb),
                highlightthickness=2,
                highlightbackground="#ffffff" if i == self.selected_swatch[key] else "#5a3d44",
                cursor="hand2",
            )
            btn.grid(row=i // 8, column=i % 8, padx=3, pady=3)
            btn.bind("<Button-1>", lambda _e, p=key, idx=i: self._select_swatch(p, idx))
            btn.bind("<Enter>", lambda _e, n=name: self.status.set(n))
            self.swatch_buttons[key].append(btn)

        intensity_row = tk.Frame(card, bg=CARD)
        intensity_row.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Label(intensity_row, text="Intensity", style="Muted.TLabel").pack(anchor=tk.W)
        scale = tk.Scale(
            intensity_row,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            bg=CARD,
            fg=TEXT,
            highlightthickness=0,
            troughcolor="#3d2a30",
            activebackground=ACCENT,
            length=260,
            showvalue=True,
            command=lambda val, p=key: self._set_intensity(p, val),
        )
        scale.set(int(getattr(self.settings, key).intensity * 100))
        scale.pack(fill=tk.X)

    def _select_swatch(self, product: str, index: int) -> None:
        self.selected_swatch[product] = index
        name, rgb = SWATCHES[product][index]
        getattr(self.settings, product).color_bgr = rgb_to_bgr(rgb)
        for i, btn in enumerate(self.swatch_buttons[product]):
            btn.configure(highlightbackground="#ffffff" if i == index else "#5a3d44")
        self.status.set(f"{product.title()}: {name}")

    def _set_intensity(self, product: str, value: str) -> None:
        getattr(self.settings, product).intensity = float(value) / 100.0

    def start_camera(self) -> None:
        self._release_camera()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = None
            self.status.set("Webcam not found")
            messagebox.showwarning("Webcam", "Could not open the webcam.")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.status.set("Live camera")

    def save_result(self) -> None:
        if self.current_processed is None:
            messagebox.showinfo("Save", "Nothing to save yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            initialfile="makeup_tryon.jpg",
        )
        if not path:
            return
        cv2.imwrite(path, self.current_processed)
        self.status.set(f"Saved {Path(path).name}")

    def _release_camera(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _tick(self) -> None:
        if not self._running:
            return
        if self.cap is not None:
            ok, frame = self.cap.read()
            if ok:
                if self.mirror.get():
                    frame = cv2.flip(frame, 1)
                self._process_and_show(frame, self.video_detector)
        self.root.after(15, self._tick)

    def _process_and_show(self, frame, detector: FaceDetector) -> None:
        self.current_raw = frame
        if self.show_original.get():
            processed = frame
            self.status.set("Original (makeup hidden)")
        else:
            landmarks = detector.detect_bgr(frame)
            if landmarks:
                processed = apply_makeup(frame, landmarks, self.settings)
                self.status.set("Face detected — makeup live")
            else:
                processed = frame
                self.status.set("No face detected")
        self.current_processed = processed
        self._show_frame(processed)
        self._update_fps()

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_w = max(self.preview.winfo_width(), 640)
        max_h = max(self.preview.winfo_height(), 480)
        image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.photo_image, text="")

    def _update_fps(self) -> None:
        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._last_fps_time
        if elapsed >= 0.5:
            self.fps_text.set(f"{self._frame_count / elapsed:.0f} FPS")
            self._frame_count = 0
            self._last_fps_time = now

    def on_close(self) -> None:
        self._running = False
        self._release_camera()
        try:
            self.video_detector.close()
        except Exception:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()