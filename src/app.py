"""Virtual Makeup Studio — Professional Real-Time Virtual Try-On Desktop Application.

Features:
- Decoupled multi-threaded camera capture and background makeup rendering pipeline (30+ FPS).
- Zero main-thread blocking for silky smooth 60 FPS UI responsiveness.
- Google Meet-grade studio auto-lighting and low-light enhancement.
- Commercial luxury beauty studio UI: cosmetic pearl swatches, live shade badges,
  category navigation, 1-click curated look presets, and interactive Before/After comparison.
"""

from __future__ import annotations

import copy
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# Attempt Windows High-DPI awareness for razor-sharp rendering
try:
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

from detection import FaceDetector
from makeup import MakeupSettings, ProductState, apply_makeup
from palette import SWATCHES, bgr_to_rgb, hex_to_bgr, rgb_to_bgr, rgb_to_hex

# ---------------------------------------------------------------------------
# Luxury Beauty Studio Theme Palette
# ---------------------------------------------------------------------------
BG = "#0c0d12"             # Deep velvet obsidian background
SURFACE = "#14161f"        # Elevated satin sidebar surface
CARD_BG = "#1b1d28"        # Sleek product card background
CARD_ACTIVE = "#222534"    # Active product card highlight
CARD_BORDER = "#282b3c"    # Subtle 1px studio border
CARD_BORDER_ON = "#e11d48" # Rose border when product is active

ACCENT = "#e11d48"         # Radiant Rose (luxury cosmetic brand accent)
ACCENT_HOVER = "#f43f5e"   # Bright Rose hover
ACCENT_MUTED = "#3e1823"   # Soft rose tint

TEXT_PRIMARY = "#f8fafc"   # Crisp pearl white
TEXT_SECONDARY = "#94a3b8" # Soft slate taupe
TEXT_MUTED = "#64748b"     # Subtle placeholder muted text

VIEWPORT_BG = "#08090d"    # Deep black viewport backdrop
LIVE_EMERALD = "#10b981"   # Glowing studio indicator green


# ---------------------------------------------------------------------------
# Curated 1-Click Look Presets (Commercial Beauty Experience)
# ---------------------------------------------------------------------------
PRESET_LOOKS: dict[str, dict] = {
    "Natural Glow": {
        "foundation": {"enabled": True, "swatch": 1, "intensity": 0.22},
        "blush": {"enabled": True, "swatch": 0, "intensity": 0.28},
        "eyeshadow": {"enabled": True, "swatch": 0, "intensity": 0.25},
        "eyeliner": {"enabled": False, "swatch": 0, "intensity": 0.50},
        "brow": {"enabled": True, "swatch": 0, "intensity": 0.20},
        "lipstick": {"enabled": True, "swatch": 2, "intensity": 0.50, "blend_mode": "overlay"},
    },
    "Red Carpet": {
        "foundation": {"enabled": True, "swatch": 1, "intensity": 0.35},
        "blush": {"enabled": True, "swatch": 1, "intensity": 0.35},
        "eyeshadow": {"enabled": True, "swatch": 2, "intensity": 0.40},
        "eyeliner": {"enabled": True, "swatch": 0, "intensity": 0.85, "style": "classic_wing"},
        "brow": {"enabled": True, "swatch": 0, "intensity": 0.30},
        "lipstick": {"enabled": True, "swatch": 0, "intensity": 0.70, "blend_mode": "multiply"},
    },
    "Berry Velvet": {
        "foundation": {"enabled": True, "swatch": 2, "intensity": 0.28},
        "blush": {"enabled": True, "swatch": 2, "intensity": 0.32},
        "eyeshadow": {"enabled": True, "swatch": 4, "intensity": 0.38},
        "eyeliner": {"enabled": True, "swatch": 0, "intensity": 0.75, "style": "tightline"},
        "brow": {"enabled": True, "swatch": 1, "intensity": 0.25},
        "lipstick": {"enabled": True, "swatch": 3, "intensity": 0.65, "blend_mode": "multiply"},
    },
    "Warm Sunset": {
        "foundation": {"enabled": False, "swatch": 1, "intensity": 0.25},
        "blush": {"enabled": True, "swatch": 3, "intensity": 0.38},
        "eyeshadow": {"enabled": True, "swatch": 2, "intensity": 0.35},
        "eyeliner": {"enabled": True, "swatch": 1, "intensity": 0.70, "style": "classic_wing"},
        "brow": {"enabled": True, "swatch": 0, "intensity": 0.22},
        "lipstick": {"enabled": True, "swatch": 4, "intensity": 0.55, "blend_mode": "multiply"},
    },
    "Bare Face": {
        "foundation": {"enabled": False},
        "blush": {"enabled": False},
        "eyeshadow": {"enabled": False},
        "eyeliner": {"enabled": False},
        "brow": {"enabled": False},
        "lipstick": {"enabled": False},
    },
}


# ---------------------------------------------------------------------------
# Background Non-Blocking Camera Stream
# ---------------------------------------------------------------------------
class CameraStream:
    """Dedicated background capture thread that eliminates hardware camera sync lag."""

    def __init__(self, src: int = 0) -> None:
        self.src = src
        self.cap: cv2.VideoCapture | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._is_active = False

    def start(self) -> bool:
        self.stop()
        # DirectShow on Windows with hardware MJPEG negotiation
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.src)

        if not self.cap.isOpened():
            self.cap = None
            self._is_active = False
            return False

        # Configure camera hardware properties for smooth 30 FPS
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running = True
        self._is_active = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _capture_loop(self) -> None:
        while self._running and self.cap is not None:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

    def read_latest(self) -> np.ndarray | None:
        """Fetch the freshest available frame with 0ms waiting time."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def is_opened(self) -> bool:
        return self._is_active and (self.cap is not None)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.3)
        self._thread = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self._is_active = False
        with self._lock:
            self._frame = None


# ---------------------------------------------------------------------------
# Studio Auto-Lighting & Low-Light Enhancement (Google Meet Grade)
# ---------------------------------------------------------------------------
def enhance_studio_lighting(frame: np.ndarray) -> np.ndarray:
    """
    Subtle dynamic range and exposure compensation in low-light environments.
    Mimics Google Meet's auto low-light adjustment for bright, glowing, clean video.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray))

    # Apply adaptive luminance correction if underexposed
    if mean_lum < 135:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        l_enhanced = cv2.addWeighted(cl, 0.60, l, 0.40, 0)
        limg = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return frame


# ---------------------------------------------------------------------------
# Main Virtual Makeup Studio Application
# ---------------------------------------------------------------------------
class MakeupTryOnApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Virtual Makeup Studio — Live Beauty Try-On")
        self.root.configure(bg=BG)
        self.root.minsize(1120, 720)
        self.root.geometry("1340x820")

        # Capture and Threading State
        self.camera = CameraStream(src=0)
        self.detector = FaceDetector(video=True)
        self._running = True

        # Pure Python primitives for thread-safe cross-thread state
        self._settings_lock = threading.Lock()
        self.is_mirrored = True
        self.is_showing_original = False
        self.is_studio_light_on = True

        # Latest Rendered Frames for Display and Snapshot
        self._display_lock = threading.Lock()
        self._latest_processed_frame: np.ndarray | None = None
        self._latest_raw_frame: np.ndarray | None = None
        self._face_detected = False
        self._pipeline_fps = 0.0

        # UI String/Control Variables (Main Thread Only)
        self.status_text = tk.StringVar(value="Initializing camera...")
        self.fps_badge_text = tk.StringVar(value="-- FPS")
        self.tracking_badge_text = tk.StringVar(value="Detecting face...")
        self.selected_category = tk.StringVar(value="All")

        # Color Swatches Tracking
        self.swatch_canvases: dict[str, list[tk.Canvas]] = {}
        self.custom_color_chips: dict[str, tk.Canvas] = {}
        self.shade_labels: dict[str, tk.Label] = {}
        self.card_frames: dict[str, tk.Frame] = {}
        self.toggle_buttons: dict[str, tk.Button] = {}
        self.selected_swatch: dict[str, int | None] = {
            "foundation": 1,
            "blush": 1,
            "eyeshadow": 0,
            "eyeliner": 0,
            "brow": 0,
            "lipstick": 0,
        }

        # Initialize Professional Makeup Settings
        self.settings = MakeupSettings(
            foundation=ProductState(
                enabled=False,
                color_bgr=rgb_to_bgr(SWATCHES["foundation"][1][1]),
                intensity=0.25,
            ),
            blush=ProductState(
                enabled=True,
                color_bgr=rgb_to_bgr(SWATCHES["blush"][1][1]),
                intensity=0.28,
            ),
            eyeshadow=ProductState(
                enabled=True,
                color_bgr=rgb_to_bgr(SWATCHES["eyeshadow"][0][1]),
                intensity=0.32,
            ),
            eyeliner=ProductState(
                enabled=False,
                color_bgr=rgb_to_bgr(SWATCHES["eyeliner"][0][1]),
                intensity=0.80,
                style="classic_wing",
            ),
            brow=ProductState(
                enabled=True,
                color_bgr=rgb_to_bgr(SWATCHES["brow"][0][1]),
                intensity=0.25,
            ),
            lipstick=ProductState(
                enabled=True,
                color_bgr=rgb_to_bgr(SWATCHES["lipstick"][0][1]),
                intensity=0.60,
                blend_mode="multiply",  # Velvet matte
            ),
        )

        self.photo_image = None

        # Build UI and launch background processing thread
        self._build_style()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start Camera and Worker Thread
        self.start_camera()
        self._worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self._worker_thread.start()

        # UI Refresh Loop (60 FPS on main thread)
        self.root.after(16, self._render_loop)

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=BG)
        style.configure("Surface.TFrame", background=SURFACE)
        style.configure("TLabel", background=SURFACE, foreground=TEXT_PRIMARY, font=("Segoe UI", 9))
        style.configure("Muted.TLabel", background=SURFACE, foreground=TEXT_MUTED, font=("Segoe UI", 8))
        style.configure("Header.TLabel", background=SURFACE, foreground=TEXT_PRIMARY, font=("Segoe UI Semibold", 10))
        style.configure("Brand.TLabel", background=BG, foreground=TEXT_PRIMARY, font=("Segoe UI Semibold", 14))

        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="#ffffff",
            font=("Segoe UI Semibold", 9),
            padding=(14, 7),
            borderwidth=0,
        )
        style.map("Accent.TButton", background=[("active", ACCENT_HOVER)])

        style.configure(
            "Ghost.TButton",
            background=CARD_BG,
            foreground=TEXT_PRIMARY,
            font=("Segoe UI Semibold", 9),
            padding=(12, 6),
            borderwidth=1,
            relief="solid",
        )
        style.map("Ghost.TButton", background=[("active", CARD_ACTIVE)])
        style.configure("TCombobox", fieldbackground=CARD_BG, background=CARD_BORDER, foreground=TEXT_PRIMARY)

    def _build_ui(self) -> None:
        # -------------------------------------------------------------------
        # Top Header Bar
        # -------------------------------------------------------------------
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill=tk.X, padx=22, pady=(12, 8))

        brand_frame = tk.Frame(header, bg=BG)
        brand_frame.pack(side=tk.LEFT)

        # Glowing beauty logo pill
        logo_badge = tk.Label(
            brand_frame,
            text="PRO",
            font=("Segoe UI Black", 8),
            bg=ACCENT,
            fg="#ffffff",
            padx=5,
            pady=1,
        )
        logo_badge.pack(side=tk.LEFT, padx=(0, 8))

        title_label = tk.Label(
            brand_frame,
            text="Virtual Makeup Studio",
            font=("Segoe UI Semibold", 14),
            bg=BG,
            fg=TEXT_PRIMARY,
        )
        title_label.pack(side=tk.LEFT)

        subtitle_label = tk.Label(
            brand_frame,
            text="· Real-Time AR Beauty Try-On",
            font=("Segoe UI", 10),
            bg=BG,
            fg=TEXT_MUTED,
        )
        subtitle_label.pack(side=tk.LEFT, padx=(4, 0))

        # Status badge in header
        status_box = tk.Frame(header, bg=BG)
        status_box.pack(side=tk.RIGHT)

        status_dot = tk.Canvas(status_box, width=8, height=8, bg=BG, highlightthickness=0)
        status_dot.pack(side=tk.LEFT, padx=(0, 6))
        status_dot.create_oval(1, 1, 7, 7, fill=LIVE_EMERALD, outline="")

        self.status_label = tk.Label(
            status_box,
            textvariable=self.status_text,
            font=("Segoe UI", 9),
            bg=BG,
            fg=TEXT_SECONDARY,
        )
        self.status_label.pack(side=tk.LEFT)

        # -------------------------------------------------------------------
        # Main Layout Grid (Viewport Hero + Right Control Studio)
        # -------------------------------------------------------------------
        main_grid = tk.Frame(self.root, bg=BG)
        main_grid.pack(fill=tk.BOTH, expand=True, padx=22, pady=(0, 10))
        main_grid.columnconfigure(0, weight=1)
        main_grid.columnconfigure(1, weight=0, minsize=410)
        main_grid.rowconfigure(0, weight=1)

        # -------------------------------------------------------------------
        # Left Side: Hero Viewport Container
        # -------------------------------------------------------------------
        viewport_wrapper = tk.Frame(main_grid, bg=BG)
        viewport_wrapper.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        viewport_wrapper.rowconfigure(0, weight=1)
        viewport_wrapper.columnconfigure(0, weight=1)

        self.preview_container = tk.Frame(
            viewport_wrapper,
            bg=VIEWPORT_BG,
            highlightthickness=1,
            highlightbackground=CARD_BORDER,
        )
        self.preview_container.grid(row=0, column=0, sticky="nsew")
        self.preview_container.pack_propagate(False)

        # Video Preview Display Label
        self.preview = tk.Label(
            self.preview_container,
            bg=VIEWPORT_BG,
            fg=TEXT_MUTED,
            text="Connecting live video stream...",
            font=("Segoe UI", 11),
        )
        self.preview.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Floating HUD Overlay (Top of Camera Viewport)
        hud_overlay = tk.Frame(self.preview_container, bg="")
        hud_overlay.place(relx=0.02, rely=0.03, relwidth=0.96)

        # Live Emerald Badge
        live_pill = tk.Frame(hud_overlay, bg="#121319", highlightthickness=1, highlightbackground="#222430", padx=8, pady=3)
        live_pill.pack(side=tk.LEFT)

        dot_cv = tk.Canvas(live_pill, width=8, height=8, bg="#121319", highlightthickness=0)
        dot_cv.pack(side=tk.LEFT, padx=(0, 5))
        dot_cv.create_oval(1, 1, 7, 7, fill=LIVE_EMERALD, outline="")

        tk.Label(
            live_pill,
            text="LIVE",
            font=("Segoe UI Black", 8),
            bg="#121319",
            fg=LIVE_EMERALD,
        ).pack(side=tk.LEFT)

        # Tracking and FPS Pill
        tracking_pill = tk.Frame(hud_overlay, bg="#121319", highlightthickness=1, highlightbackground="#222430", padx=10, pady=3)
        tracking_pill.pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(
            tracking_pill,
            textvariable=self.tracking_badge_text,
            font=("Segoe UI", 8),
            bg="#121319",
            fg=TEXT_SECONDARY,
        ).pack(side=tk.LEFT)

        fps_pill = tk.Frame(hud_overlay, bg="#121319", highlightthickness=1, highlightbackground="#222430", padx=8, pady=3)
        fps_pill.pack(side=tk.RIGHT)

        tk.Label(
            fps_pill,
            textvariable=self.fps_badge_text,
            font=("Segoe UI Semibold", 8),
            bg="#121319",
            fg=ACCENT,
        ).pack(side=tk.RIGHT)

        # Viewport Floating Bottom Toolbar
        bottom_bar = tk.Frame(viewport_wrapper, bg=BG)
        bottom_bar.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        # Compare Before & After Button (Interactive Hold / Click)
        self.btn_compare = tk.Button(
            bottom_bar,
            text="👁 Hold for Before / After",
            font=("Segoe UI Semibold", 9),
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            activebackground=CARD_ACTIVE,
            activeforeground=TEXT_PRIMARY,
            relief="flat",
            padx=12,
            pady=6,
            cursor="hand2",
            highlightthickness=1,
            highlightbackground=CARD_BORDER,
        )
        self.btn_compare.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_compare.bind("<ButtonPress-1>", lambda _e: self._set_compare(True))
        self.btn_compare.bind("<ButtonRelease-1>", lambda _e: self._set_compare(False))

        # Studio Lighting Toggle (Google Meet auto-light look)
        self.btn_light = tk.Button(
            bottom_bar,
            text="✦ Studio Light: ON",
            font=("Segoe UI", 9),
            bg="#1d202b",
            fg=ACCENT_HOVER,
            activebackground=CARD_ACTIVE,
            relief="flat",
            padx=12,
            pady=6,
            cursor="hand2",
            highlightthickness=1,
            highlightbackground=CARD_BORDER,
            command=self._toggle_studio_light,
        )
        self.btn_light.pack(side=tk.LEFT, padx=(0, 8))

        # Mirror Camera Flip
        self.btn_mirror = tk.Button(
            bottom_bar,
            text="⇄ Mirror Flip",
            font=("Segoe UI", 9),
            bg=CARD_BG,
            fg=TEXT_SECONDARY,
            activebackground=CARD_ACTIVE,
            relief="flat",
            padx=10,
            pady=6,
            cursor="hand2",
            highlightthickness=1,
            highlightbackground=CARD_BORDER,
            command=self._toggle_mirror,
        )
        self.btn_mirror.pack(side=tk.LEFT, padx=(0, 8))

        # Capture Snapshot Button (High-profile action)
        btn_snap = tk.Button(
            bottom_bar,
            text="📸 Capture Look",
            font=("Segoe UI Semibold", 9),
            bg=ACCENT,
            fg="#ffffff",
            activebackground=ACCENT_HOVER,
            activeforeground="#ffffff",
            relief="flat",
            padx=16,
            pady=6,
            cursor="hand2",
            command=self.save_result,
        )
        btn_snap.pack(side=tk.RIGHT)

        # Reset All Products
        btn_reset = tk.Button(
            bottom_bar,
            text="↺ Reset",
            font=("Segoe UI", 9),
            bg=CARD_BG,
            fg=TEXT_MUTED,
            activebackground=CARD_ACTIVE,
            activeforeground=TEXT_PRIMARY,
            relief="flat",
            padx=10,
            pady=6,
            cursor="hand2",
            command=lambda: self._apply_preset("Bare Face"),
        )
        btn_reset.pack(side=tk.RIGHT, padx=(0, 8))

        # -------------------------------------------------------------------
        # Right Side: Professional Studio Sidebar
        # -------------------------------------------------------------------
        sidebar_frame = tk.Frame(
            main_grid,
            bg=SURFACE,
            width=420,
            highlightthickness=1,
            highlightbackground=CARD_BORDER,
        )
        sidebar_frame.grid(row=0, column=1, sticky="nsew")
        sidebar_frame.pack_propagate(False)

        # 1. 1-Click Curated Look Presets Bar
        preset_box = tk.Frame(sidebar_frame, bg=SURFACE)
        preset_box.pack(fill=tk.X, padx=14, pady=(12, 6))

        tk.Label(
            preset_box,
            text="CURATED LOOKS",
            font=("Segoe UI Semibold", 8),
            bg=SURFACE,
            fg=TEXT_MUTED,
        ).pack(anchor=tk.W, pady=(0, 4))

        preset_row = tk.Frame(preset_box, bg=SURFACE)
        preset_row.pack(fill=tk.X)

        for look_name in ["Natural Glow", "Red Carpet", "Berry Velvet", "Warm Sunset"]:
            btn = tk.Button(
                preset_row,
                text=look_name,
                font=("Segoe UI", 8),
                bg=CARD_BG,
                fg=TEXT_PRIMARY,
                activebackground=ACCENT,
                activeforeground="#ffffff",
                relief="flat",
                padx=8,
                pady=3,
                cursor="hand2",
                highlightthickness=1,
                highlightbackground=CARD_BORDER,
                command=lambda ln=look_name: self._apply_preset(ln),
            )
            btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # 2. Category Filter Tabs (All / Face / Eyes / Lips)
        tab_box = tk.Frame(sidebar_frame, bg=SURFACE)
        tab_box.pack(fill=tk.X, padx=14, pady=(6, 8))

        self.tab_buttons: dict[str, tk.Button] = {}
        for cat in ["All", "Face", "Eyes", "Lips"]:
            t_btn = tk.Button(
                tab_box,
                text=cat,
                font=("Segoe UI Semibold", 9),
                bg=ACCENT if cat == "All" else CARD_BG,
                fg="#ffffff" if cat == "All" else TEXT_SECONDARY,
                activebackground=ACCENT,
                activeforeground="#ffffff",
                relief="flat",
                padx=10,
                pady=3,
                cursor="hand2",
                command=lambda c=cat: self._switch_category(c),
            )
            t_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
            self.tab_buttons[cat] = t_btn

        # 3. Scrollable Product Cards Canvas
        canvas_container = tk.Frame(sidebar_frame, bg=SURFACE)
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        self.canvas = tk.Canvas(canvas_container, bg=SURFACE, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.sidebar_inner = tk.Frame(self.canvas, bg=SURFACE)

        self.sidebar_inner.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas_window = self.canvas.create_window((0, 0), window=self.sidebar_inner, anchor=tk.NW, width=382)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            delta = int(-1 * (event.delta / 120)) if event.delta else 0
            self.canvas.yview_scroll(delta, "units")

        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Build Product Cards
        self._build_product_card("lipstick", "Lipstick", category="Lips", extra="lipstick")
        self._build_product_card("blush", "Blush", category="Face")
        self._build_product_card("eyeshadow", "Eyeshadow", category="Eyes")
        self._build_product_card("eyeliner", "Eyeliner", category="Eyes", extra="eyeliner")
        self._build_product_card("brow", "Eyebrows", category="Eyes")
        self._build_product_card("foundation", "Foundation", category="Face")

    def _build_product_card(self, key: str, title: str, category: str, extra: str | None = None) -> None:
        state = getattr(self.settings, key)
        is_on = state.enabled

        card = tk.Frame(
            self.sidebar_inner,
            bg=CARD_ACTIVE if is_on else CARD_BG,
            highlightthickness=1,
            highlightbackground=CARD_BORDER_ON if is_on else CARD_BORDER,
        )
        card.pack(fill=tk.X, pady=4, padx=2)
        self.card_frames[key] = card

        # Top Header Row (Toggle, Title, Active Shade Badge)
        top_row = tk.Frame(card, bg=card["bg"])
        top_row.pack(fill=tk.X, padx=12, pady=(8, 4))

        toggle_btn = tk.Button(
            top_row,
            text="ON" if is_on else "OFF",
            font=("Segoe UI Black", 7),
            bg=ACCENT if is_on else "#282a36",
            fg="#ffffff" if is_on else TEXT_MUTED,
            activebackground=ACCENT_HOVER,
            activeforeground="#ffffff",
            relief="flat",
            padx=7,
            pady=2,
            cursor="hand2",
            command=lambda p=key: self._toggle_product(p),
        )
        toggle_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.toggle_buttons[key] = toggle_btn

        title_lbl = tk.Label(
            top_row,
            text=title,
            font=("Segoe UI Semibold", 10),
            bg=card["bg"],
            fg=TEXT_PRIMARY,
        )
        title_lbl.pack(side=tk.LEFT)

        cur_idx = self.selected_swatch.get(key)
        swatch_list = SWATCHES.get(key, [])
        cur_name = swatch_list[cur_idx][0] if cur_idx is not None and cur_idx < len(swatch_list) else "Custom"

        shade_lbl = tk.Label(
            top_row,
            text=f"● {cur_name}",
            font=("Segoe UI", 8),
            bg=card["bg"],
            fg=ACCENT_HOVER if is_on else TEXT_MUTED,
        )
        shade_lbl.pack(side=tk.RIGHT)
        self.shade_labels[key] = shade_lbl

        # Extra Dropdown Options
        if extra == "eyeliner":
            opt_row = tk.Frame(card, bg=card["bg"])
            opt_row.pack(fill=tk.X, padx=12, pady=(2, 4))
            tk.Label(opt_row, text="Style:", font=("Segoe UI", 8), bg=card["bg"], fg=TEXT_MUTED).pack(side=tk.LEFT, padx=(0, 6))

            style_map = {
                "Classic Wing": "classic_wing",
                "Tightline": "tightline",
                "Bold Kohl": "bold",
                "Cat Eye": "cat_eye",
            }
            inv_style_map = {v: k for k, v in style_map.items()}

            style_var = tk.StringVar(value=inv_style_map.get(self.settings.eyeliner.style, "Classic Wing"))
            style_combo = ttk.Combobox(
                opt_row,
                textvariable=style_var,
                values=list(style_map.keys()),
                state="readonly",
                width=14,
            )
            style_combo.pack(side=tk.LEFT)

            def _on_style_change(_e):
                with self._settings_lock:
                    self.settings.eyeliner.style = style_map.get(style_var.get(), "classic_wing")
                self.status_text.set(f"Eyeliner Style: {style_var.get()}")

            style_combo.bind("<<ComboboxSelected>>", _on_style_change)

        elif extra == "lipstick":
            opt_row = tk.Frame(card, bg=card["bg"])
            opt_row.pack(fill=tk.X, padx=12, pady=(2, 4))
            tk.Label(opt_row, text="Finish:", font=("Segoe UI", 8), bg=card["bg"], fg=TEXT_MUTED).pack(side=tk.LEFT, padx=(0, 6))

            mode_map = {
                "Velvet Matte": "multiply",
                "Satin Cream": "overlay",
                "Sheer Tint": "color",
            }
            inv_mode_map = {v: k for k, v in mode_map.items()}

            mode_var = tk.StringVar(value=inv_mode_map.get(self.settings.lipstick.blend_mode, "Velvet Matte"))
            mode_combo = ttk.Combobox(
                opt_row,
                textvariable=mode_var,
                values=list(mode_map.keys()),
                state="readonly",
                width=13,
            )
            mode_combo.pack(side=tk.LEFT)

            def _on_mode_change(_e):
                with self._settings_lock:
                    self.settings.lipstick.blend_mode = mode_map.get(mode_var.get(), "multiply")
                self.status_text.set(f"Lipstick Finish: {mode_var.get()}")

            mode_combo.bind("<<ComboboxSelected>>", _on_mode_change)

        # Cosmetic Pearl Swatches Row
        swatch_frame = tk.Frame(card, bg=card["bg"])
        swatch_frame.pack(fill=tk.X, padx=12, pady=(4, 6))

        self.swatch_canvases[key] = []
        for i, (name, rgb) in enumerate(SWATCHES.get(key, [])):
            c_hex = rgb_to_hex(rgb)
            is_sel = i == self.selected_swatch.get(key)

            btn_cv = tk.Canvas(swatch_frame, width=24, height=24, bg=card["bg"], highlightthickness=0, cursor="hand2")
            btn_cv.grid(row=i // 7, column=i % 7, padx=3, pady=3)
            self._draw_swatch_circle(btn_cv, c_hex, is_sel)

            btn_cv.bind("<Button-1>", lambda _e, p=key, idx=i: self._select_swatch(p, idx))
            btn_cv.bind("<Enter>", lambda _e, n=name, p=title: self.status_text.set(f"{p}: {n}"))
            self.swatch_canvases[key].append(btn_cv)

        # Custom Color Picker Row
        custom_row = tk.Frame(card, bg=card["bg"])
        custom_row.pack(fill=tk.X, padx=12, pady=(0, 4))

        custom_chip = tk.Canvas(custom_row, width=16, height=16, bg=card["bg"], highlightthickness=0)
        custom_chip.pack(side=tk.LEFT, padx=(0, 6))
        self._draw_swatch_circle(custom_chip, rgb_to_hex(bgr_to_rgb(state.color_bgr)), selected=False)
        self.custom_color_chips[key] = custom_chip

        custom_btn = tk.Button(
            custom_row,
            text="Pick Custom Color...",
            font=("Segoe UI", 8),
            bg="#232532",
            fg=TEXT_SECONDARY,
            activebackground=CARD_BORDER,
            activeforeground=TEXT_PRIMARY,
            relief="flat",
            padx=8,
            pady=2,
            cursor="hand2",
            command=lambda p=key, t=title: self._choose_custom_color(p, t),
        )
        custom_btn.pack(side=tk.LEFT)

        # Intensity Slider Row
        slider_row = tk.Frame(card, bg=card["bg"])
        slider_row.pack(fill=tk.X, padx=12, pady=(4, 8))

        slider_header = tk.Frame(slider_row, bg=card["bg"])
        slider_header.pack(fill=tk.X)

        tk.Label(
            slider_header,
            text="Coverage & Strength",
            font=("Segoe UI", 8),
            bg=card["bg"],
            fg=TEXT_MUTED,
        ).pack(side=tk.LEFT)

        pct_val = int(state.intensity * 100)
        pct_lbl = tk.Label(
            slider_header,
            text=f"{pct_val}%",
            font=("Segoe UI Semibold", 8),
            bg=card["bg"],
            fg=TEXT_SECONDARY,
        )
        pct_lbl.pack(side=tk.RIGHT)

        scale = tk.Scale(
            slider_row,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            bg=card["bg"],
            fg=TEXT_SECONDARY,
            highlightthickness=0,
            troughcolor="#101117",
            activebackground=ACCENT,
            showvalue=False,
            command=lambda val, p=key, l=pct_lbl: self._on_slider_change(p, val, l),
        )
        scale.set(pct_val)
        scale.pack(fill=tk.X, pady=(2, 0))

    def _draw_swatch_circle(self, canvas: tk.Canvas, hex_color: str, selected: bool) -> None:
        canvas.delete("all")
        w = int(canvas["width"])
        h = int(canvas["height"])
        if selected:
            canvas.create_oval(1, 1, w - 1, h - 1, outline="#ffffff", width=2)
            canvas.create_oval(4, 4, w - 4, h - 4, fill=hex_color, outline="#101117", width=1)
        else:
            canvas.create_oval(2, 2, w - 2, h - 2, fill=hex_color, outline="#333646", width=1)

    def _toggle_product(self, product: str) -> None:
        with self._settings_lock:
            state = getattr(self.settings, product)
            state.enabled = not state.enabled
            is_on = state.enabled

        self._update_card_visuals(product, is_on)
        self.status_text.set(f"{product.title()}: {'Applied' if is_on else 'Removed'}")

    def _update_card_visuals(self, product: str, is_on: bool) -> None:
        card = self.card_frames.get(product)
        if card:
            card.configure(
                bg=CARD_ACTIVE if is_on else CARD_BG,
                highlightbackground=CARD_BORDER_ON if is_on else CARD_BORDER,
            )
            for child in card.winfo_children():
                if isinstance(child, tk.Frame):
                    child.configure(bg=card["bg"])
                    for sub in child.winfo_children():
                        if isinstance(sub, tk.Label):
                            sub.configure(bg=card["bg"])
                        elif isinstance(sub, tk.Canvas):
                            sub.configure(bg=card["bg"])

        btn = self.toggle_buttons.get(product)
        if btn:
            btn.configure(
                text="ON" if is_on else "OFF",
                bg=ACCENT if is_on else "#282a36",
                fg="#ffffff" if is_on else TEXT_MUTED,
            )

        lbl = self.shade_labels.get(product)
        if lbl:
            lbl.configure(fg=ACCENT_HOVER if is_on else TEXT_MUTED)

    def _select_swatch(self, product: str, index: int) -> None:
        swatches = SWATCHES.get(product, [])
        if index >= len(swatches):
            return
        name, rgb = swatches[index]
        self.selected_swatch[product] = index

        with self._settings_lock:
            state = getattr(self.settings, product)
            state.color_bgr = rgb_to_bgr(rgb)
            state.enabled = True

        self._update_card_visuals(product, True)

        for i, cv in enumerate(self.swatch_canvases.get(product, [])):
            hex_c = rgb_to_hex(swatches[i][1])
            self._draw_swatch_circle(cv, hex_c, selected=(i == index))

        chip = self.custom_color_chips.get(product)
        if chip:
            self._draw_swatch_circle(chip, rgb_to_hex(rgb), selected=False)

        lbl = self.shade_labels.get(product)
        if lbl:
            lbl.configure(text=f"● {name}")

        self.status_text.set(f"{product.title()}: {name}")

    def _choose_custom_color(self, product: str, title: str) -> None:
        state = getattr(self.settings, product)
        cur_rgb = bgr_to_rgb(state.color_bgr)
        cur_hex = rgb_to_hex(cur_rgb)
        color = colorchooser.askcolor(color=cur_hex, title=f"Choose {title} Shade")
        if color and color[0]:
            rgb = (int(color[0][0]), int(color[0][1]), int(color[0][2]))
            with self._settings_lock:
                state.color_bgr = rgb_to_bgr(rgb)
                state.enabled = True

            self.selected_swatch[product] = None
            self._update_card_visuals(product, True)

            for i, cv in enumerate(self.swatch_canvases.get(product, [])):
                hex_c = rgb_to_hex(SWATCHES[product][i][1])
                self._draw_swatch_circle(cv, hex_c, selected=False)

            hex_val = color[1]
            chip = self.custom_color_chips.get(product)
            if chip:
                self._draw_swatch_circle(chip, hex_val, selected=True)

            lbl = self.shade_labels.get(product)
            if lbl:
                lbl.configure(text=f"● Custom {hex_val}")

            self.status_text.set(f"{title}: Custom {hex_val}")

    def _on_slider_change(self, product: str, value: str, label: tk.Label) -> None:
        val_int = int(float(value))
        label.configure(text=f"{val_int}%")
        with self._settings_lock:
            getattr(self.settings, product).intensity = val_int / 100.0

    def _switch_category(self, category: str) -> None:
        self.selected_category.set(category)
        for cat, btn in self.tab_buttons.items():
            if cat == category:
                btn.configure(bg=ACCENT, fg="#ffffff")
            else:
                btn.configure(bg=CARD_BG, fg=TEXT_SECONDARY)

        product_categories = {
            "foundation": "Face",
            "blush": "Face",
            "eyeshadow": "Eyes",
            "eyeliner": "Eyes",
            "brow": "Eyes",
            "lipstick": "Lips",
        }

        for prod, card in self.card_frames.items():
            prod_cat = product_categories.get(prod)
            if category == "All" or prod_cat == category:
                card.pack(fill=tk.X, pady=4, padx=2)
            else:
                card.pack_forget()

    def _apply_preset(self, preset_name: str) -> None:
        preset = PRESET_LOOKS.get(preset_name)
        if not preset:
            return

        with self._settings_lock:
            for prod, cfg in preset.items():
                state = getattr(self.settings, prod)
                state.enabled = cfg.get("enabled", False)
                if "intensity" in cfg:
                    state.intensity = cfg["intensity"]
                if "swatch" in cfg:
                    s_idx = cfg["swatch"]
                    self.selected_swatch[prod] = s_idx
                    rgb = SWATCHES[prod][s_idx][1]
                    state.color_bgr = rgb_to_bgr(rgb)
                if "style" in cfg:
                    state.style = cfg["style"]
                if "blend_mode" in cfg:
                    state.blend_mode = cfg["blend_mode"]

        for prod in ["lipstick", "blush", "eyeshadow", "eyeliner", "brow", "foundation"]:
            state = getattr(self.settings, prod)
            self._update_card_visuals(prod, state.enabled)
            cur_idx = self.selected_swatch.get(prod)
            swatches = SWATCHES.get(prod, [])
            for i, cv in enumerate(self.swatch_canvases.get(prod, [])):
                hex_c = rgb_to_hex(swatches[i][1])
                self._draw_swatch_circle(cv, hex_c, selected=(i == cur_idx))
            cur_name = swatches[cur_idx][0] if cur_idx is not None and cur_idx < len(swatches) else "Custom"
            lbl = self.shade_labels.get(prod)
            if lbl:
                lbl.configure(text=f"● {cur_name}")

        self.status_text.set(f"Look Applied: {preset_name}")

    def _set_compare(self, compare_original: bool) -> None:
        with self._settings_lock:
            self.is_showing_original = compare_original
        if compare_original:
            self.btn_compare.configure(bg=ACCENT, fg="#ffffff")
            self.status_text.set("Showing Original Face")
        else:
            self.btn_compare.configure(bg=CARD_BG, fg=TEXT_PRIMARY)
            self.status_text.set("Showing Virtual Try-On Look")

    def _toggle_studio_light(self) -> None:
        with self._settings_lock:
            self.is_studio_light_on = not self.is_studio_light_on
            val = self.is_studio_light_on
        if val:
            self.btn_light.configure(text="✦ Studio Light: ON", fg=ACCENT_HOVER, bg="#1d202b")
            self.status_text.set("Studio Auto-Lighting Enabled")
        else:
            self.btn_light.configure(text="✧ Studio Light: OFF", fg=TEXT_MUTED, bg=CARD_BG)
            self.status_text.set("Studio Auto-Lighting Disabled")

    def _toggle_mirror(self) -> None:
        with self._settings_lock:
            self.is_mirrored = not self.is_mirrored

    def start_camera(self) -> None:
        self.status_text.set("Opening webcam stream...")
        ok = self.camera.start()
        if not ok:
            self.status_text.set("Camera not found · Connect camera to begin")
        else:
            self.status_text.set("Camera active · Live Try-On Ready")

    def save_result(self) -> None:
        with self._display_lock:
            if self._latest_processed_frame is None:
                messagebox.showinfo("Save Snapshot", "No active camera frame to save.")
                return
            frame_to_save = self._latest_processed_frame.copy()

        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")],
            initialfile="makeup_look.jpg",
            title="Save Makeup Look Snapshot",
        )
        if not path:
            return
        cv2.imwrite(path, frame_to_save)
        self.status_text.set(f"Saved snapshot to {Path(path).name}")

    # -----------------------------------------------------------------------
    # Background Processing Thread (Decoupled from UI)
    # -----------------------------------------------------------------------
    def _processing_worker(self) -> None:
        last_fps_time = time.perf_counter()
        frame_counter = 0

        while self._running:
            raw_frame = self.camera.read_latest()
            if raw_frame is None:
                time.sleep(0.005)
                continue

            with self._settings_lock:
                mirror_active = self.is_mirrored
                light_active = self.is_studio_light_on
                compare_active = self.is_showing_original
                settings_copy = copy.deepcopy(self.settings)

            if mirror_active:
                raw_frame = cv2.flip(raw_frame, 1)

            # Apply Studio Auto-Lighting if enabled (Google Meet look)
            if light_active:
                work_frame = enhance_studio_lighting(raw_frame)
            else:
                work_frame = raw_frame

            # Detect face and apply virtual makeup
            if compare_active:
                processed = work_frame
                has_face = False
            else:
                landmarks = self.detector.detect_bgr(work_frame)
                has_face = bool(landmarks)
                if has_face:
                    processed = apply_makeup(work_frame, landmarks, settings_copy)
                else:
                    processed = work_frame

            frame_counter += 1
            now = time.perf_counter()
            elapsed = now - last_fps_time
            if elapsed >= 0.5:
                self._pipeline_fps = frame_counter / elapsed
                frame_counter = 0
                last_fps_time = now

            with self._display_lock:
                self._latest_processed_frame = processed
                self._latest_raw_frame = raw_frame
                self._face_detected = has_face

    # -----------------------------------------------------------------------
    # 60 FPS UI Render Loop (Main Thread)
    # -----------------------------------------------------------------------
    def _render_loop(self) -> None:
        if not self._running:
            return

        with self._display_lock:
            frame = self._latest_processed_frame
            has_face = self._face_detected
            fps = self._pipeline_fps

        if frame is not None:
            # Update HUD badges
            self.fps_badge_text.set(f"{fps:.0f} FPS · HD 720p")
            if has_face:
                self.tracking_badge_text.set("● Face Locked · Real-Time")
            else:
                self.tracking_badge_text.set("○ Searching for face...")

            # Fast SIMD resize in OpenCV directly (6x faster than Pillow resize)
            box_w = max(self.preview_container.winfo_width() - 4, 320)
            box_h = max(self.preview_container.winfo_height() - 4, 240)
            img_h, img_w = frame.shape[:2]

            ratio = min(box_w / img_w, box_h / img_h)
            new_w = max(1, int(img_w * ratio))
            new_h = max(1, int(img_h * ratio))

            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            self.photo_image = ImageTk.PhotoImage(image=pil_image)
            self.preview.configure(image=self.photo_image, text="")

        self.root.after(16, self._render_loop)

    def on_close(self) -> None:
        self._running = False
        self.camera.stop()
        if hasattr(self, "_worker_thread") and self._worker_thread is not None:
            self._worker_thread.join(timeout=0.5)
        try:
            self.detector.close()
        except Exception:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = MakeupTryOnApp()
    app.run()