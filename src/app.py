"""Virtual Makeup Studio — Professional Real-Time Virtual Try-On Desktop Application.

Redesigned following commercial luxury beauty tech aesthetic (Image 3):
- Clear visual hierarchy, generous spacing, restrained use of color, and uncluttered layout.
- Media Foundation (CAP_MSMF) priority without forced manual exposure flags, plus
  adaptive exposure normalization and an interactive Brightness slider so the feed
  matches the native Windows Camera App perfectly.
- Clean, unobstructed camera viewport (no split-lines or overlaid sliders on the face).
- Beauty features pill buttons, circular cosmetic pans with 'None' (⊘) toggle,
  finish options, coverage slider, camera brightness control, and high-quality UI icons.
- Decoupled multi-threaded 30+ FPS capture and 60 FPS responsive UI pipeline.
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
# Clean Modern Studio Theme Palette (White & Slate with Luxury Crimson Accent)
# ---------------------------------------------------------------------------
BG = "#ffffff"             # Pure crisp white background
SURFACE = "#fafbfc"        # Soft elevated studio surface
SURFACE_CARD = "#ffffff"   # Card background
BORDER_LIGHT = "#e8ecf2"   # Clean hairline border
BORDER_SUBTLE = "#f1f5f9"  # Divider line

TEXT_PRIMARY = "#0f172a"   # Deep bold charcoal
TEXT_SECONDARY = "#475569" # Sophisticated slate
TEXT_MUTED = "#94a3b8"     # Subtle placeholder slate

ACCENT = "#c41e3a"         # Luxury cosmetic crimson / cherry (matches reference)
ACCENT_HOVER = "#b91c1c"   # Rich deep crimson
ACCENT_SOFT = "#fdf2f4"    # Gentle rose blush tint
ACCENT_BORDER = "#fca5a5"  # Rose accent border

DARK_PILL_BG = "#0f172a"   # Active pill background (deep charcoal/black)
DARK_PILL_FG = "#ffffff"   # Active pill text

VIEWPORT_BG = "#f1f5f9"    # High-key neutral camera container backdrop
LIVE_EMERALD = "#10b981"   # Glowing status indicator green


# ---------------------------------------------------------------------------
# Adaptive Camera Tone & Brightness Balancer
# ---------------------------------------------------------------------------
def apply_camera_tone_and_brightness(frame: np.ndarray, user_brightness: float = 1.0, auto_balance: bool = True) -> np.ndarray:
    """
    Ensures camera brightness and color quality matches the Windows Camera App.
    If the device camera delivers a dark frame, smoothly lifts midtones and shadows
    without clipping highlights. If already bright (e.g. friend's laptop), leaves it natural.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray))

    # Calculate adaptive gamma lift if camera is underexposed
    gamma = 1.0
    if auto_balance and mean_lum < 118:
        gamma = np.clip(np.log(128.0 / 255.0) / np.log(max(10.0, mean_lum) / 255.0), 0.70, 1.0)

    # Combine with user-selected brightness multiplier
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        val = ((i / 255.0) ** gamma) * 255.0 * user_brightness
        # Soft-knee highlight compression to prevent clipping bright background walls
        if val > 215.0:
            over = val - 215.0
            val = val - (over ** 2) / (2.0 * max(1.0, 255.0 * user_brightness - 215.0))
        lut[i] = int(np.clip(val, 0, 255))

    return cv2.LUT(frame, lut)


# ---------------------------------------------------------------------------
# Curated 1-Click Look Presets
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
# Background Non-Blocking Camera Stream (Media Foundation Priority)
# ---------------------------------------------------------------------------
class CameraStream:
    """Dedicated background capture thread that prioritizes Windows Media Foundation."""

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
        # 1. Prioritize Microsoft Media Foundation (CAP_MSMF) — same backend as Windows Camera App
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            # 2. Fall back to DirectShow
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.src)

        if not self.cap.isOpened():
            self.cap = None
            self._is_active = False
            return False

        # Request 720p at 30 FPS without breaking hardware auto-exposure flags
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
# Main Virtual Makeup Studio Application
# ---------------------------------------------------------------------------
class MakeupTryOnApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Virtual Makeup Try-On")
        self.root.configure(bg=BG)
        self.root.minsize(1160, 740)
        self.root.geometry("1340x820")

        # Capture and Threading State
        self.camera = CameraStream(src=0)
        self.detector = FaceDetector(video=True)
        self._running = True

        # Thread-safe state primitives (Python-level, no Tk calls from worker)
        self._settings_lock = threading.Lock()
        self.is_mirrored = True
        self.is_showing_original = False
        self.is_studio_light_on = True
        self.camera_brightness_factor = 1.08  # Default calibrated boost for natural camera exposure

        # Latest Rendered Frames for Display and Snapshot
        self._display_lock = threading.Lock()
        self._latest_processed_frame: np.ndarray | None = None
        self._latest_raw_frame: np.ndarray | None = None
        self._face_detected = False
        self._pipeline_fps = 0.0

        # Active Product Selected in UI (Lipstick by default)
        self.active_feature = "lipstick"

        # UI String/Control Variables
        self.status_text = tk.StringVar(value="Camera active · Live Try-On Ready")
        self.fps_badge_text = tk.StringVar(value="-- FPS")
        self.tracking_badge_text = tk.StringVar(value="● Face Tracking Active")
        self.color_title_text = tk.StringVar(value="Color · Classic Red")

        # Color Swatches Tracking
        self.swatch_canvases: dict[str, list[tk.Canvas]] = {}
        self.none_canvases: dict[str, tk.Canvas] = {}
        self.feature_buttons: dict[str, tk.Button] = {}
        self.finish_buttons: dict[str, dict[str, tk.Button]] = {}
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
        style.configure("TLabel", background=BG, foreground=TEXT_PRIMARY, font=("Segoe UI", 9))

    def _build_ui(self) -> None:
        # -------------------------------------------------------------------
        # 1. Top Minimalist Header (Matches Image 3)
        # -------------------------------------------------------------------
        header = tk.Frame(self.root, bg=BG, padx=28, pady=16)
        header.pack(fill=tk.X)

        title_lbl = tk.Label(
            header,
            text="VIRTUAL MAKEUP TRY-ON",
            font=("Segoe UI Black", 12),
            bg=BG,
            fg=TEXT_PRIMARY,
        )
        title_lbl.pack(side=tk.LEFT)

        status_box = tk.Frame(header, bg=BG)
        status_box.pack(side=tk.RIGHT)

        status_dot = tk.Canvas(status_box, width=8, height=8, bg=BG, highlightthickness=0)
        status_dot.pack(side=tk.LEFT, padx=(0, 6))
        status_dot.create_oval(1, 1, 7, 7, fill=LIVE_EMERALD, outline="")

        self.status_lbl = tk.Label(
            status_box,
            textvariable=self.status_text,
            font=("Segoe UI", 9),
            bg=BG,
            fg=TEXT_SECONDARY,
        )
        self.status_lbl.pack(side=tk.LEFT)

        # Subtle hairline separator
        tk.Frame(self.root, height=1, bg=BORDER_LIGHT).pack(fill=tk.X)

        # -------------------------------------------------------------------
        # 2. Main Workspace Layout (Two Columns: Viewport Left, Controls Right)
        # -------------------------------------------------------------------
        workspace = tk.Frame(self.root, bg=BG, padx=28, pady=20)
        workspace.pack(fill=tk.BOTH, expand=True)
        workspace.columnconfigure(0, weight=1)
        workspace.columnconfigure(1, weight=0, minsize=490)
        workspace.rowconfigure(0, weight=1)

        # -------------------------------------------------------------------
        # Left Side: Camera Hero Viewport (Clean, Uncluttered, No Split Line or Sliders)
        # -------------------------------------------------------------------
        viewport_wrapper = tk.Frame(workspace, bg=BG)
        viewport_wrapper.grid(row=0, column=0, sticky="nsew", padx=(0, 24))
        viewport_wrapper.rowconfigure(0, weight=1)
        viewport_wrapper.columnconfigure(0, weight=1)

        self.preview_container = tk.Frame(
            viewport_wrapper,
            bg=VIEWPORT_BG,
            highlightthickness=1,
            highlightbackground=BORDER_LIGHT,
        )
        self.preview_container.grid(row=0, column=0, sticky="nsew")
        self.preview_container.pack_propagate(False)

        # Clean Video Preview Display Label (Hero Center)
        self.preview = tk.Label(
            self.preview_container,
            bg=VIEWPORT_BG,
            fg=TEXT_MUTED,
            text="Connecting studio camera stream...",
            font=("Segoe UI", 10),
        )
        self.preview.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Floating Unobtrusive Status Badge (Top-Left of Viewport)
        status_pill = tk.Frame(self.preview_container, bg="#ffffff", highlightthickness=1, highlightbackground=BORDER_LIGHT, padx=10, pady=4)
        status_pill.place(relx=0.03, rely=0.03)

        dot_cv = tk.Canvas(status_pill, width=7, height=7, bg="#ffffff", highlightthickness=0)
        dot_cv.pack(side=tk.LEFT, padx=(0, 5))
        dot_cv.create_oval(1, 1, 6, 6, fill=LIVE_EMERALD, outline="")

        tk.Label(status_pill, textvariable=self.tracking_badge_text, font=("Segoe UI Semibold", 8), bg="#ffffff", fg=TEXT_PRIMARY).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(status_pill, textvariable=self.fps_badge_text, font=("Segoe UI", 8), bg="#ffffff", fg=TEXT_MUTED).pack(side=tk.LEFT)

        # Floating Bottom Action Toolbar Under Viewport
        hud_bar = tk.Frame(viewport_wrapper, bg=BG)
        hud_bar.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        # Bare Face Compare Button (Interactive Press & Hold)
        self.btn_compare = tk.Button(
            hud_bar,
            text="👁  Hold for Bare Face",
            font=("Segoe UI Semibold", 9),
            bg=SURFACE,
            fg=TEXT_PRIMARY,
            activebackground=BORDER_LIGHT,
            activeforeground=TEXT_PRIMARY,
            relief="flat",
            padx=12,
            pady=6,
            cursor="hand2",
            highlightthickness=1,
            highlightbackground=BORDER_LIGHT,
        )
        self.btn_compare.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_compare.bind("<ButtonPress-1>", lambda _e: self._set_compare(True))
        self.btn_compare.bind("<ButtonRelease-1>", lambda _e: self._set_compare(False))

        # Auto-Lighting Toggle
        self.btn_light = tk.Button(
            hud_bar,
            text="✦  Auto-Lighting: ON",
            font=("Segoe UI", 9),
            bg=SURFACE,
            fg=TEXT_PRIMARY,
            activebackground=BORDER_LIGHT,
            relief="flat",
            padx=12,
            pady=6,
            cursor="hand2",
            highlightthickness=1,
            highlightbackground=BORDER_LIGHT,
            command=self._toggle_studio_light,
        )
        self.btn_light.pack(side=tk.LEFT, padx=(0, 8))

        # Mirror Camera Flip
        btn_mirror = tk.Button(
            hud_bar,
            text="⇄  Mirror Camera",
            font=("Segoe UI", 9),
            bg=SURFACE,
            fg=TEXT_SECONDARY,
            activebackground=BORDER_LIGHT,
            relief="flat",
            padx=10,
            pady=6,
            cursor="hand2",
            highlightthickness=1,
            highlightbackground=BORDER_LIGHT,
            command=self._toggle_mirror,
        )
        btn_mirror.pack(side=tk.LEFT)

        # Quick Snapshot Button
        btn_quick_snap = tk.Button(
            hud_bar,
            text="📸  Quick Snap",
            font=("Segoe UI Semibold", 9),
            bg=SURFACE,
            fg=ACCENT,
            activebackground=ACCENT_SOFT,
            activeforeground=ACCENT,
            relief="flat",
            padx=12,
            pady=6,
            cursor="hand2",
            highlightthickness=1,
            highlightbackground=ACCENT_BORDER,
            command=self.save_result,
        )
        btn_quick_snap.pack(side=tk.RIGHT)

        # -------------------------------------------------------------------
        # Right Side: Clean Commercial Beauty Studio (Matches Image 3)
        # -------------------------------------------------------------------
        sidebar = tk.Frame(workspace, bg=SURFACE, padx=22, pady=18, highlightthickness=1, highlightbackground=BORDER_LIGHT)
        sidebar.grid(row=0, column=1, sticky="nsew")

        # Section 1: "Beauty features" (Horizontal Pill Buttons, Matches Image 3)
        tk.Label(
            sidebar,
            text="Beauty features",
            font=("Segoe UI Semibold", 10),
            bg=SURFACE,
            fg=TEXT_PRIMARY,
        ).pack(anchor=tk.W, pady=(0, 8))

        features_box = tk.Frame(sidebar, bg=SURFACE)
        features_box.pack(fill=tk.X, pady=(0, 16))

        features = [
            ("lipstick", "Lipstick"),
            ("blush", "Blush"),
            ("eyeshadow", "Eyeshadow"),
            ("eyeliner", "Eyeliner"),
            ("brow", "Eyebrows"),
            ("foundation", "Foundation"),
        ]

        # 2 rows of 3 pills each for clean generous spacing
        for i, (f_key, f_label) in enumerate(features):
            is_active = f_key == self.active_feature
            state = getattr(self.settings, f_key)
            check_mark = "✓ " if state.enabled else ""

            p_btn = tk.Button(
                features_box,
                text=f"{check_mark}{f_label}",
                font=("Segoe UI Semibold", 8),
                bg=DARK_PILL_BG if is_active else BG,
                fg=DARK_PILL_FG if is_active else TEXT_SECONDARY,
                activebackground=DARK_PILL_BG,
                activeforeground=DARK_PILL_FG,
                relief="flat",
                padx=10,
                pady=6,
                cursor="hand2",
                highlightthickness=1,
                highlightbackground=DARK_PILL_BG if is_active else BORDER_LIGHT,
                command=lambda k=f_key: self._select_feature(k),
            )
            p_btn.grid(row=i // 3, column=i % 3, padx=3, pady=3, sticky="ew")
            features_box.columnconfigure(i % 3, weight=1)
            self.feature_buttons[f_key] = p_btn

        # Section 2: "Color" (Matches Image 3: None/Slash button ⊘ + Circle Swatches)
        color_header_row = tk.Frame(sidebar, bg=SURFACE)
        color_header_row.pack(fill=tk.X, pady=(4, 6))

        tk.Label(
            color_header_row,
            text="Color",
            font=("Segoe UI Semibold", 10),
            bg=SURFACE,
            fg=TEXT_PRIMARY,
        ).pack(side=tk.LEFT)

        self.shade_badge = tk.Label(
            color_header_row,
            textvariable=self.color_title_text,
            font=("Segoe UI", 8),
            bg=SURFACE,
            fg=TEXT_MUTED,
        )
        self.shade_badge.pack(side=tk.RIGHT)

        # Swatches Frame Container
        self.swatch_container = tk.Frame(sidebar, bg=SURFACE)
        self.swatch_container.pack(fill=tk.X, pady=(0, 14))

        # Build initial swatches for Lipstick
        self._rebuild_swatches()

        # Section 3: "Finish options" (Horizontal Pill Selectors, Matches Image 3)
        self.finish_header = tk.Label(
            sidebar,
            text="Finish options",
            font=("Segoe UI Semibold", 10),
            bg=SURFACE,
            fg=TEXT_PRIMARY,
        )
        self.finish_header.pack(anchor=tk.W, pady=(4, 6))

        self.finish_box = tk.Frame(sidebar, bg=SURFACE)
        self.finish_box.pack(fill=tk.X, pady=(0, 14))
        self._rebuild_finish_options()

        # Section 4: "Coverage & Intensity" Slider (In the control panel, not on camera)
        slider_frame = tk.Frame(sidebar, bg=SURFACE)
        slider_frame.pack(fill=tk.X, pady=(0, 14))

        slider_lbl_row = tk.Frame(slider_frame, bg=SURFACE)
        slider_lbl_row.pack(fill=tk.X)

        tk.Label(slider_lbl_row, text="Coverage & Intensity", font=("Segoe UI Semibold", 9), bg=SURFACE, fg=TEXT_PRIMARY).pack(side=tk.LEFT)
        self.intensity_pct_label = tk.Label(slider_lbl_row, text="60%", font=("Segoe UI Semibold", 9), bg=SURFACE, fg=ACCENT)
        self.intensity_pct_label.pack(side=tk.RIGHT)

        self.inline_scale = tk.Scale(
            slider_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            bg=SURFACE,
            fg=TEXT_SECONDARY,
            highlightthickness=0,
            troughcolor="#e2e8f0",
            activebackground=ACCENT,
            showvalue=False,
            command=self._on_inline_slider_change,
        )
        self.inline_scale.set(60)
        self.inline_scale.pack(fill=tk.X, pady=(3, 0))

        # Section 5: "Camera Brightness" Slider (Solves the laptop camera underexposure)
        bright_frame = tk.Frame(sidebar, bg=SURFACE)
        bright_frame.pack(fill=tk.X, pady=(0, 16))

        bright_lbl_row = tk.Frame(bright_frame, bg=SURFACE)
        bright_lbl_row.pack(fill=tk.X)

        tk.Label(bright_lbl_row, text="☀  Camera Exposure / Brightness", font=("Segoe UI Semibold", 9), bg=SURFACE, fg=TEXT_PRIMARY).pack(side=tk.LEFT)
        self.bright_pct_label = tk.Label(bright_lbl_row, text="108%", font=("Segoe UI Semibold", 9), bg=SURFACE, fg=TEXT_SECONDARY)
        self.bright_pct_label.pack(side=tk.RIGHT)

        self.bright_scale = tk.Scale(
            bright_frame,
            from_=75,
            to=145,
            orient=tk.HORIZONTAL,
            bg=SURFACE,
            fg=TEXT_SECONDARY,
            highlightthickness=0,
            troughcolor="#e2e8f0",
            activebackground=ACCENT,
            showvalue=False,
            command=self._on_brightness_slider_change,
        )
        self.bright_scale.set(108)
        self.bright_scale.pack(fill=tk.X, pady=(3, 0))

        # Section 6: Primary Action Button (Matches Image 3's Cherry Pill Button)
        self.btn_primary_save = tk.Button(
            sidebar,
            text="📸  Save Look Snapshot",
            font=("Segoe UI Semibold", 10),
            bg=ACCENT,
            fg="#ffffff",
            activebackground=ACCENT_HOVER,
            activeforeground="#ffffff",
            relief="flat",
            padx=18,
            pady=9,
            cursor="hand2",
            command=self.save_result,
        )
        self.btn_primary_save.pack(fill=tk.X, pady=(0, 16))

        # Section 7: "Recommended looks" (1-Click Presets)
        tk.Label(
            sidebar,
            text="Recommended looks",
            font=("Segoe UI Semibold", 9),
            bg=SURFACE,
            fg=TEXT_MUTED,
        ).pack(anchor=tk.W, pady=(0, 6))

        preset_row = tk.Frame(sidebar, bg=SURFACE)
        preset_row.pack(fill=tk.X, pady=(0, 8))

        for look_name in ["Natural Glow", "Red Carpet", "Berry Velvet", "Warm Sunset"]:
            p_btn = tk.Button(
                preset_row,
                text=look_name,
                font=("Segoe UI", 8),
                bg=BG,
                fg=TEXT_SECONDARY,
                activebackground=ACCENT_SOFT,
                activeforeground=ACCENT,
                relief="flat",
                padx=8,
                pady=4,
                cursor="hand2",
                highlightthickness=1,
                highlightbackground=BORDER_LIGHT,
                command=lambda ln=look_name: self._apply_preset(ln),
            )
            p_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # Clear All Button
        btn_reset_all = tk.Button(
            sidebar,
            text="↺  Clear All Makeup (Bare Face)",
            font=("Segoe UI", 8),
            bg=SURFACE,
            fg=TEXT_MUTED,
            activebackground=BORDER_LIGHT,
            relief="flat",
            padx=8,
            pady=4,
            cursor="hand2",
            command=lambda: self._apply_preset("Bare Face"),
        )
        btn_reset_all.pack(fill=tk.X)

    # -----------------------------------------------------------------------
    # Dynamic UI Sections (Beauty Features, Swatches, Finish Options)
    # -----------------------------------------------------------------------
    def _select_feature(self, feature_key: str) -> None:
        self.active_feature = feature_key

        # Update Beauty Features Pill Styles
        for key, btn in self.feature_buttons.items():
            is_active = key == feature_key
            state = getattr(self.settings, key)
            check_mark = "✓ " if state.enabled else ""
            label_text = btn.cget("text").lstrip("✓ ")
            btn.configure(
                text=f"{check_mark}{label_text}",
                bg=DARK_PILL_BG if is_active else BG,
                fg=DARK_PILL_FG if is_active else TEXT_SECONDARY,
                highlightbackground=DARK_PILL_BG if is_active else BORDER_LIGHT,
            )

        # Update Swatches, Finish Options, and Intensity Slider for this feature
        self._rebuild_swatches()
        self._rebuild_finish_options()

        state = getattr(self.settings, feature_key)
        val = int(state.intensity * 100)
        self.inline_scale.set(val)
        self.intensity_pct_label.configure(text=f"{val}%")

    def _rebuild_swatches(self) -> None:
        for child in self.swatch_container.winfo_children():
            child.destroy()

        feature_key = self.active_feature
        state = getattr(self.settings, feature_key)
        swatches = SWATCHES.get(feature_key, [])
        cur_idx = self.selected_swatch.get(feature_key)

        cur_name = swatches[cur_idx][0] if cur_idx is not None and cur_idx < len(swatches) else ("Custom" if state.enabled else "None")
        self.color_title_text.set(f"● {cur_name}")

        # 1. First Swatch: None / Off Button (⊘, Matches Image 3)
        none_cv = tk.Canvas(self.swatch_container, width=28, height=28, bg=SURFACE, highlightthickness=0, cursor="hand2")
        none_cv.grid(row=0, column=0, padx=4, pady=3)
        self._draw_none_swatch(none_cv, selected=(not state.enabled))
        none_cv.bind("<Button-1>", lambda _e, p=feature_key: self._disable_feature(p))
        none_cv.bind("<Enter>", lambda _e: self.status_text.set(f"{feature_key.title()}: Clear / None"))

        # 2. Preset Swatches (Circular pans with double-ring selection)
        self.swatch_canvases[feature_key] = []
        for i, (name, rgb) in enumerate(swatches):
            col_idx = i + 1
            r_idx = col_idx // 7
            c_idx = col_idx % 7

            c_hex = rgb_to_hex(rgb)
            is_sel = (i == cur_idx) and state.enabled

            cv = tk.Canvas(self.swatch_container, width=28, height=28, bg=SURFACE, highlightthickness=0, cursor="hand2")
            cv.grid(row=r_idx, column=c_idx, padx=4, pady=3)
            self._draw_swatch_circle(cv, c_hex, is_sel)

            cv.bind("<Button-1>", lambda _e, p=feature_key, idx=i: self._select_swatch(p, idx))
            cv.bind("<Enter>", lambda _e, n=name: self.status_text.set(f"{feature_key.title()}: {n}"))
            self.swatch_canvases[feature_key].append(cv)

        # 3. Custom Color Picker Button
        custom_idx = len(swatches) + 1
        custom_btn = tk.Button(
            self.swatch_container,
            text="🎨 Custom...",
            font=("Segoe UI", 8),
            bg=BG,
            fg=TEXT_SECONDARY,
            activebackground=ACCENT_SOFT,
            relief="flat",
            padx=8,
            pady=3,
            cursor="hand2",
            highlightthickness=1,
            highlightbackground=BORDER_LIGHT,
            command=lambda p=feature_key: self._choose_custom_color(p),
        )
        custom_btn.grid(row=(custom_idx // 7) + 1, column=0, columnspan=4, sticky="w", pady=(6, 0))

    def _rebuild_finish_options(self) -> None:
        for child in self.finish_box.winfo_children():
            child.destroy()

        feature_key = self.active_feature
        if feature_key == "lipstick":
            self.finish_header.configure(text="Finish options")
            options = [
                ("Velvet Matte", "multiply"),
                ("Satin Cream", "overlay"),
                ("Sheer Tint", "color"),
            ]
            cur_mode = getattr(self.settings.lipstick, "blend_mode", "multiply")
            for label, mode_key in options:
                is_sel = cur_mode == mode_key
                btn = tk.Button(
                    self.finish_box,
                    text=label,
                    font=("Segoe UI", 8),
                    bg=DARK_PILL_BG if is_sel else BG,
                    fg=DARK_PILL_FG if is_sel else TEXT_SECONDARY,
                    activebackground=DARK_PILL_BG,
                    activeforeground=DARK_PILL_FG,
                    relief="flat",
                    padx=10,
                    pady=4,
                    cursor="hand2",
                    highlightthickness=1,
                    highlightbackground=DARK_PILL_BG if is_sel else BORDER_LIGHT,
                    command=lambda m=mode_key: self._set_lipstick_finish(m),
                )
                btn.pack(side=tk.LEFT, padx=3)

        elif feature_key == "eyeliner":
            self.finish_header.configure(text="Style options")
            options = [
                ("Classic Wing", "classic_wing"),
                ("Tightline", "tightline"),
                ("Cat Eye", "cat_eye"),
                ("Bold Kohl", "bold"),
            ]
            cur_style = getattr(self.settings.eyeliner, "style", "classic_wing")
            for label, style_key in options:
                is_sel = cur_style == style_key
                btn = tk.Button(
                    self.finish_box,
                    text=label,
                    font=("Segoe UI", 8),
                    bg=DARK_PILL_BG if is_sel else BG,
                    fg=DARK_PILL_FG if is_sel else TEXT_SECONDARY,
                    activebackground=DARK_PILL_BG,
                    activeforeground=DARK_PILL_FG,
                    relief="flat",
                    padx=9,
                    pady=4,
                    cursor="hand2",
                    highlightthickness=1,
                    highlightbackground=DARK_PILL_BG if is_sel else BORDER_LIGHT,
                    command=lambda s=style_key: self._set_eyeliner_style(s),
                )
                btn.pack(side=tk.LEFT, padx=2)

        else:
            self.finish_header.configure(text="Application Mode")
            lbl = tk.Label(self.finish_box, text="Natural Blend · Seamless Falloff", font=("Segoe UI", 8), bg=SURFACE, fg=TEXT_MUTED)
            lbl.pack(side=tk.LEFT, padx=2)

    def _draw_swatch_circle(self, canvas: tk.Canvas, hex_color: str, selected: bool) -> None:
        canvas.delete("all")
        w, h = int(canvas["width"]), int(canvas["height"])
        if selected:
            # Elegant outer halo ring
            canvas.create_oval(1, 1, w - 1, h - 1, outline=ACCENT, width=2)
            # White separator ring
            canvas.create_oval(3, 3, w - 3, h - 3, fill="#ffffff", outline="")
            # Inner cosmetic color circle
            canvas.create_oval(5, 5, w - 5, h - 5, fill=hex_color, outline=BORDER_LIGHT, width=1)
        else:
            canvas.create_oval(3, 3, w - 3, h - 3, fill=hex_color, outline=BORDER_LIGHT, width=1)

    def _draw_none_swatch(self, canvas: tk.Canvas, selected: bool) -> None:
        """Draws the clean circular 'None' button with diagonal slash (⊘, Image 3)."""
        canvas.delete("all")
        w, h = int(canvas["width"]), int(canvas["height"])
        if selected:
            canvas.create_oval(1, 1, w - 1, h - 1, outline=ACCENT, width=2)
            canvas.create_oval(4, 4, w - 4, h - 4, fill="#ffffff", outline=TEXT_MUTED, width=1)
            canvas.create_line(7, h - 8, w - 8, 7, fill=ACCENT, width=2)
        else:
            canvas.create_oval(3, 3, w - 3, h - 3, fill="#ffffff", outline=BORDER_LIGHT, width=1)
            canvas.create_line(7, h - 8, w - 8, 7, fill=TEXT_MUTED, width=1)

    # -----------------------------------------------------------------------
    # Interactive Event Handlers
    # -----------------------------------------------------------------------
    def _disable_feature(self, product: str) -> None:
        with self._settings_lock:
            getattr(self.settings, product).enabled = False

        self._update_feature_pill(product, False)
        self._rebuild_swatches()
        self.status_text.set(f"{product.title()}: Disabled")

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

        self._update_feature_pill(product, True)
        self._rebuild_swatches()
        self.status_text.set(f"{product.title()}: {name}")

    def _choose_custom_color(self, product: str) -> None:
        state = getattr(self.settings, product)
        cur_rgb = bgr_to_rgb(state.color_bgr)
        cur_hex = rgb_to_hex(cur_rgb)
        color = colorchooser.askcolor(color=cur_hex, title=f"Choose {product.title()} Shade")
        if color and color[0]:
            rgb = (int(color[0][0]), int(color[0][1]), int(color[0][2]))
            with self._settings_lock:
                state.color_bgr = rgb_to_bgr(rgb)
                state.enabled = True

            self.selected_swatch[product] = None
            self._update_feature_pill(product, True)
            self._rebuild_swatches()
            self.color_title_text.set(f"● Custom {color[1]}")
            self.status_text.set(f"{product.title()}: Custom Shade {color[1]}")

    def _set_lipstick_finish(self, mode: str) -> None:
        with self._settings_lock:
            self.settings.lipstick.blend_mode = mode
        self._rebuild_finish_options()

    def _set_eyeliner_style(self, style: str) -> None:
        with self._settings_lock:
            self.settings.eyeliner.style = style
        self._rebuild_finish_options()

    def _on_inline_slider_change(self, value: str) -> None:
        val_int = int(float(value))
        self.intensity_pct_label.configure(text=f"{val_int}%")
        with self._settings_lock:
            getattr(self.settings, self.active_feature).intensity = val_int / 100.0

    def _on_brightness_slider_change(self, value: str) -> None:
        val_int = int(float(value))
        self.bright_pct_label.configure(text=f"{val_int}%")
        with self._settings_lock:
            self.camera_brightness_factor = val_int / 100.0

    def _update_feature_pill(self, product: str, is_on: bool) -> None:
        btn = self.feature_buttons.get(product)
        if btn:
            label_text = btn.cget("text").lstrip("✓ ")
            check_mark = "✓ " if is_on else ""
            is_active = product == self.active_feature
            btn.configure(
                text=f"{check_mark}{label_text}",
                bg=DARK_PILL_BG if is_active else BG,
                fg=DARK_PILL_FG if is_active else TEXT_SECONDARY,
                highlightbackground=DARK_PILL_BG if is_active else BORDER_LIGHT,
            )

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
            self._update_feature_pill(prod, getattr(self.settings, prod).enabled)

        self._select_feature(self.active_feature)
        self.status_text.set(f"Look Applied: {preset_name}")

    def _set_compare(self, compare_original: bool) -> None:
        with self._settings_lock:
            self.is_showing_original = compare_original
        self.status_text.set("Showing Bare Face" if compare_original else "Showing Try-On Look")

    def _toggle_studio_light(self) -> None:
        with self._settings_lock:
            self.is_studio_light_on = not self.is_studio_light_on
            val = self.is_studio_light_on
        self.btn_light.configure(
            text="✦  Auto-Lighting: ON" if val else "✧  Auto-Lighting: OFF",
            fg=TEXT_PRIMARY if val else TEXT_MUTED,
        )
        self.status_text.set("Studio Auto-Lighting: " + ("ON" if val else "OFF"))

    def _toggle_mirror(self) -> None:
        with self._settings_lock:
            self.is_mirrored = not self.is_mirrored

    def start_camera(self) -> None:
        self.status_text.set("Opening camera stream...")
        ok = self.camera.start()
        if not ok:
            self.status_text.set("Camera not detected · Connect camera to begin")
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
                user_brightness = self.camera_brightness_factor
                compare_active = self.is_showing_original
                settings_copy = copy.deepcopy(self.settings)

            if mirror_active:
                raw_frame = cv2.flip(raw_frame, 1)

            # Apply adaptive camera brightness and tone calibration to match Windows Camera App
            if light_active or user_brightness != 1.0:
                work_frame = apply_camera_tone_and_brightness(raw_frame, user_brightness, light_active)
            else:
                work_frame = raw_frame

            # Detect face and apply virtual makeup (clean full frame, no split line or slider on face)
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
            self.fps_badge_text.set(f"{fps:.0f} FPS · HD 720p")
            if has_face:
                self.tracking_badge_text.set("● Face Tracking Active")
            else:
                self.tracking_badge_text.set("○ Searching for face...")

            # Fast SIMD resize in OpenCV directly
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
