"""Desktop GUI: live virtual makeup try-on with professional studio aesthetic."""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# Attempt Windows High-DPI awareness so fonts and UI are razor-sharp
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

# Sleek Professional Dark Studio Palette
BG = "#101014"              # Deep studio charcoal
PANEL = "#16161c"           # Side panel charcoal
CARD = "#1e1e26"            # Elevated card surface
CARD_HOVER = "#262630"      # Elevated hover state
ACCENT = "#e05368"          # Refined studio rose-coral
ACCENT_HOVER = "#eb677b"    # Active accent
ACCENT_LIGHT = "#2a1c22"    # Subtle accent tint
TEXT = "#f0f0f5"            # Crisp off-white text
MUTED = "#8e8e9c"           # Secondary muted text
BORDER = "#2b2b36"          # Subtle 1px studio border
BORDER_ACTIVE = "#e05368"   # Active selection border
PREVIEW_BG = "#08080a"      # High-contrast viewport background


class MakeupTryOnApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Virtual Makeup Studio")
        self.root.configure(bg=BG)
        self.root.minsize(1060, 680)
        self.root.geometry("1280x780")

        self.cap = None
        self.video_detector = FaceDetector(video=True)

        self.current_raw = None
        self.current_landmarks = None
        self.current_processed = None
        self.photo_image = None

        self.mirror = tk.BooleanVar(value=True)
        self.show_original = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Initializing camera...")
        self.fps_text = tk.StringVar(value="0 FPS")
        self._last_fps_time = time.perf_counter()
        self._frame_count = 0
        self._running = True

        self.swatch_buttons: dict[str, list[tk.Canvas]] = {}
        self.custom_color_buttons: dict[str, tk.Button] = {}
        self.selected_swatch: dict[str, int | None] = {
            "foundation": 1,
            "blush": 1,
            "eyeshadow": 0,
            "eyeliner": 0,
            "brow": 0,
            "lipstick": 0,
        }

        # Initialize Professional Studio Makeup Settings
        self.settings = MakeupSettings(
            foundation=ProductState(
                enabled=False,
                color_bgr=rgb_to_bgr(SWATCHES["foundation"][1][1]),
                intensity=0.28,
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
                blend_mode="multiply",  # Velvety matte finish
            ),
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

        style.configure("TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 15))
        style.configure("Sub.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 8))
        style.configure("Header.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI Semibold", 10))
        style.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 8))
        style.configure("Status.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))

        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="#ffffff",
            font=("Segoe UI Semibold", 9),
            padding=(12, 6),
            borderwidth=0,
        )
        style.map("Accent.TButton", background=[("active", ACCENT_HOVER)])

        style.configure(
            "Ghost.TButton",
            background=CARD,
            foreground=TEXT,
            font=("Segoe UI Semibold", 9),
            padding=(12, 6),
            borderwidth=1,
            relief="solid",
        )
        style.map("Ghost.TButton", background=[("active", CARD_HOVER)])

        style.configure("TCheckbutton", background=CARD, foreground=TEXT, font=("Segoe UI Semibold", 9))
        style.configure("Bar.TCheckbutton", background=BG, foreground=TEXT, font=("Segoe UI", 9))
        style.configure("TCombobox", fieldbackground=CARD, background=BORDER, foreground=TEXT)

    def _build_ui(self) -> None:
        # Professional Header Bar
        header = ttk.Frame(self.root, style="TFrame")
        header.pack(fill=tk.X, padx=20, pady=(10, 6))

        title_box = tk.Frame(header, bg=BG)
        title_box.pack(anchor=tk.W)


        ttk.Label(title_box, text="Virtual Makeup Studio", style="Title.TLabel").pack(side=tk.LEFT)

        ttk.Label(
            header,
            style="Sub.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

        # Main Body Grid: Responsive & non-jumping
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=6)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0, minsize=370)
        body.rowconfigure(0, weight=1)

        # Left: Preview container
        self.preview_container = tk.Frame(
            body,
            bg=PREVIEW_BG,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.preview_container.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.preview_container.pack_propagate(False)

        self.preview = tk.Label(self.preview_container, bg=PREVIEW_BG, fg=MUTED, text="Initializing camera...")
        self.preview.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Right: Scrollable Controls Sidebar (Fixed width, clean layout)
        sidebar_frame = tk.Frame(
            body,
            bg=PANEL,
            width=370,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        sidebar_frame.grid(row=0, column=1, sticky="nsew")
        sidebar_frame.pack_propagate(False)

        self.canvas = tk.Canvas(sidebar_frame, bg=PANEL, highlightthickness=0)
        scroll = ttk.Scrollbar(sidebar_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.sidebar_inner = tk.Frame(self.canvas, bg=PANEL)

        self.sidebar_inner.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas_window = self.canvas.create_window((0, 0), window=self.sidebar_inner, anchor=tk.NW, width=348)
        self.canvas.configure(yscrollcommand=scroll.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=8)

        # Mousewheel scrolling
        def _on_mousewheel(event):
            delta = int(-1 * (event.delta / 120)) if event.delta else 0
            self.canvas.yview_scroll(delta, "units")

        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Action Buttons (Live Camera & Save Snapshot)
        actions = tk.Frame(self.sidebar_inner, bg=PANEL)
        actions.pack(fill=tk.X, pady=(2, 6), padx=4)

        btn_row = tk.Frame(actions, bg=PANEL)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="Live Camera", style="Accent.TButton", command=self.start_camera).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4)
        )
        ttk.Button(btn_row, text="Save Snapshot", style="Ghost.TButton", command=self.save_result).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0)
        )

        # Clean Product Cards (No emojis, sleek layout)
        self._build_product_card("foundation", "Foundation", extra="foundation")
        self._build_product_card("blush", "Blush")
        self._build_product_card("eyeshadow", "Eyeshadow")
        self._build_product_card("eyeliner", "Eyeliner", extra="eyeliner")
        self._build_product_card("brow", "Eyebrows")
        self._build_product_card("lipstick", "Lipstick", extra="lipstick")

        # Footer
        footer = ttk.Frame(self.root, style="TFrame")
        footer.pack(fill=tk.X, padx=20, pady=(6, 8))
        ttk.Checkbutton(footer, text="Mirror Camera", variable=self.mirror, style="Bar.TCheckbutton").pack(
            side=tk.LEFT, padx=(0, 16)
        )
        ttk.Checkbutton(
            footer, text="Show Original", variable=self.show_original, style="Bar.TCheckbutton"
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(footer, textvariable=self.fps_text, style="Status.TLabel").pack(side=tk.RIGHT)
        ttk.Label(footer, textvariable=self.status, style="Status.TLabel").pack(side=tk.RIGHT, padx=16)

    def _build_product_card(self, key: str, title: str, extra: str | None = None) -> None:
        card = tk.Frame(
            self.sidebar_inner,
            bg=CARD,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        card.pack(fill=tk.X, pady=3, padx=4)

        top = tk.Frame(card, bg=CARD)
        top.pack(fill=tk.X, padx=10, pady=(6, 3))
        state = getattr(self.settings, key)
        enabled_var = tk.BooleanVar(value=state.enabled)

        def toggle() -> None:
            state.enabled = enabled_var.get()

        cb = ttk.Checkbutton(top, text=title, variable=enabled_var, command=toggle, style="TCheckbutton")
        cb.pack(side=tk.LEFT)

        # Extra Options for specific products
        if extra == "eyeliner":
            opt_row = tk.Frame(card, bg=CARD)
            opt_row.pack(fill=tk.X, padx=10, pady=(1, 3))
            ttk.Label(opt_row, text="Style:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 6))

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
                self.settings.eyeliner.style = style_map.get(style_var.get(), "classic_wing")

            style_combo.bind("<<ComboboxSelected>>", _on_style_change)

        elif extra == "lipstick":
            opt_row = tk.Frame(card, bg=CARD)
            opt_row.pack(fill=tk.X, padx=10, pady=(1, 3))
            ttk.Label(opt_row, text="Finish:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 6))

            mode_map = {
                "Matte": "multiply",
                "Satin": "overlay",
                "Tint": "color",
            }
            inv_mode_map = {v: k for k, v in mode_map.items()}

            mode_var = tk.StringVar(value=inv_mode_map.get(self.settings.lipstick.blend_mode, "Matte"))
            mode_combo = ttk.Combobox(
                opt_row,
                textvariable=mode_var,
                values=list(mode_map.keys()),
                state="readonly",
                width=11,
            )
            mode_combo.pack(side=tk.LEFT)

            def _on_mode_change(_e):
                self.settings.lipstick.blend_mode = mode_map.get(mode_var.get(), "multiply")

            mode_combo.bind("<<ComboboxSelected>>", _on_mode_change)

        # Swatch Row & Custom Color Picker Button
        swatch_row = tk.Frame(card, bg=CARD)
        swatch_row.pack(fill=tk.X, padx=10, pady=2)

        self.swatch_buttons[key] = []
        for i, (name, rgb) in enumerate(SWATCHES.get(key, [])):
            btn = tk.Canvas(
                swatch_row,
                width=20,
                height=20,
                bg=rgb_to_hex(rgb),
                highlightthickness=1,
                highlightbackground=BORDER_ACTIVE if i == self.selected_swatch.get(key) else BORDER,
                cursor="hand2",
            )
            btn.grid(row=i // 7, column=i % 7, padx=2, pady=2)
            btn.bind("<Button-1>", lambda _e, p=key, idx=i: self._select_swatch(p, idx))
            btn.bind("<Enter>", lambda _e, n=name: self.status.set(f"{title}: {n}"))
            self.swatch_buttons[key].append(btn)

        # Custom Color Picker button
        custom_row = tk.Frame(card, bg=CARD)
        custom_row.pack(fill=tk.X, padx=10, pady=(1, 3))
        custom_btn = tk.Button(
            custom_row,
            text="Custom Color...",
            bg=CARD_HOVER,
            fg=TEXT,
            activebackground=BORDER,
            activeforeground=TEXT,
            font=("Segoe UI", 8),
            relief="flat",
            cursor="hand2",
            padx=6,
            pady=1,
            command=lambda p=key, t=title: self._choose_custom_color(p, t),
        )
        custom_btn.pack(side=tk.LEFT)
        self.custom_color_buttons[key] = custom_btn

        # Intensity Slider
        intensity_row = tk.Frame(card, bg=CARD)
        intensity_row.pack(fill=tk.X, padx=10, pady=(1, 5))
        ttk.Label(intensity_row, text="Intensity", style="Muted.TLabel").pack(anchor=tk.W)
        scale = tk.Scale(
            intensity_row,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            bg=CARD,
            fg=TEXT,
            highlightthickness=0,
            troughcolor="#121217",
            activebackground=ACCENT,
            showvalue=True,
            command=lambda val, p=key: self._set_intensity(p, val),
        )
        scale.set(int(state.intensity * 100))
        scale.pack(fill=tk.X)

    def _select_swatch(self, product: str, index: int) -> None:
        self.selected_swatch[product] = index
        name, rgb = SWATCHES[product][index]
        getattr(self.settings, product).color_bgr = rgb_to_bgr(rgb)
        for i, btn in enumerate(self.swatch_buttons.get(product, [])):
            btn.configure(highlightbackground=BORDER_ACTIVE if i == index else BORDER)
        self.custom_color_buttons[product].configure(bg=CARD_HOVER, text="Custom Color...", fg=TEXT)
        self.status.set(f"{product.title()}: {name}")

    def _choose_custom_color(self, product: str, title: str) -> None:
        cur_state = getattr(self.settings, product)
        cur_rgb = bgr_to_rgb(cur_state.color_bgr)
        cur_hex = rgb_to_hex(cur_rgb)
        color = colorchooser.askcolor(color=cur_hex, title=f"Choose {title} Shade")
        if color and color[0]:
            rgb = (int(color[0][0]), int(color[0][1]), int(color[0][2]))
            cur_state.color_bgr = rgb_to_bgr(rgb)
            self.selected_swatch[product] = None
            for btn in self.swatch_buttons.get(product, []):
                btn.configure(highlightbackground=BORDER)
            hex_val = color[1]
            self.custom_color_buttons[product].configure(
                bg=hex_val,
                fg="#000000" if (rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114) > 150 else "#ffffff",
                text=f"Custom: {hex_val}",
            )
            self.status.set(f"{title}: Custom {hex_val}")

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
            messagebox.showinfo("Webcam", "Could not open webcam.")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.status.set("Camera active")

    def save_result(self) -> None:
        if self.current_processed is None:
            messagebox.showinfo("Save Snapshot", "No active frame to save.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            initialfile="makeup_snapshot.jpg",
        )
        if not path:
            return
        cv2.imwrite(path, self.current_processed)
        self.status.set(f"Saved snapshot: {Path(path).name}")

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

    def _process_and_show(self, frame: np.ndarray, detector: FaceDetector) -> None:
        self.current_raw = frame
        if self.show_original.get():
            processed = frame
            self.status.set("Original face (makeup hidden)")
        else:
            landmarks = detector.detect_bgr(frame)
            self.current_landmarks = landmarks
            if landmarks:
                processed = apply_makeup(frame, landmarks, self.settings)
                self.status.set("Face detected - Real-time tracking")
            else:
                processed = frame
                self.status.set("No face detected")
        self.current_processed = processed
        self._show_frame(processed)
        self._update_fps()

    def _show_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        box_w = max(self.preview_container.winfo_width() - 8, 320)
        box_h = max(self.preview_container.winfo_height() - 8, 240)

        img_w, img_h = image.size
        ratio = min(box_w / img_w, box_h / img_h)
        new_size = (max(1, int(img_w * ratio)), max(1, int(img_h * ratio)))

        image = image.resize(new_size, Image.Resampling.BILINEAR)
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


if __name__ == "__main__":
    app = MakeupTryOnApp()
    app.run()