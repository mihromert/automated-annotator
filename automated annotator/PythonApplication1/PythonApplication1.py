import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

import cv2
from ultralytics import YOLO

# --- LOGIC CONSTANTS ---
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_EXTS = {".pt", ".onnx"}

# --- THEME CONSTANTS (Azzurro / Professional Blue Theme) ---
FONT_MAIN = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_TITLE = ("Segoe UI", 18, "bold")
FONT_BTN = ("Segoe UI", 11, "bold")

# Azzurro Palette
COLOR_BG = "#E3F2FD"       # Light Blue
COLOR_CARD = "#FFFFFF"     # White
COLOR_ACCENT = "#1565C0"   # Dark Blue
COLOR_BTN_RUN = "#81D4FA"  # Light Sky Blue
COLOR_BTN_STOP = "#FFAB91" # Soft Red/Orange
COLOR_INPUT_BG = "#FAFAFA" # Input background


def yolo_txt_line(class_id: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    """Standard Box Format (xywh normalized)"""
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    xc_n = min(max(xc / w, 0.0), 1.0)
    yc_n = min(max(yc / h, 0.0), 1.0)
    bw_n = min(max(bw / w, 0.0), 1.0)
    bh_n = min(max(bh / h, 0.0), 1.0)

    return f"{class_id} {xc_n:.6f} {yc_n:.6f} {bw_n:.6f} {bh_n:.6f}"


def yolo_obb_line(class_id: int, xy4, w: int, h: int) -> str:
    """OBB Format (x1 y1 x2 y2 x3 y3 x4 y4 normalized)"""
    points = []
    for x, y in xy4:
        xn = min(max(x / w, 0.0), 1.0)
        yn = min(max(y / h, 0.0), 1.0)
        points.extend([xn, yn])

    coords_str = " ".join([f"{p:.6f}" for p in points])
    return f"{class_id} {coords_str}"


def draw_box(img, x1, y1, x2, y2, label: str):
    """Draw standard rectangle bbox"""
    x1i, y1i, x2i, y2i = map(lambda v: int(round(v)), [x1, y1, x2, y2])
    color = (255, 191, 0)  # Deep Sky Blue (BGR)
    cv2.rectangle(img, (x1i, y1i), (x2i, y2i), color, 2)
    if label:
        cv2.putText(
            img, label, (x1i, max(0, y1i - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )


def draw_obb(img, xy4, label: str):
    """Draw oriented bounding box (OBB) polygon"""
    pts = xy4.astype(np.int32)
    pts = pts.reshape((-1, 1, 2))
    color = (0, 255, 127)  # Spring Green (BGR)

    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

    if label:
        x1, y1 = pts[0][0]
        cv2.putText(
            img, label, (int(x1), max(0, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Auto Annotator Pro (ONNX + OBB Optional)")
        self.geometry("1000x740")
        self.minsize(980, 700)

        self.model_path = tk.StringVar()
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()

        self.conf = tk.StringVar(value="0.40")
        self.use_iou = tk.BooleanVar(value=False)
        self.iou = tk.StringVar(value="0.70")
        self.save_images = tk.BooleanVar(value=True)

        # NEW: Task mode for ONNX (OBB optional)
        # Auto: try to load and run; Detect: force bbox; OBB: force obb
        self.task_mode = tk.StringVar(value="Auto")

        self._worker_thread = None
        self._stop_requested = False

        self._build_ui()

    def _build_ui(self):
        self.configure(bg=COLOR_BG)

        root = tk.Frame(self, bg=COLOR_BG)
        root.pack(fill="both", expand=True, padx=25, pady=20)

        header = tk.Frame(root, bg=COLOR_BG)
        header.pack(fill="x", pady=(0, 15))

        title = tk.Label(
            header,
            text="YOLO Auto Annotator",
            font=FONT_TITLE,
            bg=COLOR_BG,
            fg=COLOR_ACCENT,
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            header,
            text="Automated bounding box & optional OBB generation utility (PT/ONNX)",
            font=FONT_MAIN,
            fg="#546E7A",
            bg=COLOR_BG,
        )
        subtitle.pack(anchor="w")

        content = tk.Frame(root, bg=COLOR_BG)
        content.pack(fill="both", expand=True)

        left = tk.Frame(content, bg=COLOR_BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 15))

        right = tk.Frame(content, bg=COLOR_BG)
        right.pack(side="right", fill="both", expand=True)

        self._build_card_model(left)
        self._build_card_folders(left)
        self._build_card_settings(left)
        self._build_card_actions(left)

        self._build_card_status(right)
        self._build_card_log(right)

    def _card(self, parent, title: str):
        frame = tk.Frame(parent, bg=COLOR_BG)
        frame.pack(fill="x", pady=8)

        lbl = tk.Label(frame, text=title, font=FONT_BOLD, bg=COLOR_BG, fg="#455A64")
        lbl.pack(anchor="w", padx=2, pady=(0, 4))

        card_body = tk.LabelFrame(
            frame,
            text="",
            bg=COLOR_CARD,
            bd=0,
            highlightthickness=1,
            highlightbackground="#B0BEC5",
            relief="flat",
            padx=15,
            pady=15,
        )
        card_body.pack(fill="x")
        return card_body

    def _build_card_model(self, parent):
        card = self._card(parent, "Model Configuration")

        row = tk.Frame(card, bg=COLOR_CARD)
        row.pack(fill="x")

        tk.Label(row, text="Model Path (.pt/.onnx):", bg=COLOR_CARD, font=FONT_MAIN, width=20, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.model_path, font=FONT_MAIN, bg=COLOR_INPUT_BG, bd=1, relief="solid").pack(
            side="left", fill="x", expand=True, padx=8, ipady=4
        )
        tk.Button(row, text="Browse...", bg="#E1F5FE", font=FONT_MAIN, command=self._pick_model, relief="groove").pack(
            side="left"
        )

        # NEW: Task mode selector for ONNX
        r2 = tk.Frame(card, bg=COLOR_CARD)
        r2.pack(fill="x", pady=(10, 0))

        tk.Label(r2, text="ONNX Task Mode:", bg=COLOR_CARD, font=FONT_MAIN, width=20, anchor="w").pack(side="left")

        task_menu = tk.OptionMenu(r2, self.task_mode, "Auto", "Detect (BBox)", "OBB")
        task_menu.config(font=FONT_MAIN, bg=COLOR_INPUT_BG, relief="solid", bd=1, highlightthickness=0)
        task_menu["menu"].config(font=FONT_MAIN)
        task_menu.pack(side="left")

        tk.Label(
            r2,
            text="(Used only for .onnx)",
            bg=COLOR_CARD,
            fg="#888",
            font=("Segoe UI", 9),
        ).pack(side="left", padx=10)

    def _build_card_folders(self, parent):
        card = self._card(parent, "Directories")

        def folder_row(label_text, var, cmd):
            r = tk.Frame(card, bg=COLOR_CARD)
            r.pack(fill="x", pady=6)
            tk.Label(r, text=label_text, bg=COLOR_CARD, font=FONT_MAIN, width=20, anchor="w").pack(side="left")
            tk.Entry(r, textvariable=var, font=FONT_MAIN, bg=COLOR_INPUT_BG, bd=1, relief="solid").pack(
                side="left", fill="x", expand=True, padx=8, ipady=4
            )
            tk.Button(r, text="Browse...", bg="#E1F5FE", font=FONT_MAIN, command=cmd, relief="groove").pack(side="left")

        folder_row("Input Images:", self.input_dir, self._pick_input_dir)
        folder_row("Output Labels:", self.output_dir, self._pick_output_dir)

    def _build_card_settings(self, parent):
        card = self._card(parent, "Parameters")

        r1 = tk.Frame(card, bg=COLOR_CARD)
        r1.pack(fill="x", pady=6)
        tk.Label(r1, text="Confidence Thresh:", bg=COLOR_CARD, font=FONT_MAIN, width=20, anchor="w").pack(side="left")
        tk.Entry(r1, textvariable=self.conf, width=8, font=FONT_MAIN, bg=COLOR_INPUT_BG, justify="center").pack(
            side="left", padx=(8, 5)
        )
        tk.Label(r1, text="(0.0 - 1.0)", bg=COLOR_CARD, fg="#888", font=("Segoe UI", 9)).pack(side="left")

        r2 = tk.Frame(card, bg=COLOR_CARD)
        r2.pack(fill="x", pady=6)
        tk.Checkbutton(
            r2,
            text="Enable IoU / NMS Filtering",
            variable=self.use_iou,
            bg=COLOR_CARD,
            activebackground=COLOR_CARD,
            font=FONT_MAIN,
            command=self._toggle_iou,
        ).pack(side="left")

        self.iou_frame = tk.Frame(card, bg=COLOR_CARD)
        self.iou_frame.pack(fill="x", pady=(2, 0))
        tk.Label(self.iou_frame, text="IoU Threshold:", bg=COLOR_CARD, font=FONT_MAIN, width=20, anchor="w").pack(
            side="left"
        )
        tk.Entry(self.iou_frame, textvariable=self.iou, width=8, font=FONT_MAIN, bg=COLOR_INPUT_BG, justify="center").pack(
            side="left", padx=(8, 5)
        )
        self.iou_frame.pack_forget()

        r3 = tk.Frame(card, bg=COLOR_CARD)
        r3.pack(fill="x", pady=(8, 0))
        tk.Checkbutton(
            r3,
            text="Generate preview images (annotated)",
            variable=self.save_images,
            bg=COLOR_CARD,
            activebackground=COLOR_CARD,
            font=FONT_MAIN,
        ).pack(side="left")

    def _build_card_actions(self, parent):
        card = self._card(parent, "Actions")

        btns = tk.Frame(card, bg=COLOR_CARD)
        btns.pack(fill="x", pady=5)

        self.run_btn = tk.Button(
            btns,
            text="START PROCESSING",
            bg=COLOR_BTN_RUN,
            activebackground="#4FC3F7",
            font=FONT_BTN,
            relief="flat",
            height=2,
            cursor="hand2",
            command=self.start,
        )
        self.run_btn.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.stop_btn = tk.Button(
            btns,
            text="STOP",
            bg=COLOR_BTN_STOP,
            activebackground="#FF8A65",
            font=FONT_BTN,
            relief="flat",
            height=2,
            state="disabled",
            cursor="hand2",
            command=self.stop,
        )
        self.stop_btn.pack(side="right", fill="x", expand=False, ipadx=20)

    def _build_card_status(self, parent):
        card = self._card(parent, "Current Status")

        self.status = tk.Label(
            card,
            text="Ready to start.",
            bg=COLOR_CARD,
            fg="#333",
            font=("Segoe UI", 11),
            justify="left",
            wraplength=400,
        )
        self.status.pack(fill="x")

    def _build_card_log(self, parent):
        card = self._card(parent, "Process Log")

        log_frame = tk.Frame(card, bg=COLOR_CARD)
        log_frame.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side="right", fill="y")

        self.log = tk.Text(
            log_frame,
            height=20,
            width=40,
            font=("Consolas", 9),
            bg="#F5F5F5",
            relief="flat",
            yscrollcommand=scrollbar.set,
        )
        self.log.pack(fill="both", expand=True)
        scrollbar.config(command=self.log.yview)
        self.log.configure(state="disabled")

    def _toggle_iou(self):
        if self.use_iou.get():
            self.iou_frame.pack(fill="x", pady=(5, 0))
        else:
            self.iou_frame.pack_forget()

    def _pick_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[
                ("YOLO model", "*.pt *.onnx"),
                ("PyTorch model", "*.pt"),
                ("ONNX model", "*.onnx"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.model_path.set(path)

    def _pick_input_dir(self):
        path = filedialog.askdirectory(title="Select Input Image Folder")
        if path:
            self.input_dir.set(path)

    def _pick_output_dir(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_dir.set(path)

    def _log_line(self, s: str):
        self.log.configure(state="normal")
        self.log.insert("end", s + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_status(self, s: str):
        self.status.config(text=s)
        self.update_idletasks()

    def validate_inputs(self) -> bool:
        mp = self.model_path.get().strip()
        inp = self.input_dir.get().strip()
        out = self.output_dir.get().strip()

        if not mp or not Path(mp).exists():
            messagebox.showerror("Validation Error", "Model file not found or invalid path.")
            return False

        ext = Path(mp).suffix.lower()
        if ext not in MODEL_EXTS:
            messagebox.showerror("Validation Error", "Model must be a .pt or .onnx file.")
            return False

        if not inp or not Path(inp).exists():
            messagebox.showerror("Validation Error", "Input directory not found.")
            return False

        if not out:
            messagebox.showerror("Validation Error", "Please select an output directory.")
            return False

        try:
            conf = float(self.conf.get())
            if not (0.0 < conf <= 1.0):
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Confidence must be a float between 0.0 and 1.0")
            return False

        if self.use_iou.get():
            try:
                iou = float(self.iou.get())
                if not (0.0 < iou <= 1.0):
                    raise ValueError
            except ValueError:
                messagebox.showerror("Input Error", "IoU must be a float between 0.0 and 1.0")
                return False

        return True

    def start(self):
        if self._worker_thread and self._worker_thread.is_alive():
            return
        if not self.validate_inputs():
            return

        self._stop_requested = False
        self.run_btn.configure(state="disabled", text="PROCESSING...")
        self.stop_btn.configure(state="normal")

        self._log_line("--- Starting Annotation Task ---")
        self._set_status("Initializing model and checking files...")

        self._worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self._worker_thread.start()

    def stop(self):
        self._stop_requested = True
        self._set_status("Stop requested. Waiting for current image to finish...")
        self._log_line(">>> STOP COMMAND RECEIVED")

    def _load_model(self, mp: str):
        """
        Loads PT or ONNX model.
        For ONNX, uses task_mode if selected.
        """
        ext = Path(mp).suffix.lower()

        if ext == ".pt":
            return YOLO(mp)

        # ONNX
        mode = self.task_mode.get()
        if mode == "Detect (BBox)":
            return YOLO(mp, task="detect")
        if mode == "OBB":
            return YOLO(mp, task="obb")

        # Auto mode: try without task first; fallback to detect
        try:
            return YOLO(mp)
        except Exception:
            return YOLO(mp, task="detect")

    def _run_worker(self):
        try:
            mp = self.model_path.get().strip()
            model = self._load_model(mp)

            in_dir = Path(self.input_dir.get().strip())
            out_dir = Path(self.output_dir.get().strip())
            out_dir.mkdir(parents=True, exist_ok=True)

            preview_dir = None
            if self.save_images.get():
                preview_dir = out_dir / "preview_annotated"
                preview_dir.mkdir(parents=True, exist_ok=True)

            images = [p for p in sorted(in_dir.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
            if not images:
                self._set_status("No valid images found in input directory.")
                raise RuntimeError("Input directory is empty or contains no supported images.")

            conf_thresh = float(self.conf.get())
            use_iou = self.use_iou.get()
            iou_val = float(self.iou.get()) if use_iou else None

            self._set_status(f"Found {len(images)} images. Starting processing loop.")
            self._log_line(f"Model: {mp}")
            self._log_line(f"ONNX Task Mode: {self.task_mode.get()} (only if .onnx)")
            self._log_line(f"Input: {len(images)} images")
            self._log_line(f"Conf: {conf_thresh} | IoU: {iou_val if use_iou else 'Default'}")
            self._log_line("-" * 40)

            for idx, img_path in enumerate(images, start=1):
                if self._stop_requested:
                    self._log_line("Process terminated by user.")
                    break

                img = cv2.imread(str(img_path))
                if img is None:
                    self._log_line(f"Error reading: {img_path.name}")
                    continue

                h, w = img.shape[:2]

                predict_kwargs = {
                    "source": str(img_path),
                    "conf": conf_thresh,
                    "verbose": False,
                }
                if use_iou:
                    predict_kwargs["iou"] = iou_val

                results = model.predict(**predict_kwargs)
                r = results[0]

                lines = []

                # --- OBB DETECTION (OPTIONAL) ---
                if getattr(r, "obb", None) is not None and r.obb is not None:
                    xy4_tensor = r.obb.xyxyxyxy.cpu().numpy()
                    cls_tensor = r.obb.cls.cpu().numpy().astype(int)
                    conf_tensor = r.obb.conf.cpu().numpy()

                    for xy4, c, cf in zip(xy4_tensor, cls_tensor, conf_tensor):
                        if cf < conf_thresh:
                            continue
                        lines.append(yolo_obb_line(c, xy4, w, h))
                        if preview_dir is not None:
                            draw_obb(img, xy4, f"{c} {cf:.2f}")

                # --- STANDARD BBOX DETECTION ---
                elif getattr(r, "boxes", None) is not None and r.boxes is not None:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    cls = r.boxes.cls.cpu().numpy().astype(int)
                    confs = r.boxes.conf.cpu().numpy()

                    for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, confs):
                        if cf < conf_thresh:
                            continue
                        lines.append(yolo_txt_line(c, x1, y1, x2, y2, w, h))
                        if preview_dir is not None:
                            draw_box(img, x1, y1, x2, y2, f"{c} {cf:.2f}")

                # --- SAVE LABELS ---
                label_path = out_dir / f"{img_path.stem}.txt"
                label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

                # --- SAVE PREVIEW ---
                if preview_dir is not None:
                    cv2.imwrite(str(preview_dir / img_path.name), img)

                self._log_line(f"[{idx}/{len(images)}] {img_path.name}: {len(lines)} objects")
                self._set_status(f"Processing: {idx}/{len(images)} ({img_path.name})")

            if self._stop_requested:
                self._set_status("Stopped.")
            else:
                self._set_status("Processing Complete.")

            self._log_line("-" * 40)
            self._log_line("Task Finished.")

        except Exception as e:
            self._set_status("Error occurred.")
            self._log_line(f"ERROR: {e}")
            messagebox.showerror("Runtime Error", f"An error occurred:\n{e}")
        finally:
            self.run_btn.configure(state="normal", text="START PROCESSING")
            self.stop_btn.configure(state="disabled")


if __name__ == "__main__":
    # NOTE: For ONNX models you typically need:
    #   pip install onnxruntime
    # or (GPU):
    #   pip install onnxruntime-gpu
    app = App()
    app.mainloop()