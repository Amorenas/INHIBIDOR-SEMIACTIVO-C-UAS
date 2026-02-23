import time
import threading
import queue
from pathlib import Path

import tkinter as tk
from PIL import Image, ImageTk

import numpy as np
import sounddevice as sd
from scipy.signal import butter, filtfilt, spectrogram, welch


# ============================================================
# RUTAS (assets/)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

BUQUE_PATH = ASSETS_DIR / "buque.png"
DRON_PATH = ASSETS_DIR / "dron.png"
ALERTA_WAV_PATH = ASSETS_DIR / "alerta.wav"


# ============================================================
# DSP (basado en TFG Aznar):
# Butterworth orden 5, banda 3000–6000 Hz,
# spectrogram + Sxx_dB = 10log10(Sxx + 1e-10), media en banda
# ============================================================
RATE = 44100
CHANNELS = 1

WINDOW_SECONDS_DEFAULT = 5.0
F_LOW_DEFAULT = 3000.0
F_HIGH_DEFAULT = 6000.0
BP_ORDER_DEFAULT = 5


# ============================================================
# UMBRAL ADAPTATIVO + AUTO RECALIB
# ============================================================
ADAPTIVE_DEFAULT = True

CALIB_SECONDS_DEFAULT = 3.0          # silencio al recalibrar
MARGIN_LOW_DB_DEFAULT = 10.0         # baseline + 10 dB => dispara detección
MARGIN_HIGH_DB_DEFAULT = 70.0        # baseline + 70 dB => techo alto (configurable)
HYST_DB_DEFAULT = 3.0                # histéresis OFF

USE_UPPER_THRESHOLD_DEFAULT = True   # (TFG) activar doble umbral

AUTO_RECALIB_EVERY_SEC = 180         # 3 minutos


# Si quieres forzar un input: pon índice (1, 6, 12, 30, ...)
FORCE_INPUT_DEVICE_INDEX = None


# ============================================================
# UI / ESTILO
# ============================================================
BG_MAIN = "#050B12"
FRAME_COLOR = "#1C252F"
PANEL_BG = "#243B4E"
TEXT_MAIN = "#E2F4FF"

ACCENT_SAFE = "#00D68F"
ACCENT_DANGER = "#FF3B3B"
ACCENT_DANGER_DARK = "#992222"
ACCENT_AMBER = "#F5C542"

STATUS_STRIP_BG = "#0B1118"
LED_OFF = "#555555"

GRID_DIM = "#1E2E3A"
SPEC_LINE = "#00FF7F"

DIST_MAX_METROS = 15.0


def clamp(v, a, b):
    return max(a, min(b, v))


def safe_db10(x, eps=1e-10):
    return 10.0 * np.log10(np.maximum(x, eps))


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999999)
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)


def pick_usb_input_device_index():
    """
    Heurística para elegir el input USB (si existe).
    Si no se encuentra, devuelve None => usa default del sistema.
    """
    try:
        devs = sd.query_devices()
        candidates = []
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) <= 0:
                continue
            name = (d.get("name") or "").lower()
            if "usb" in name and ("pnp" in name or "sound" in name or "micro" in name):
                candidates.append((i, d))
        if not candidates:
            return None
        candidates.sort(key=lambda x: abs(float(x[1].get("default_samplerate", 44100.0)) - 44100.0))
        return candidates[0][0]
    except Exception:
        return None


class AudioWorker(threading.Thread):
    """
    Captura audio del micro en PC y calcula:
      - Espectro para UI (Welch)
      - avg_energy_db (TFG) a partir de ventana WINDOW_SECONDS
      - Umbral adaptativo: baseline en silencio (media) y crea thr_low/thr_high
      - Detección con histéresis y (opcional) doble umbral

    Publica:
      ("MIC_OK"/"MIC_ERR", msg)
      ("SPEC", freqs, db_vals)
      ("DET", detected_bool, avg_energy_db, thr_low, thr_high, baseline_mean, baseline_std, calibrating)
      ("CALIB_DONE", baseline_mean, baseline_std, thr_low, thr_high)
    """

    def __init__(self, out_q: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.q = out_q
        self.stop_event = stop_event

        self.lock = threading.Lock()

        # DSP
        self.window_seconds = WINDOW_SECONDS_DEFAULT
        self.f_low = F_LOW_DEFAULT
        self.f_high = F_HIGH_DEFAULT
        self.bp_order = BP_ORDER_DEFAULT

        # Adaptive
        self.adaptive = ADAPTIVE_DEFAULT
        self.calib_seconds = CALIB_SECONDS_DEFAULT
        self.margin_low_db = MARGIN_LOW_DB_DEFAULT
        self.margin_high_db = MARGIN_HIGH_DB_DEFAULT
        self.hyst_db = HYST_DB_DEFAULT
        self.use_upper = USE_UPPER_THRESHOLD_DEFAULT

        # thresholds runtime
        self.thr_low = None
        self.thr_high = None
        self.baseline_mean = None
        self.baseline_std = None

        self._calibrating = True
        self._calib_start_ts = time.time()
        self._calib_values = []

        # Detection state (hysteresis)
        self._det_state = False

        # device
        self.device_index = FORCE_INPUT_DEVICE_INDEX
        if self.device_index is None:
            auto_usb = pick_usb_input_device_index()
            if auto_usb is not None:
                self.device_index = auto_usb

        self._buf = np.zeros(int(RATE * self.window_seconds), dtype=np.float32)
        self._buf_fill = 0
        self._last_det_ts = 0.0
        self._det_period = 0.30

        # External command
        self._recalibrate_flag = False

    def push(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            pass

    def request_recalibration(self):
        with self.lock:
            self._recalibrate_flag = True

    def set_params(self, window_seconds, f_low, f_high, bp_order,
                   adaptive, calib_seconds, margin_low_db, margin_high_db, hyst_db, use_upper):
        with self.lock:
            self.window_seconds = float(window_seconds)
            self.f_low = float(f_low)
            self.f_high = float(f_high)
            self.bp_order = int(bp_order)

            self.adaptive = bool(adaptive)
            self.calib_seconds = float(calib_seconds)
            self.margin_low_db = float(margin_low_db)
            self.margin_high_db = float(margin_high_db)
            self.hyst_db = float(hyst_db)
            self.use_upper = bool(use_upper)

            n = max(2048, int(RATE * self.window_seconds))
            self._buf = np.zeros(n, dtype=np.float32)
            self._buf_fill = 0

            if self.adaptive:
                self._recalibrate_flag = True

    def _start_calibration(self):
        self._calibrating = True
        self._calib_start_ts = time.time()
        self._calib_values = []
        self.baseline_mean = None
        self.baseline_std = None
        self.thr_low = None
        self.thr_high = None
        self._det_state = False

    def _finalize_calibration(self):
        if len(self._calib_values) < 5:
            self.baseline_mean = -30.0
            self.baseline_std = 2.0
        else:
            arr = np.array(self._calib_values, dtype=np.float32)
            self.baseline_mean = float(np.mean(arr))
            self.baseline_std = float(np.std(arr))

        self.thr_low = self.baseline_mean + self.margin_low_db
        self.thr_high = self.baseline_mean + self.margin_high_db

        self._calibrating = False
        self.push(("CALIB_DONE", self.baseline_mean, self.baseline_std, self.thr_low, self.thr_high))

    def run(self):
        try:
            if self.device_index is None:
                in_idx = sd.default.device[0]
                dinfo = sd.query_devices(in_idx)
                self.push(("MIC_OK", f"INPUT default idx={in_idx}: {dinfo.get('name','?')}"))
            else:
                dinfo = sd.query_devices(self.device_index)
                self.push(("MIC_OK", f"INPUT idx={self.device_index}: {dinfo.get('name','?')}"))
        except Exception as e:
            self.push(("MIC_ERR", f"No puedo consultar dispositivos: {e}"))

        if not self.adaptive:
            self._calibrating = False

        def callback(indata, frames, time_info, status):
            if self.stop_event.is_set():
                raise sd.CallbackStop()

            x = indata[:, 0].astype(np.float32)

            # ---- ESPECTRO PARA UI ----
            try:
                f, Pxx = welch(x, fs=RATE, nperseg=min(1024, len(x)))
                Pxx_db = safe_db10(Pxx, eps=1e-20)
                mask = f <= 8000
                self.push(("SPEC", f[mask], Pxx_db[mask]))
            except Exception:
                pass

            # ---- BUFFER VENTANA ----
            with self.lock:
                nbuf = len(self._buf)
                do_recal = self._recalibrate_flag
                if do_recal:
                    self._recalibrate_flag = False

            if do_recal and self.adaptive:
                self._start_calibration()

            n = len(x)
            if n >= nbuf:
                with self.lock:
                    self._buf[:] = x[-nbuf:]
                    self._buf_fill = nbuf
            else:
                with self.lock:
                    remaining = nbuf - self._buf_fill
                    if n <= remaining:
                        self._buf[self._buf_fill:self._buf_fill + n] = x
                        self._buf_fill += n
                    else:
                        shift = n - remaining
                        self._buf[:-shift] = self._buf[shift:]
                        self._buf[-n:] = x
                        self._buf_fill = nbuf

            now = time.time()
            if now - self._last_det_ts < self._det_period:
                return
            self._last_det_ts = now

            with self.lock:
                if self._buf_fill < len(self._buf):
                    return

                buf_copy = self._buf.copy()
                f_low = self.f_low
                f_high = self.f_high
                order = self.bp_order

                adaptive = self.adaptive
                calib_seconds = self.calib_seconds
                hyst_db = self.hyst_db
                use_upper = self.use_upper

                thr_low = self.thr_low
                thr_high = self.thr_high
                baseline_mean = self.baseline_mean
                baseline_std = self.baseline_std
                calibrating = self._calibrating
                det_state = self._det_state
                calib_start = self._calib_start_ts

            # Pipeline TFG: bandpass -> spectrogram -> mean dB in band
            try:
                x_i16 = np.int16(np.clip(buf_copy, -1.0, 1.0) * 32767)
                xf = bandpass_filter(x_i16.astype(np.float32), f_low, f_high, RATE, order=order)

                f_sp, t_sp, Sxx = spectrogram(xf, fs=RATE)
                Sxx_dB = safe_db10(Sxx, eps=1e-10)

                low_idx = int(np.searchsorted(f_sp, f_low))
                high_idx = int(np.searchsorted(f_sp, f_high))
                low_idx = clamp(low_idx, 0, len(f_sp) - 1)
                high_idx = clamp(high_idx, low_idx + 1, len(f_sp))

                avg_db = float(np.mean(Sxx_dB[low_idx:high_idx, :]))
            except Exception:
                return

            # ---- Adaptive calibration / thresholds ----
            if adaptive:
                if calibrating:
                    with self.lock:
                        self._calib_values.append(avg_db)

                    if (now - calib_start) >= calib_seconds:
                        self._finalize_calibration()
                        with self.lock:
                            thr_low = self.thr_low
                            thr_high = self.thr_high
                            baseline_mean = self.baseline_mean
                            baseline_std = self.baseline_std
                            calibrating = self._calibrating

                detected = False

                # Lógica de detección:
                # - Entrada ON si avg >= thr_low
                # - Salida OFF si avg <= thr_low - hyst
                # - Si use_upper: además debe cumplirse avg <= thr_high (si existe),
                #   y si se pasa por arriba => forzamos OFF (evita jammer fuerte).
                if thr_low is not None:
                    if use_upper and (thr_high is not None) and (avg_db > thr_high):
                        det_state = False
                    else:
                        if not det_state:
                            if avg_db >= thr_low:
                                det_state = True
                        else:
                            if avg_db <= (thr_low - hyst_db):
                                det_state = False

                    with self.lock:
                        self._det_state = det_state

                    detected = det_state
            else:
                detected = False

            self.push((
                "DET",
                bool(detected),
                float(avg_db),
                thr_low,
                thr_high,
                baseline_mean,
                baseline_std,
                bool(calibrating)
            ))

        try:
            with sd.InputStream(
                device=self.device_index,
                channels=CHANNELS,
                samplerate=RATE,
                dtype="float32",
                callback=callback,
            ):
                while not self.stop_event.is_set():
                    time.sleep(0.05)
        except Exception as e:
            self.push(("MIC_ERR", f"No se pudo abrir el micrófono: {e}"))


class DetectorUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DETECTOR DE UAS - CONSOLA DE COMBATE")
        self.root.configure(bg=BG_MAIN)
        self.root.geometry("1400x720")

        self.stop_event = threading.Event()
        self.msg_q = queue.Queue(maxsize=120)

        # DSP params
        self.window_seconds = WINDOW_SECONDS_DEFAULT
        self.f_low = F_LOW_DEFAULT
        self.f_high = F_HIGH_DEFAULT
        self.bp_order = BP_ORDER_DEFAULT

        # Adaptive params
        self.adaptive = ADAPTIVE_DEFAULT
        self.calib_seconds = CALIB_SECONDS_DEFAULT
        self.margin_low_db = MARGIN_LOW_DB_DEFAULT
        self.margin_high_db = MARGIN_HIGH_DB_DEFAULT
        self.hyst_db = HYST_DB_DEFAULT
        self.use_upper = USE_UPPER_THRESHOLD_DEFAULT

        # runtime status
        self.detected = False
        self.avg_energy_db = None
        self.thr_low = None
        self.thr_high = None
        self.baseline_mean = None
        self.baseline_std = None
        self.calibrating = True

        self.sound_enabled = True
        self.blink_enabled = True
        self.alarm_active = False
        self.blink_on = True

        self.after_ids = []
        self.config_win = None

        # assets
        self.ship_img = None
        self.drone_img = None
        self._load_assets()

        # ===================== TOP BAR =====================
        title_frame = tk.Frame(root, bg=BG_MAIN)
        title_frame.pack(pady=(5, 0), fill="x")

        tk.Label(
            title_frame, text="DETECTOR DE UAS",
            font=("Verdana", 28, "bold"),
            fg=TEXT_MAIN, bg=BG_MAIN
        ).pack(side="left", padx=10)

        # Botón RECALIBRAR EN MAIN (lo que pediste)
        tk.Button(
            title_frame, text="RECALIBRAR UMBRAL",
            font=("Verdana", 10, "bold"),
            command=self.recalibrate,
            bg="#29323C", fg="white",
            activebackground="#3C4A58", activeforeground="white",
            relief="raised", bd=2
        ).pack(side="right", padx=(6, 10))

        tk.Button(
            title_frame, text="CONFIGURACIÓN",
            font=("Verdana", 10, "bold"),
            command=self.open_config,
            bg="#29323C", fg="white",
            activebackground="#3C4A58", activeforeground="white",
            relief="raised", bd=2
        ).pack(side="right", padx=(0, 6))

        status_strip = tk.Frame(root, bg=STATUS_STRIP_BG)
        status_strip.pack(fill="x", padx=10, pady=(5, 10))

        self._create_status_block(status_strip, "MODO", "DEFENSA UAS", 0)
        self._create_status_block(status_strip, "SENSOR", "MIC USB (LOCAL)", 1)

        self.lbl_calib = tk.Label(
            status_strip,
            text="CALIBRANDO...",
            font=("Consolas", 11, "bold"),
            bg=STATUS_STRIP_BG,
            fg=ACCENT_AMBER
        )
        self.lbl_calib.grid(row=0, column=2, padx=10, sticky="w")

        leds_frame = tk.Frame(status_strip, bg=STATUS_STRIP_BG)
        leds_frame.grid(row=0, column=3, padx=10, sticky="e")

        self.led_mic = self._create_led(leds_frame, "MIC", LED_OFF)
        self.led_mic.pack(side="left", padx=5)
        self.led_dsp = self._create_led(leds_frame, "DSP", LED_OFF)
        self.led_dsp.pack(side="left", padx=5)
        self.led_alert = self._create_led(leds_frame, "ALERTA", LED_OFF)
        self.led_alert.pack(side="left", padx=5)

        # ===================== CENTER (3 PANELES) =====================
        center_frame = tk.Frame(root, bg=BG_MAIN)
        center_frame.pack(fill="both", expand=True, padx=20, pady=10)

        center_frame.columnconfigure(0, weight=1)
        center_frame.columnconfigure(1, weight=1)
        center_frame.columnconfigure(2, weight=1)
        center_frame.rowconfigure(0, weight=1)

        # --------- Panel Espectro (izquierda) ----------
        self.spec_frame = tk.Frame(center_frame, bg=PANEL_BG,
                                   highlightbackground=FRAME_COLOR, highlightthickness=3)
        self.spec_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.spec_frame.rowconfigure(1, weight=1)
        self.spec_frame.columnconfigure(0, weight=1)

        tk.Label(self.spec_frame, text="ANALIZADOR DE ESPECTRO (dB)",
                 font=("Verdana", 16, "bold"),
                 bg=PANEL_BG, fg=TEXT_MAIN).grid(row=0, column=0, sticky="ew", pady=(8, 4))

        self.spec_canvas = tk.Canvas(self.spec_frame, bg=PANEL_BG, highlightthickness=0)
        self.spec_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # --------- Panel Energía (centro) ----------
        self.energy_frame = tk.Frame(center_frame, bg=PANEL_BG,
                                     highlightbackground=FRAME_COLOR, highlightthickness=3)
        self.energy_frame.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)

        tk.Label(self.energy_frame, text="NIVEL ENERGÉTICO (dB)",
                 font=("Verdana", 16, "bold"),
                 bg=PANEL_BG, fg=TEXT_MAIN).grid(row=0, column=0, sticky="ew", pady=(8, 2))

        self.energy_frame.rowconfigure(1, weight=1)
        self.energy_frame.rowconfigure(2, weight=0)
        self.energy_frame.columnconfigure(0, weight=1)

        # IMPORTANTE: márgenes muy pequeños para que llene el recuadro
        self.energy_canvas = tk.Canvas(self.energy_frame, bg=PANEL_BG, highlightthickness=0)
        self.energy_canvas.grid(row=1, column=0, sticky="nsew", padx=2, pady=(0, 2))

        footer = tk.Frame(self.energy_frame, bg=PANEL_BG)
        footer.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 8))
        footer.columnconfigure(0, weight=1)

        self.lbl_energy_value = tk.Label(
            footer, text="ENERGÍA: --- dB",
            font=("Consolas", 14, "bold"),
            bg=PANEL_BG, fg=TEXT_MAIN
        )
        self.lbl_energy_value.grid(row=0, column=0, sticky="w")

        self.btn_rearm = tk.Button(
            footer,
            text="REARMAR / SILENCIAR ALARMA",
            font=("Verdana", 11, "bold"),
            command=self.rearm_alarm,
            bg="#29323C", fg="white",
            activebackground="#3C4A58", activeforeground="white",
            relief="raised", bd=3
        )
        self.btn_rearm.grid(row=0, column=1, sticky="e", padx=(10, 0))

        # --------- Panel Distancia (derecha) ----------
        self.dist_frame = tk.Frame(center_frame, bg=PANEL_BG,
                                   highlightbackground=FRAME_COLOR, highlightthickness=3)
        self.dist_frame.grid(row=0, column=2, sticky="nsew", padx=8, pady=8)
        self.dist_frame.rowconfigure(1, weight=1)
        self.dist_frame.columnconfigure(0, weight=1)

        tk.Label(self.dist_frame, text="DISTANCIA UAS ENEMIGO",
                 font=("Verdana", 16, "bold"),
                 bg=PANEL_BG, fg=TEXT_MAIN).grid(row=0, column=0, sticky="ew", pady=(8, 4))

        self.dist_canvas = tk.Canvas(self.dist_frame, bg=PANEL_BG, highlightthickness=0)
        self.dist_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))

        self.lbl_dist_value = tk.Label(
            self.dist_frame, text="DISTANCIA: ---",
            font=("Consolas", 13, "bold"),
            bg=PANEL_BG, fg=TEXT_MAIN
        )
        self.lbl_dist_value.grid(row=2, column=0, sticky="ew", pady=(0, 12))

        # ===================== BOTTOM STATUS =====================
        bottom = tk.Frame(root, bg=PANEL_BG, highlightbackground=FRAME_COLOR, highlightthickness=3)
        bottom.pack(fill="x", padx=20, pady=(0, 12))

        tk.Label(bottom, text="DETECCIÓN REGISTRADA:",
                 font=("Verdana", 18, "bold"),
                 bg=PANEL_BG, fg=TEXT_MAIN).pack(pady=8)

        self.lbl_status = tk.Label(
            bottom,
            text="CALIBRANDO... MANTÉN SILENCIO",
            font=("Verdana", 20, "bold"),
            bg=ACCENT_AMBER, fg="black"
        )
        self.lbl_status.pack(fill="x", ipady=18, padx=20, pady=(0, 12))

        # ===================== AUDIO WORKER =====================
        self.worker = AudioWorker(self.msg_q, self.stop_event)
        self.worker.set_params(
            self.window_seconds, self.f_low, self.f_high, self.bp_order,
            self.adaptive, self.calib_seconds, self.margin_low_db, self.margin_high_db, self.hyst_db, self.use_upper
        )
        self.worker.start()

        # loops
        self.after_ids.append(self.root.after(200, self._first_draw))
        self.after_ids.append(self.root.after(60, self._poll_queue))
        self.after_ids.append(self.root.after(250, self._blink_loop))

        # AUTO RECALIB CADA 3 MINUTOS (lo que pediste)
        self.after_ids.append(self.root.after(int(AUTO_RECALIB_EVERY_SEC * 1000), self._auto_recalib_loop))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------
    # ASSETS
    # ------------------------------------------------------------
    def _load_assets(self):
        try:
            img_ship = Image.open(BUQUE_PATH).resize((140, 90), Image.LANCZOS)
            self.ship_img = ImageTk.PhotoImage(img_ship)
        except Exception:
            self.ship_img = None

        try:
            img_drone = Image.open(DRON_PATH).resize((70, 70), Image.LANCZOS)
            self.drone_img = ImageTk.PhotoImage(img_drone)
        except Exception:
            self.drone_img = None

    # ------------------------------------------------------------
    # TOP HELPERS
    # ------------------------------------------------------------
    def _create_status_block(self, parent, title, value, column):
        frame = tk.Frame(parent, bg=STATUS_STRIP_BG)
        frame.grid(row=0, column=column, padx=14, sticky="w")

        tk.Label(frame, text=title,
                 font=("Verdana", 8, "bold"),
                 bg=STATUS_STRIP_BG, fg="#6FA3C8").pack(anchor="w")

        tk.Label(frame, text=value,
                 font=("Consolas", 11, "bold"),
                 bg=STATUS_STRIP_BG, fg=TEXT_MAIN).pack(anchor="w")

    def _create_led(self, parent, text, color):
        block = tk.Frame(parent, bg=STATUS_STRIP_BG)
        led = tk.Label(block, bg=color, width=2, height=1, relief="sunken")
        led.pack(side="left", padx=(0, 4))
        tk.Label(block, text=text,
                 font=("Verdana", 8, "bold"),
                 bg=STATUS_STRIP_BG, fg=TEXT_MAIN).pack(side="left")
        block._led = led
        return block

    # ------------------------------------------------------------
    # AUTO RECALIB LOOP
    # ------------------------------------------------------------
    def _auto_recalib_loop(self):
        # Recalibra SIEMPRE cada 3 min (tal y como pediste)
        self.recalibrate(auto=True)
        self.after_ids.append(self.root.after(int(AUTO_RECALIB_EVERY_SEC * 1000), self._auto_recalib_loop))

    # ------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------
    def open_config(self):
        if self.config_win is not None and self.config_win.winfo_exists():
            self.config_win.lift()
            return

        win = tk.Toplevel(self.root)
        self.config_win = win
        win.title("Configuración rápida")
        win.configure(bg=BG_MAIN)
        win.resizable(False, False)

        pad = {"padx": 10, "pady": 6}

        # DSP
        self.var_win = tk.StringVar(value=str(self.window_seconds))
        self.var_lowf = tk.StringVar(value=str(self.f_low))
        self.var_highf = tk.StringVar(value=str(self.f_high))
        self.var_order = tk.StringVar(value=str(self.bp_order))

        # Adaptive
        self.var_adapt = tk.BooleanVar(value=self.adaptive)
        self.var_calibsec = tk.StringVar(value=str(self.calib_seconds))
        self.var_mlow = tk.StringVar(value=str(self.margin_low_db))
        self.var_mhigh = tk.StringVar(value=str(self.margin_high_db))
        self.var_hyst = tk.StringVar(value=str(self.hyst_db))
        self.var_upper = tk.BooleanVar(value=self.use_upper)

        # Alarm UI
        self.var_sound = tk.BooleanVar(value=self.sound_enabled)
        self.var_blink = tk.BooleanVar(value=self.blink_enabled)

        r = 0
        tk.Label(win, text="Ventana detección (s)",
                 bg=BG_MAIN, fg=TEXT_MAIN, font=("Verdana", 10, "bold")).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(win, textvariable=self.var_win, width=12).grid(row=r, column=1, **pad)

        r += 1
        tk.Label(win, text="Banda low (Hz)",
                 bg=BG_MAIN, fg=TEXT_MAIN, font=("Verdana", 10, "bold")).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(win, textvariable=self.var_lowf, width=12).grid(row=r, column=1, **pad)

        r += 1
        tk.Label(win, text="Banda high (Hz)",
                 bg=BG_MAIN, fg=TEXT_MAIN, font=("Verdana", 10, "bold")).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(win, textvariable=self.var_highf, width=12).grid(row=r, column=1, **pad)

        r += 1
        tk.Label(win, text="Orden Butterworth",
                 bg=BG_MAIN, fg=TEXT_MAIN, font=("Verdana", 10, "bold")).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(win, textvariable=self.var_order, width=12).grid(row=r, column=1, **pad)

        r += 1
        tk.Checkbutton(
            win, text="Umbral ADAPTATIVO (recomendado)",
            variable=self.var_adapt,
            bg=BG_MAIN, fg=TEXT_MAIN,
            selectcolor=BG_MAIN,
            activebackground=BG_MAIN, activeforeground=TEXT_MAIN
        ).grid(row=r, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 0))

        r += 1
        tk.Checkbutton(
            win, text="Usar UMBRAL SUPERIOR (anti-jammer / reduce falsos +)",
            variable=self.var_upper,
            bg=BG_MAIN, fg=TEXT_MAIN,
            selectcolor=BG_MAIN,
            activebackground=BG_MAIN, activeforeground=TEXT_MAIN
        ).grid(row=r, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

        r += 1
        tk.Label(win, text="Calibración (s) en silencio",
                 bg=BG_MAIN, fg=TEXT_MAIN, font=("Verdana", 10, "bold")).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(win, textvariable=self.var_calibsec, width=12).grid(row=r, column=1, **pad)

        r += 1
        tk.Label(win, text="Margen LOW (dB) sobre baseline",
                 bg=BG_MAIN, fg=TEXT_MAIN, font=("Verdana", 10, "bold")).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(win, textvariable=self.var_mlow, width=12).grid(row=r, column=1, **pad)

        r += 1
        tk.Label(win, text="Margen HIGH (dB) sobre baseline",
                 bg=BG_MAIN, fg=TEXT_MAIN, font=("Verdana", 10, "bold")).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(win, textvariable=self.var_mhigh, width=12).grid(row=r, column=1, **pad)

        r += 1
        tk.Label(win, text="Histeresis OFF (dB) (evita parpadeo)",
                 bg=BG_MAIN, fg=TEXT_MAIN, font=("Verdana", 10, "bold")).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(win, textvariable=self.var_hyst, width=12).grid(row=r, column=1, **pad)

        r += 1
        tk.Checkbutton(
            win, text="Activar sonido de alarma",
            variable=self.var_sound,
            bg=BG_MAIN, fg=TEXT_MAIN,
            selectcolor=BG_MAIN,
            activebackground=BG_MAIN, activeforeground=TEXT_MAIN
        ).grid(row=r, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 0))

        r += 1
        tk.Checkbutton(
            win, text="Parpadeo cuando hay alarma",
            variable=self.var_blink,
            bg=BG_MAIN, fg=TEXT_MAIN,
            selectcolor=BG_MAIN,
            activebackground=BG_MAIN, activeforeground=TEXT_MAIN
        ).grid(row=r, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

        r += 1
        btns = tk.Frame(win, bg=BG_MAIN)
        btns.grid(row=r, column=0, columnspan=2, pady=10)

        tk.Button(
            btns, text="RECALIBRAR",
            command=self.recalibrate,
            font=("Verdana", 10, "bold"),
            bg="#29323C", fg="white",
            activebackground="#3C4A58", activeforeground="white"
        ).pack(side="left", padx=6)

        tk.Button(
            btns, text="PROBAR ALARMA",
            command=self.test_alarm,
            font=("Verdana", 10, "bold"),
            bg="#29323C", fg="white",
            activebackground="#3C4A58", activeforeground="white"
        ).pack(side="left", padx=6)

        tk.Button(
            btns, text="Guardar",
            command=self.save_config,
            font=("Verdana", 10, "bold"),
            bg="#29323C", fg="white",
            activebackground="#3C4A58", activeforeground="white"
        ).pack(side="left", padx=6)

    def save_config(self):
        try:
            # DSP
            self.window_seconds = float(self.var_win.get())
            self.f_low = float(self.var_lowf.get())
            self.f_high = float(self.var_highf.get())
            self.bp_order = int(float(self.var_order.get()))

            # Adaptive
            self.adaptive = bool(self.var_adapt.get())
            self.calib_seconds = float(self.var_calibsec.get())
            self.margin_low_db = float(self.var_mlow.get())
            self.margin_high_db = float(self.var_mhigh.get())
            self.hyst_db = float(self.var_hyst.get())
            self.use_upper = bool(self.var_upper.get())
        except Exception:
            pass

        if self.f_high <= self.f_low:
            self.f_high = self.f_low + 1000.0
        self.bp_order = int(clamp(self.bp_order, 1, 10))

        self.calib_seconds = clamp(self.calib_seconds, 1.0, 10.0)
        self.hyst_db = clamp(self.hyst_db, 0.5, 10.0)

        self.sound_enabled = bool(self.var_sound.get())
        self.blink_enabled = bool(self.var_blink.get())

        self.worker.set_params(
            self.window_seconds, self.f_low, self.f_high, self.bp_order,
            self.adaptive, self.calib_seconds, self.margin_low_db, self.margin_high_db, self.hyst_db, self.use_upper
        )

        if not self.sound_enabled and self.alarm_active:
            self.rearm_alarm()

        if self.config_win is not None and self.config_win.winfo_exists():
            self.config_win.destroy()
            self.config_win = None

    def recalibrate(self, auto: bool = False):
        # Si hay alarma, la cortamos antes de calibrar (si no, te recalibra con el WAV sonando)
        if self.alarm_active:
            self.rearm_alarm()

        tag = "AUTO-RECALIB (3 min)" if auto else "RECALIBRANDO..."
        self.lbl_status.config(text=f"{tag}... MANTÉN SILENCIO", bg=ACCENT_AMBER)
        self.lbl_calib.config(text=f"{tag}...", fg=ACCENT_AMBER)

        self.calibrating = True
        self.worker.request_recalibration()

    # ------------------------------------------------------------
    # ALARMA
    # ------------------------------------------------------------
    def _play_alarm_loop(self):
        try:
            import winsound
            if not ALERTA_WAV_PATH.exists():
                return
            winsound.PlaySound(
                str(ALERTA_WAV_PATH),
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
            )
        except Exception:
            pass

    def _stop_alarm(self):
        try:
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

    def trigger_alarm(self):
        if self.alarm_active or not self.sound_enabled:
            return
        self.alarm_active = True
        self.led_alert._led.config(bg=ACCENT_DANGER)
        self._play_alarm_loop()

    def rearm_alarm(self):
        self._stop_alarm()
        self.alarm_active = False
        self.led_alert._led.config(bg=LED_OFF)
        self.lbl_status.config(bg=ACCENT_SAFE)

    def test_alarm(self):
        if not ALERTA_WAV_PATH.exists():
            self.lbl_status.config(text="FALTA assets/alerta.wav", bg=ACCENT_AMBER)
            return
        try:
            import winsound
            winsound.PlaySound(str(ALERTA_WAV_PATH), winsound.SND_FILENAME | winsound.SND_ASYNC)
            self.lbl_status.config(text="PRUEBA ALARMA: debería sonar ahora", bg=ACCENT_AMBER)
        except Exception:
            self.lbl_status.config(text="ERROR: alerta.wav no reproducible (WAV PCM 16-bit)", bg=ACCENT_AMBER)

    # ------------------------------------------------------------
    # DIBUJO
    # ------------------------------------------------------------
    def _first_draw(self):
        self._draw_spectrum_grid()
        self._draw_energy_bar(None)
        self._draw_distance(None)

    def _draw_spectrum_grid(self):
        c = self.spec_canvas
        c.delete("all")
        w = max(c.winfo_width(), 300)
        h = max(c.winfo_height(), 220)

        m = 10
        x0, y0, x1, y1 = m, m, w - m, h - m

        c.create_rectangle(x0, y0, x1, y1, outline=GRID_DIM)
        for i in range(1, 4):
            y = y0 + i * (y1 - y0) / 4
            c.create_line(x0, y, x1, y, fill=GRID_DIM)
        for i in range(1, 4):
            x = x0 + i * (x1 - x0) / 4
            c.create_line(x, y0, x, y1, fill=GRID_DIM)

        c.create_text(x0, y1 + 2, text="0 Hz", fill=TEXT_MAIN, font=("Verdana", 9), anchor="nw")
        c.create_text(x1, y1 + 2, text="8000 Hz", fill=TEXT_MAIN, font=("Verdana", 9), anchor="ne")

    def _draw_spectrum(self, freqs, db_vals):
        self._draw_spectrum_grid()
        if freqs is None or db_vals is None or len(freqs) < 5:
            return

        c = self.spec_canvas
        w = max(c.winfo_width(), 300)
        h = max(c.winfo_height(), 220)

        m = 10
        x0, y0, x1, y1 = m, m, w - m, h - m

        fmin, fmax = 0.0, 8000.0
        dmin, dmax = -120.0, 0.0

        pts = []
        for f, db in zip(freqs, db_vals):
            ff = clamp(float(f), fmin, fmax)
            dd = clamp(float(db), dmin, dmax)
            x = x0 + (ff - fmin) / (fmax - fmin) * (x1 - x0)
            y = y1 - (dd - dmin) / (dmax - dmin) * (y1 - y0)
            pts.append((x, y))

        for i in range(len(pts) - 1):
            c.create_line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], fill=SPEC_LINE, width=2)

        bx0 = x0 + (self.f_low - fmin) / (fmax - fmin) * (x1 - x0)
        bx1 = x0 + (self.f_high - fmin) / (fmax - fmin) * (x1 - x0)
        c.create_rectangle(bx0, y0, bx1, y1, outline=ACCENT_AMBER, width=1)
        c.create_text((bx0 + bx1) / 2, y0 + 8, text="BANDA", fill=ACCENT_AMBER, font=("Verdana", 9, "bold"))

    def _draw_energy_bar(self, energy_db):
        """
        Canvas dominante + escala adaptativa alrededor de umbrales (si existen),
        con márgenes mínimos para que llene el recuadro.
        """
        c = self.energy_canvas
        c.delete("all")

        w = max(c.winfo_width(), 320)
        h = max(c.winfo_height(), 260)

        # MÍNIMO margen: así rellena el panel
        m = 1
        x0, y0, x1, y1 = m, m, w - m, h - m

        c.create_rectangle(x0, y0, x1, y1, outline=GRID_DIM)

        # Escala basada en umbrales si existen
        if self.thr_low is not None and self.thr_high is not None:
            vmin = self.thr_low - 20.0
            vmax = self.thr_high + 10.0
        else:
            vmin, vmax = -60.0, 60.0

        if energy_db is not None and energy_db < vmin + 5.0:
            vmin = energy_db - 12.0
            vmax = vmin + 60.0

        if vmax <= vmin:
            vmax = vmin + 10.0

        def y_from_val(val):
            val = clamp(float(val), vmin, vmax)
            return y1 - (val - vmin) / (vmax - vmin) * (y1 - y0)

        for i in range(1, 5):
            y = y0 + i * (y1 - y0) / 5
            c.create_line(x0, y, x1, y, fill=GRID_DIM)

        if energy_db is not None:
            y_fill = y_from_val(energy_db)
            fill_color = ACCENT_AMBER if self.detected else ACCENT_SAFE
            c.create_rectangle(x0, y_fill, x1, y1, fill=fill_color, outline="")

        if self.thr_low is not None:
            y_low = y_from_val(self.thr_low)
            c.create_line(x0, y_low, x1, y_low, fill=ACCENT_AMBER, width=2)
            c.create_text(x1 - 4, y_low - 10, text=f"THR LOW {self.thr_low:.2f} dB",
                          fill=ACCENT_AMBER, font=("Verdana", 9), anchor="e")

        if self.use_upper and (self.thr_high is not None):
            y_high = y_from_val(self.thr_high)
            c.create_line(x0, y_high, x1, y_high, fill=ACCENT_AMBER, width=2)
            c.create_text(x1 - 4, y_high - 10, text=f"THR HIGH {self.thr_high:.2f} dB",
                          fill=ACCENT_AMBER, font=("Verdana", 9), anchor="e")

    def _draw_distance(self, dist_m):
        c = self.dist_canvas
        c.delete("all")
        w = max(c.winfo_width(), 320)
        h = max(c.winfo_height(), 240)

        x0 = 70
        x1 = w - 70
        y_line = h * 0.60

        c.create_line(x0, y_line, x1, y_line, fill=TEXT_MAIN, width=4)

        for d in [0, 5, 10, 15]:
            x = x0 + (d / DIST_MAX_METROS) * (x1 - x0)
            c.create_line(x, y_line - 10, x, y_line + 10, fill=TEXT_MAIN, width=2)
            c.create_text(x, y_line + 22, text=f"{d} m", fill=TEXT_MAIN, font=("Verdana", 10))

        ship_x, ship_y = x0, y_line - 55
        if self.ship_img is not None:
            c.create_image(ship_x, ship_y, image=self.ship_img, anchor="center")

        if dist_m is None:
            return
        d_clamped = clamp(dist_m, 0.0, DIST_MAX_METROS)
        drone_x = x0 + (d_clamped / DIST_MAX_METROS) * (x1 - x0)
        drone_y = y_line - 55
        if self.drone_img is not None:
            c.create_image(drone_x, drone_y, image=self.drone_img, anchor="center")

    def _estimate_distance_from_energy(self, avg_db):
        # Mapeo monotónico usando thr_low/thr_high (si existen)
        if avg_db is None or self.thr_low is None:
            return None

        lo = self.thr_low
        hi = self.thr_high if (self.thr_high is not None) else (lo + 30.0)

        if hi <= lo:
            return None

        v = clamp(avg_db, lo, hi)
        t = (v - lo) / (hi - lo)
        return (1.0 - t) * 15.0 + t * 1.0

    # ------------------------------------------------------------
    # LOOP
    # ------------------------------------------------------------
    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_q.get_nowait()
                kind = msg[0]

                if kind == "MIC_OK":
                    self.led_mic._led.config(bg=ACCENT_SAFE)

                elif kind == "MIC_ERR":
                    self.led_mic._led.config(bg=ACCENT_DANGER)
                    self.lbl_status.config(text=msg[1], bg=ACCENT_AMBER)

                elif kind == "CALIB_DONE":
                    _, bm, bs, tl, th = msg
                    self.baseline_mean = bm
                    self.baseline_std = bs
                    self.thr_low = tl
                    self.thr_high = th
                    self.calibrating = False

                    if self.use_upper:
                        self.lbl_calib.config(
                            text=f"BASE {bm:.2f} dB | THR {tl:.2f} / {th:.2f} dB",
                            fg=ACCENT_SAFE
                        )
                    else:
                        self.lbl_calib.config(
                            text=f"BASE {bm:.2f} dB | THR {tl:.2f} dB",
                            fg=ACCENT_SAFE
                        )

                    self.lbl_status.config(text="UAS NO DETECTADO / FUERA DE PELIGRO", bg=ACCENT_SAFE)

                elif kind == "SPEC":
                    _, f, db = msg
                    self.led_dsp._led.config(bg=ACCENT_SAFE)
                    self._draw_spectrum(f, db)

                elif kind == "DET":
                    _, detected, avg_db, tl, th, bm, bs, calibrating = msg
                    self.detected = bool(detected)
                    self.avg_energy_db = float(avg_db)

                    self.thr_low = tl
                    self.thr_high = th
                    self.baseline_mean = bm
                    self.baseline_std = bs
                    self.calibrating = bool(calibrating)

                    self.lbl_energy_value.config(text=f"ENERGÍA: {self.avg_energy_db:6.2f} dB")
                    self._draw_energy_bar(self.avg_energy_db)

                    if self.calibrating and self.adaptive:
                        self.lbl_calib.config(text="CALIBRANDO...", fg=ACCENT_AMBER)
                        self.lbl_status.config(text="CALIBRANDO... MANTÉN SILENCIO", bg=ACCENT_AMBER)
                        self.lbl_dist_value.config(text="DISTANCIA: ---")
                        self._draw_distance(None)
                        continue

                    dist = self._estimate_distance_from_energy(self.avg_energy_db) if self.detected else None
                    self._draw_distance(dist)
                    self.lbl_dist_value.config(text="DISTANCIA: ---" if dist is None else f"DISTANCIA: ≈ {dist:4.1f} m")

                    # estado + alarma
                    if self.detected:
                        self.lbl_status.config(text="UAS DETECTADO (UMBRAL ADAPTATIVO)", bg=ACCENT_AMBER)
                        if not self.alarm_active:
                            self.trigger_alarm()
                    else:
                        self.lbl_status.config(text="UAS NO DETECTADO / FUERA DE PELIGRO")
                        if not self.alarm_active:
                            self.lbl_status.config(bg=ACCENT_SAFE)

        except queue.Empty:
            pass

        self.after_ids.append(self.root.after(60, self._poll_queue))

    def _blink_loop(self):
        if self.alarm_active and self.blink_enabled:
            self.blink_on = not self.blink_on
            color = ACCENT_DANGER if self.blink_on else ACCENT_DANGER_DARK
            self.lbl_status.config(bg=color)
            self.led_alert._led.config(bg=color)
        self.after_ids.append(self.root.after(250, self._blink_loop))

    # ------------------------------------------------------------
    # CLOSE
    # ------------------------------------------------------------
    def on_close(self):
        self.stop_event.set()

        for aid in self.after_ids:
            try:
                self.root.after_cancel(aid)
            except Exception:
                pass
        self.after_ids.clear()

        try:
            self.rearm_alarm()
        except Exception:
            pass

        try:
            self.root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = DetectorUI(root)
    root.mainloop()
