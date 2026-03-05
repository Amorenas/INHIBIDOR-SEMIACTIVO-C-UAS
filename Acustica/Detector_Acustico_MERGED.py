# Detector_Acustico_MERGED.py
# ------------------------------------------------------------
# APP UNIFICADA: DETECCIÓN ACÚSTICA + DOA (MA-USB8 + SIPEED 6+1)
#
# Requisitos (Windows):
#   py -m pip install numpy scipy sounddevice pyserial pillow
#
# Estructura recomendada:
#   Desktop/Detector_Acustico/
#       Detector_Acustico_MERGED.py
#       assets/
#           alerta.wav        (WAV PCM 16-bit recomendado)
#           buque.png         (opcional)
#           dron.png          (opcional)
#
# NOTA:
# - La detección usa: multi-banda + ratios + SNR + persistencia + umbral adaptativo
# - Auto-recalibración inteligente: cada 3 min SOLO si procede (sin detección, bajo y estable)
# - Watchdog de micro: reconecta si se cae
# - DOA: gating por detección, filtro temporal adaptativo por confianza, rechazo frames malos,
#        normalización hotmap, sector por percentil 80%, bimodal -> ambiguo, sectores navales, estela, LOCK.
# ------------------------------------------------------------

import time
import math
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from collections import deque

import tkinter as tk
from tkinter import ttk

import numpy as np

# Audio / DSP
import sounddevice as sd
from scipy.signal import butter, filtfilt, spectrogram, welch

# DOA serial
import serial
import serial.tools.list_ports

# Assets
from PIL import Image, ImageTk


# ============================================================
# RUTAS (assets/)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

BUQUE_PATH = ASSETS_DIR / "buque.png"
DRON_PATH = ASSETS_DIR / "dron.png"
ALERTA_WAV_PATH = ASSETS_DIR / "alerta.wav"


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
ACCENT_CYAN = "#2FD4FF"

STATUS_STRIP_BG = "#0B1118"
LED_OFF = "#555555"

GRID_DIM = "#1E2E3A"
SPEC_LINE = "#00FF7F"

HOTMAP_LINE = "#00FF7F"
HOTMAP_BG = "#102636"
COMPASS_LINE = "#6FA3C8"

FONT_TITLE = ("Verdana", 26, "bold")
FONT_PANEL = ("Verdana", 16, "bold")
FONT_MONO = ("Consolas", 12, "bold")


def clamp(v, a, b):
    return max(a, min(b, v))


def safe_db10(x, eps=1e-12):
    return 10.0 * np.log10(np.maximum(x, eps))


# ============================================================
# DSP CONFIG
# ============================================================
RATE = 44100
CHANNELS = 1

# Ventana para decisión (segundos) y periodo de actualización (s)
WINDOW_SECONDS_DEFAULT = 4.0
DET_PERIOD_DEFAULT = 0.25

# Bandas multi-banda (Hz)
B1_LOW, B1_HIGH = 1000.0, 3000.0
B2_LOW, B2_HIGH = 3000.0, 6000.0
B3_LOW, B3_HIGH = 6000.0, 9000.0

# Filtro “ancho” previo para spectrograma (reduce basura)
PRE_BPF_LOW = 800.0
PRE_BPF_HIGH = 9500.0
BP_ORDER_DEFAULT = 5

# Persistencia temporal: N de M
PERSIST_M_DEFAULT = 5
PERSIST_N_DEFAULT = 3

# Umbral adaptativo
ADAPTIVE_DEFAULT = True
CALIB_SECONDS_DEFAULT = 3.0

# Baseline + margen => thr_low
MARGIN_LOW_DB_DEFAULT = 8.0
HYST_DB_DEFAULT = 2.5

# Ratios / SNR gating (en dB)
R1_MIN_DB_DEFAULT = 4.0   # E36 - E13
R2_MIN_DB_DEFAULT = 2.0   # E36 - E69
SNR_MIN_DB_DEFAULT = 8.0  # E36 - mean(E13,E69)

USE_RATIOS_DEFAULT = True
USE_SNR_DEFAULT = True

# Modulación/armónicos (mínimo viable)
USE_PEAK_STABILITY_DEFAULT = True
PEAK_STD_MAX_HZ_DEFAULT = 140.0
PEAK_PROM_MIN_DB_DEFAULT = 10.0  # prominencia vs mediana del espectro

# Banda adaptativa (tracking simple)
USE_BAND_TRACKING_DEFAULT = True
TRACK_BAND_HALF_WIDTH_HZ_DEFAULT = 1500.0
TRACK_SMOOTH_DEFAULT = 0.25  # 0..1, más alto => seguimiento más rápido

# Auto-recalibración inteligente
AUTO_RECAL_EVERY_S_DEFAULT = 180.0
AUTO_LOW_SECONDS_DEFAULT = 20.0
AUTO_MARGIN_SAFE_DB_DEFAULT = 3.0
AUTO_STABLE_STD_DB_DEFAULT = 2.0

# Watchdog micro
MIC_DEAD_SECONDS_DEFAULT = 2.5
MIC_RETRY_SECONDS_DEFAULT = 1.2

# Forzar input device (None => auto)
FORCE_INPUT_DEVICE_INDEX = None


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
    Heurística para elegir el input USB.
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


@dataclass
class DetParams:
    window_seconds: float = WINDOW_SECONDS_DEFAULT
    det_period: float = DET_PERIOD_DEFAULT

    bp_order: int = BP_ORDER_DEFAULT

    adaptive: bool = ADAPTIVE_DEFAULT
    calib_seconds: float = CALIB_SECONDS_DEFAULT
    margin_low_db: float = MARGIN_LOW_DB_DEFAULT
    hyst_db: float = HYST_DB_DEFAULT

    use_ratios: bool = USE_RATIOS_DEFAULT
    r1_min_db: float = R1_MIN_DB_DEFAULT
    r2_min_db: float = R2_MIN_DB_DEFAULT

    use_snr: bool = USE_SNR_DEFAULT
    snr_min_db: float = SNR_MIN_DB_DEFAULT

    persist_m: int = PERSIST_M_DEFAULT
    persist_n: int = PERSIST_N_DEFAULT

    use_peak_stability: bool = USE_PEAK_STABILITY_DEFAULT
    peak_std_max_hz: float = PEAK_STD_MAX_HZ_DEFAULT
    peak_prom_min_db: float = PEAK_PROM_MIN_DB_DEFAULT

    use_band_tracking: bool = USE_BAND_TRACKING_DEFAULT
    track_half_width_hz: float = TRACK_BAND_HALF_WIDTH_HZ_DEFAULT
    track_smooth: float = TRACK_SMOOTH_DEFAULT

    auto_recal_every_s: float = AUTO_RECAL_EVERY_S_DEFAULT
    auto_low_seconds: float = AUTO_LOW_SECONDS_DEFAULT
    auto_margin_safe_db: float = AUTO_MARGIN_SAFE_DB_DEFAULT
    auto_stable_std_db: float = AUTO_STABLE_STD_DB_DEFAULT

    mic_dead_seconds: float = MIC_DEAD_SECONDS_DEFAULT
    mic_retry_seconds: float = MIC_RETRY_SECONDS_DEFAULT


# ============================================================
# AUDIO WORKER (detección + espectro)
# ============================================================
class AudioWorker(threading.Thread):
    """
    Publica mensajes:
      ("MIC_OK", msg)
      ("MIC_ERR", msg)
      ("SPEC", freqs, db_vals)           -> espectro para UI
      ("DET",
         detected_final, detected_raw,
         E13, E36, E69,
         R1, R2, SNR,
         avg_db_main, thr_low,
         baseline_mean, baseline_std,
         band_low, band_high,
         calibrating,
         conf_det
      )
      ("CALIB_DONE", baseline_mean, baseline_std, thr_low)
    """
    def __init__(self, out_q: queue.Queue, stop_event: threading.Event, params: DetParams):
        super().__init__(daemon=True)
        self.q = out_q
        self.stop_event = stop_event
        self.lock = threading.Lock()

        self.params = params

        # runtime thresholds
        self.baseline_mean = None
        self.baseline_std = None
        self.thr_low = None

        self._calibrating = True
        self._calib_start_ts = time.time()
        self._calib_values = []

        # detection state (hysteresis + persistence)
        self._det_state = False
        self._persist = deque(maxlen=max(3, params.persist_m))

        # device index
        self.device_index = FORCE_INPUT_DEVICE_INDEX
        if self.device_index is None:
            auto_usb = pick_usb_input_device_index()
            if auto_usb is not None:
                self.device_index = auto_usb

        # buffers
        self._buf = np.zeros(max(2048, int(RATE * params.window_seconds)), dtype=np.float32)
        self._buf_fill = 0

        self._last_det_ts = 0.0
        self._last_audio_ts = time.time()

        # band tracking
        self._main_band_low = B2_LOW
        self._main_band_high = B2_HIGH

        # peak tracking
        self._peak_hist = deque(maxlen=12)  # ~3s si det_period=0.25
        self._peak_last = None

        # auto recal tracking
        self._last_calib_done_ts = 0.0
        self._auto_low_hist = deque(maxlen=200)  # guardará (ts, avg_db_main)
        self._recalibrate_flag = False

        # reconnection loop state
        self._need_reopen = False

    def push(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            pass

    def request_recalibration(self):
        with self.lock:
            self._recalibrate_flag = True

    def _start_calibration(self):
        self._calibrating = True
        self._calib_start_ts = time.time()
        self._calib_values = []
        self.baseline_mean = None
        self.baseline_std = None
        self.thr_low = None
        self._det_state = False
        self._persist.clear()
        self._peak_hist.clear()

    def _finalize_calibration(self):
        if len(self._calib_values) < 5:
            self.baseline_mean = -30.0
            self.baseline_std = 2.0
        else:
            arr = np.array(self._calib_values, dtype=np.float32)
            self.baseline_mean = float(np.mean(arr))
            self.baseline_std = float(np.std(arr))

        self.thr_low = self.baseline_mean + self.params.margin_low_db
        self._calibrating = False
        self._last_calib_done_ts = time.time()
        self.push(("CALIB_DONE", self.baseline_mean, self.baseline_std, self.thr_low))

    def _do_auto_recal_if_needed(self, detected_final, avg_db_main):
        """Auto-recal cada X s, solo si: no detección, bajo durante Y s y estable."""
        p = self.params
        now = time.time()

        # registrar histórico
        self._auto_low_hist.append((now, float(avg_db_main) if avg_db_main is not None else None))

        # solo cada auto_recal_every_s
        if self._last_calib_done_ts <= 0:
            return
        if (now - self._last_calib_done_ts) < p.auto_recal_every_s:
            return
        if detected_final:
            return
        if self.thr_low is None:
            return

        # filtrar ventana de los últimos auto_low_seconds
        t0 = now - p.auto_low_seconds
        vals = [v for (t, v) in self._auto_low_hist if t >= t0 and v is not None]
        if len(vals) < max(6, int(p.auto_low_seconds / p.det_period * 0.5)):
            return

        vals = np.array(vals, dtype=np.float32)
        # condición "bajo": por debajo de thr_low - margin_safe
        if np.mean(vals) >= (self.thr_low - p.auto_margin_safe_db):
            return

        # condición "estable": std bajo
        if float(np.std(vals)) > p.auto_stable_std_db:
            return

        # OK -> recalibrar
        self._start_calibration()

    def _compute_band_means_db(self, Sxx_dB, f_sp, bands):
        """
        bands: [(low,high), ...]
        devuelve [mean_db,...]
        """
        out = []
        for (lo, hi) in bands:
            low_idx = int(np.searchsorted(f_sp, lo))
            high_idx = int(np.searchsorted(f_sp, hi))
            low_idx = clamp(low_idx, 0, len(f_sp) - 1)
            high_idx = clamp(high_idx, low_idx + 1, len(f_sp))
            out.append(float(np.mean(Sxx_dB[low_idx:high_idx, :])))
        return out

    def _spectral_peak_features(self, x):
        """
        Devuelve:
          peak_freq (Hz), peak_db, peak_prom_db
        """
        try:
            f, Pxx = welch(x, fs=RATE, nperseg=min(2048, len(x)))
            mask = (f >= 800.0) & (f <= 9000.0)
            f2 = f[mask]
            P2 = Pxx[mask]
            if len(P2) < 10:
                return None, None, None
            Pdb = safe_db10(P2, eps=1e-20)
            peak_i = int(np.argmax(Pdb))
            peak_freq = float(f2[peak_i])
            peak_db = float(Pdb[peak_i])
            med = float(np.median(Pdb))
            prom = peak_db - med
            return peak_freq, peak_db, prom
        except Exception:
            return None, None, None

    def _update_main_band_tracking(self, peak_freq):
        """Tracking simple: banda principal centrada en el pico (suavizada)."""
        if not self.params.use_band_tracking:
            self._main_band_low = B2_LOW
            self._main_band_high = B2_HIGH
            return

        if peak_freq is None:
            return

        half = float(self.params.track_half_width_hz)
        target_low = clamp(peak_freq - half, B1_LOW, B3_HIGH - 500.0)
        target_high = clamp(peak_freq + half, target_low + 800.0, B3_HIGH)

        a = float(self.params.track_smooth)
        self._main_band_low = (1.0 - a) * self._main_band_low + a * target_low
        self._main_band_high = (1.0 - a) * self._main_band_high + a * target_high

    def _open_stream_loop(self):
        """
        Abre InputStream y procesa callbacks. Si se cae, devuelve para reintentar.
        """
        # consulta dispositivo
        try:
            if self.device_index is None:
                in_idx = sd.default.device[0]
                dinfo = sd.query_devices(in_idx)
                self.push(("MIC_OK", f"INPUT default idx={in_idx}: {dinfo.get('name', '?')}"))
            else:
                dinfo = sd.query_devices(self.device_index)
                self.push(("MIC_OK", f"INPUT idx={self.device_index}: {dinfo.get('name', '?')}"))
        except Exception as e:
            self.push(("MIC_ERR", f"No puedo consultar dispositivos: {e}"))

        # filtros pre-bpf
        try:
            b_pre, a_pre = butter_bandpass(PRE_BPF_LOW, PRE_BPF_HIGH, RATE, order=self.params.bp_order)
        except Exception:
            b_pre, a_pre = None, None

        def callback(indata, frames, time_info, status):
            if self.stop_event.is_set():
                raise sd.CallbackStop()

            self._last_audio_ts = time.time()
            x = indata[:, 0].astype(np.float32)

            # ----- ESPECTRO PARA UI (rápido) -----
            try:
                f, Pxx = welch(x, fs=RATE, nperseg=min(1024, len(x)))
                Pxx_db = safe_db10(Pxx, eps=1e-20)
                mask = f <= 8000
                self.push(("SPEC", f[mask], Pxx_db[mask]))
            except Exception:
                pass

            # ----- Buffer ventana -----
            nbuf = len(self._buf)
            n = len(x)

            # recalibración manual pendiente
            with self.lock:
                do_recal = self._recalibrate_flag
                if do_recal:
                    self._recalibrate_flag = False

            if do_recal and self.params.adaptive:
                self._start_calibration()

            if n >= nbuf:
                self._buf[:] = x[-nbuf:]
                self._buf_fill = nbuf
            else:
                remaining = nbuf - self._buf_fill
                if n <= remaining:
                    self._buf[self._buf_fill:self._buf_fill + n] = x
                    self._buf_fill += n
                else:
                    shift = n - remaining
                    self._buf[:-shift] = self._buf[shift:]
                    self._buf[-n:] = x
                    self._buf_fill = nbuf

            # ----- cálculo rate-limited -----
            now = time.time()
            if now - self._last_det_ts < self.params.det_period:
                return
            self._last_det_ts = now

            if self._buf_fill < len(self._buf):
                return

            buf_copy = self._buf.copy()

            # pipeline: float -> int16-like -> pre bandpass -> spectrogram -> bandas
            try:
                x_i16 = np.int16(np.clip(buf_copy, -1.0, 1.0) * 32767)
                xf = x_i16.astype(np.float32)

                # pre-filter (ancho)
                if b_pre is not None:
                    xf = filtfilt(b_pre, a_pre, xf)

                # peak features para tracking / estabilidad
                peak_f, peak_db, peak_prom = self._spectral_peak_features(xf)
                if peak_f is not None:
                    self._peak_hist.append(peak_f)
                self._update_main_band_tracking(peak_f)

                # spectrogram
                f_sp, t_sp, Sxx = spectrogram(xf, fs=RATE)
                Sxx_dB = safe_db10(Sxx, eps=1e-10)

                # multi-banda fija
                E13, E36, E69 = self._compute_band_means_db(
                    Sxx_dB, f_sp,
                    [(B1_LOW, B1_HIGH), (B2_LOW, B2_HIGH), (B3_LOW, B3_HIGH)]
                )

                # banda principal (tracking) para decisión (avg_db_main)
                band_low = float(self._main_band_low) if self.params.use_band_tracking else float(B2_LOW)
                band_high = float(self._main_band_high) if self.params.use_band_tracking else float(B2_HIGH)
                avg_db_main, = self._compute_band_means_db(Sxx_dB, f_sp, [(band_low, band_high)])

            except Exception:
                return

            # ratios / snr (en dB son restas)
            R1 = float(E36 - E13)
            R2 = float(E36 - E69)
            SNR = float(E36 - 0.5 * (E13 + E69))

            # calibración adaptativa sobre avg_db_main
            if self.params.adaptive:
                if self._calibrating:
                    self._calib_values.append(float(avg_db_main))
                    if (now - self._calib_start_ts) >= self.params.calib_seconds:
                        self._finalize_calibration()

            # detección raw
            detected_raw = False
            thr_low = self.thr_low

            # peak stability gating
            peak_ok = True
            if self.params.use_peak_stability:
                peak_ok = False
                if peak_prom is not None and peak_prom >= self.params.peak_prom_min_db:
                    if len(self._peak_hist) >= 6:
                        pf_std = float(np.std(np.array(self._peak_hist, dtype=np.float32)))
                        if pf_std <= self.params.peak_std_max_hz:
                            peak_ok = True
                    else:
                        # no hay suficiente historia: no bloquees del todo, pero penaliza
                        peak_ok = True

            if thr_low is not None and (not self._calibrating):
                # condición principal + ratios + snr + peak
                cond = (avg_db_main >= thr_low)

                if self.params.use_ratios:
                    cond = cond and (R1 >= self.params.r1_min_db) and (R2 >= self.params.r2_min_db)

                if self.params.use_snr:
                    cond = cond and (SNR >= self.params.snr_min_db)

                cond = cond and peak_ok

                detected_raw = bool(cond)

                # histeresis (sobre umbral)
                if not self._det_state:
                    if detected_raw:
                        self._det_state = True
                else:
                    if avg_db_main <= (thr_low - self.params.hyst_db):
                        self._det_state = False

            # persistencia N de M (sobre estado histerético)
            self._persist.append(1 if self._det_state else 0)
            s = sum(self._persist)
            detected_final = (s >= self.params.persist_n)

            # confianza detección (0..1) basada en SNR y persistencia
            conf_det = 0.0
            if thr_low is not None and not self._calibrating:
                # SNR normalizado
                sn = clamp((SNR - self.params.snr_min_db) / 12.0, 0.0, 1.0) if self.params.use_snr else 0.6
                # persistencia
                pr = clamp(s / max(1, self.params.persist_m), 0.0, 1.0)
                # margen sobre umbral
                mg = clamp((avg_db_main - thr_low) / 10.0, 0.0, 1.0)
                conf_det = float(0.45 * sn + 0.35 * pr + 0.20 * mg)

            # auto recal inteligente
            if self.params.adaptive and (thr_low is not None) and (avg_db_main is not None):
                self._do_auto_recal_if_needed(detected_final, avg_db_main)

            self.push((
                "DET",
                bool(detected_final),
                bool(detected_raw),
                float(E13), float(E36), float(E69),
                float(R1), float(R2), float(SNR),
                float(avg_db_main),
                float(thr_low) if thr_low is not None else None,
                float(self.baseline_mean) if self.baseline_mean is not None else None,
                float(self.baseline_std) if self.baseline_std is not None else None,
                float(band_low), float(band_high),
                bool(self._calibrating),
                float(conf_det)
            ))

        # abrir stream
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

    def run(self):
        # al arrancar, calibra
        if self.params.adaptive:
            self._start_calibration()
        else:
            self._calibrating = False

        # loop de conexión/reconexión
        while not self.stop_event.is_set():
            self._open_stream_loop()

            if self.stop_event.is_set():
                break

            # si se cae, espera y reintenta
            time.sleep(self.params.mic_retry_seconds)

    def is_audio_dead(self):
        return (time.time() - self._last_audio_ts) > self.params.mic_dead_seconds


# ============================================================
# DOA WORKER (serial hotmap)
# ============================================================
@dataclass
class DoaParams:
    baud: int = 115200
    header_len: int = 16
    header_byte: int = 0xFF
    frame_len: int = 256  # 16x16

    # rechazo frames
    contrast_min: float = 10.0
    lobe_rel: float = 0.70
    lobe_min_count: int = 4

    # normalización
    gamma: float = 1.45

    # bimodal
    bimodal_ratio: float = 0.85
    peak_min_separation_cells: int = 3

    # sector percentil
    sector_weight_percent: float = 0.80

    # smoothing adaptativo por confianza
    alpha_min: float = 0.05
    alpha_max: float = 0.35

    # estela
    trail_len: int = 120


class DOAWorker(threading.Thread):
    """
    Lee hotmap del MA-USB8 por COM.
    Publica:
      ("DOA_OK", msg)
      ("DOA_ERR", msg)
      ("HOTMAP", hotmap_16x16_uint8, ts)
    """
    def __init__(self, out_q: queue.Queue, stop_event: threading.Event, port: str, params: DoaParams):
        super().__init__(daemon=True)
        self.q = out_q
        self.stop_event = stop_event
        self.port = port
        self.params = params

        self._ser = None

    def push(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            pass

    def _close(self):
        try:
            if self._ser:
                self._ser.close()
        except Exception:
            pass
        self._ser = None

    def run(self):
        p = self.params
        header = bytes([p.header_byte]) * p.header_len

        try:
            self._ser = serial.Serial(self.port, p.baud, timeout=0.2)
            self.push(("DOA_OK", f"DOA conectado en {self.port} @ {p.baud}"))
        except Exception as e:
            self.push(("DOA_ERR", f"No se pudo abrir {self.port}: {e}"))
            self._close()
            return

        buf = bytearray()

        try:
            while not self.stop_event.is_set():
                try:
                    chunk = self._ser.read(512)
                except Exception as e:
                    self.push(("DOA_ERR", f"Error leyendo serial: {e}"))
                    break

                if not chunk:
                    continue

                buf.extend(chunk)

                # buscar header
                while True:
                    idx = buf.find(header)
                    if idx < 0:
                        # mantener buffer acotado
                        if len(buf) > 4096:
                            buf = buf[-1024:]
                        break

                    # descartar antes del header
                    if idx > 0:
                        buf = buf[idx:]

                    # ¿hay frame completo?
                    need = p.header_len + p.frame_len
                    if len(buf) < need:
                        break

                    # consumir header + frame
                    frame = buf[p.header_len:need]
                    buf = buf[need:]

                    if len(frame) != p.frame_len:
                        continue

                    hm = np.frombuffer(frame, dtype=np.uint8).reshape((16, 16)).copy()
                    self.push(("HOTMAP", hm, time.time()))
        finally:
            self._close()


# ============================================================
# DOA ESTIMADOR (hotmap -> theta + sector + conf)
# ============================================================
def _cell_angle_rad(i, j, cx=7.5, cy=7.5):
    """
    Mapeo celda->ángulo (0 rad = arriba / "N").
    i: fila (0..15), j: col (0..15)
    """
    dx = (j - cx)
    dy = (i - cy)
    # Queremos 0 arriba: usar atan2(dx, -dy)
    ang = math.atan2(dx, -dy)
    if ang < 0:
        ang += 2 * math.pi
    return ang


def _minimal_arc_containing_weight(angles, weights, frac=0.80):
    """
    Devuelve (start_rad, end_rad, width_rad) del arco mínimo que contiene frac del peso.
    angles en [0,2pi), weights >=0.
    """
    if len(angles) < 3:
        return None

    ang = np.array(angles, dtype=np.float64)
    w = np.array(weights, dtype=np.float64)
    wsum = float(np.sum(w))
    if wsum <= 0:
        return None

    # ordenar por ángulo
    idx = np.argsort(ang)
    ang = ang[idx]
    w = w[idx]

    # duplicar para circularidad
    ang2 = np.concatenate([ang, ang + 2 * math.pi])
    w2 = np.concatenate([w, w])

    target = frac * wsum
    # ventana deslizante con 2 punteros
    best = None
    j = 0
    acc = 0.0
    for i in range(len(ang)):
        if j < i:
            j = i
            acc = 0.0
        while j < i + len(ang) and acc < target:
            acc += float(w2[j])
            j += 1
        if acc >= target:
            start = float(ang2[i])
            end = float(ang2[j - 1])
            width = end - start
            if best is None or width < best[2]:
                best = (start, end, width)
        acc -= float(w2[i])

    if best is None:
        return None

    start, end, width = best
    # normalizar start/end a [0,2pi)
    start = start % (2 * math.pi)
    end = end % (2 * math.pi)
    return start, end, width


def naval_sector_label(deg):
    """
    8 sectores navales:
      Proa (N), Proa-Estribor (NE), Estribor (E), Popa-Estribor (SE),
      Popa (S), Popa-Babor (SW), Babor (W), Proa-Babor (NW)
    """
    deg = deg % 360.0
    labels = [
        "PROA", "PROA-ESTRIBOR", "ESTRIBOR", "POPA-ESTRIBOR",
        "POPA", "POPA-BABOR", "BABOR", "PROA-BABOR"
    ]
    idx = int(((deg + 22.5) % 360) // 45)
    return labels[idx]


# ============================================================
# UI PRINCIPAL (merge)
# ============================================================
class MergedUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DETECTOR_ACUSTICO - MERGED (DETECCIÓN + DOA)")
        self.root.configure(bg=BG_MAIN)
        self.root.geometry("1600x820")

        # colas / stop
        self.stop_event = threading.Event()
        self.msg_q = queue.Queue(maxsize=140)

        # assets
        self.ship_img = None
        self.drone_img = None
        self._load_assets()

        # params
        self.det_params = DetParams()
        self.doa_params = DoaParams()

        # estado detección
        self.det_detected = False
        self.det_raw = False
        self.E13 = self.E36 = self.E69 = None
        self.R1 = self.R2 = self.SNR = None
        self.avg_db_main = None
        self.thr_low = None
        self.baseline_mean = None
        self.baseline_std = None
        self.band_low = B2_LOW
        self.band_high = B2_HIGH
        self.det_calibrating = True
        self.conf_det = 0.0

        # estado alarmado
        self.sound_enabled = True
        self.blink_enabled = True
        self.alarm_active = False
        self.blink_on = True

        # estado DOA
        self.doa_connected = False
        self.hotmap = None
        self.last_hotmap_ts = 0.0

        self.doa_theta = None
        self.doa_sector_width_deg = None
        self.doa_conf = 0.0
        self.doa_ambiguous = False
        self.doa_lock_state = "BUSCANDO"
        self._doa_last_good_theta = None
        self._doa_vf = np.array([1.0, 0.0], dtype=np.float64)  # vector filtrado
        self._doa_trail = deque(maxlen=self.doa_params.trail_len)
        self._doa_good_streak_start = None

        # workers
        self.audio_worker = AudioWorker(self.msg_q, self.stop_event, self.det_params)
        self.audio_worker.start()

        self.doa_worker = None  # se crea al conectar

        # after ids para cancelar
        self.after_ids = []
        self.config_win = None

        # UI build
        self._build_ui()

        # loops
        self.after_ids.append(self.root.after(200, self._first_draw))
        self.after_ids.append(self.root.after(60, self._poll_queue))
        self.after_ids.append(self.root.after(220, self._blink_loop))
        self.after_ids.append(self.root.after(500, self._watchdog_loop))

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
    # UI BUILD
    # ------------------------------------------------------------
    def _build_ui(self):
        # TOP BAR
        title_frame = tk.Frame(self.root, bg=BG_MAIN)
        title_frame.pack(pady=(6, 0), fill="x")

        tk.Label(
            title_frame, text="DETECTOR_ACUSTICO (MERGED)",
            font=FONT_TITLE, fg=TEXT_MAIN, bg=BG_MAIN
        ).pack(side="left", padx=12)

        right_box = tk.Frame(title_frame, bg=BG_MAIN)
        right_box.pack(side="right", padx=10)

        tk.Button(
            right_box, text="CONFIGURACIÓN",
            font=("Verdana", 10, "bold"),
            command=self.open_config,
            bg="#29323C", fg="white",
            activebackground="#3C4A58", activeforeground="white",
            relief="raised", bd=2
        ).pack(side="left", padx=6)

        tk.Button(
            right_box, text="RECALIBRAR UMBRAL",
            font=("Verdana", 10, "bold"),
            command=self.recalibrate,
            bg="#29323C", fg="white",
            activebackground="#3C4A58", activeforeground="white",
            relief="raised", bd=2
        ).pack(side="left", padx=6)

        tk.Button(
            right_box, text="REARMAR / SILENCIAR",
            font=("Verdana", 10, "bold"),
            command=self.rearm_alarm,
            bg="#29323C", fg="white",
            activebackground="#3C4A58", activeforeground="white",
            relief="raised", bd=2
        ).pack(side="left", padx=6)

        # STATUS STRIP
        status_strip = tk.Frame(self.root, bg=STATUS_STRIP_BG)
        status_strip.pack(fill="x", padx=10, pady=(6, 10))

        self._create_status_block(status_strip, "MODO", "DETECCIÓN + DOA", 0)
        self._create_status_block(status_strip, "SENSOR", "MIC USB (LOCAL)", 1)

        self.lbl_calib = tk.Label(
            status_strip, text="CALIBRANDO...",
            font=("Consolas", 11, "bold"),
            bg=STATUS_STRIP_BG, fg=ACCENT_AMBER
        )
        self.lbl_calib.grid(row=0, column=2, padx=10, sticky="w")

        # LEDs
        leds_frame = tk.Frame(status_strip, bg=STATUS_STRIP_BG)
        leds_frame.grid(row=0, column=3, padx=10, sticky="e")

        self.led_mic = self._create_led(leds_frame, "MIC", LED_OFF)
        self.led_mic.pack(side="left", padx=5)

        self.led_dsp = self._create_led(leds_frame, "DSP", LED_OFF)
        self.led_dsp.pack(side="left", padx=5)

        self.led_doa = self._create_led(leds_frame, "DOA", LED_OFF)
        self.led_doa.pack(side="left", padx=5)

        self.led_alert = self._create_led(leds_frame, "ALERTA", LED_OFF)
        self.led_alert.pack(side="left", padx=5)

        # CENTER GRID: 3 columnas
        center = tk.Frame(self.root, bg=BG_MAIN)
        center.pack(fill="both", expand=True, padx=18, pady=8)
        center.columnconfigure(0, weight=1)
        center.columnconfigure(1, weight=1)
        center.columnconfigure(2, weight=1)
        center.rowconfigure(0, weight=1)

        # Panel Spectrum
        self.spec_frame = tk.Frame(center, bg=PANEL_BG, highlightbackground=FRAME_COLOR, highlightthickness=3)
        self.spec_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.spec_frame.columnconfigure(0, weight=1)
        self.spec_frame.rowconfigure(1, weight=1)

        tk.Label(self.spec_frame, text="ANALIZADOR DE ESPECTRO (dB)",
                 font=FONT_PANEL, bg=PANEL_BG, fg=TEXT_MAIN).grid(row=0, column=0, pady=(8, 4), sticky="ew")

        self.spec_canvas = tk.Canvas(self.spec_frame, bg=PANEL_BG, highlightthickness=0)
        self.spec_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Panel Energía / detección
        self.energy_frame = tk.Frame(center, bg=PANEL_BG, highlightbackground=FRAME_COLOR, highlightthickness=3)
        self.energy_frame.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self.energy_frame.columnconfigure(0, weight=1)
        self.energy_frame.rowconfigure(1, weight=1)

        tk.Label(self.energy_frame, text="DETECCIÓN (MULTI-BANDA + ADAPTATIVO)",
                 font=FONT_PANEL, bg=PANEL_BG, fg=TEXT_MAIN).grid(row=0, column=0, pady=(8, 4), sticky="ew")

        self.energy_canvas = tk.Canvas(self.energy_frame, bg=PANEL_BG, highlightthickness=0)
        self.energy_canvas.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        self.lbl_energy = tk.Label(self.energy_frame, text="E_MAIN: --- dB | THR: --- dB",
                                   font=("Consolas", 13, "bold"), bg=PANEL_BG, fg=TEXT_MAIN)
        self.lbl_energy.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 4))

        self.lbl_features = tk.Label(
            self.energy_frame,
            text="E13 --- | E36 --- | E69 --- | R1 --- | R2 --- | SNR --- | PERSIST ---",
            font=("Consolas", 11, "bold"),
            bg=PANEL_BG, fg=ACCENT_CYAN
        )
        self.lbl_features.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))

        # Panel DOA
        self.doa_frame = tk.Frame(center, bg=PANEL_BG, highlightbackground=FRAME_COLOR, highlightthickness=3)
        self.doa_frame.grid(row=0, column=2, sticky="nsew", padx=8, pady=8)
        self.doa_frame.columnconfigure(0, weight=1)
        self.doa_frame.rowconfigure(1, weight=1)
        self.doa_frame.rowconfigure(2, weight=1)

        top_doa = tk.Frame(self.doa_frame, bg=PANEL_BG)
        top_doa.grid(row=0, column=0, sticky="ew", pady=(8, 4))
        top_doa.columnconfigure(0, weight=1)

        tk.Label(top_doa, text="DOA (DIRECCIÓN DE LLEGADA)",
                 font=FONT_PANEL, bg=PANEL_BG, fg=TEXT_MAIN).grid(row=0, column=0, sticky="w", padx=10)

        # puerto COM + conectar
        self.cmb_ports = ttk.Combobox(top_doa, values=self._list_serial_ports(), width=18, state="readonly")
        self.cmb_ports.grid(row=0, column=1, sticky="e", padx=(0, 8))
        self.btn_doa = tk.Button(
            top_doa, text="CONECTAR",
            command=self.toggle_doa_connection,
            font=("Verdana", 10, "bold"),
            bg="#29323C", fg="white",
            activebackground="#3C4A58", activeforeground="white",
            relief="raised", bd=2
        )
        self.btn_doa.grid(row=0, column=2, sticky="e", padx=(0, 10))

        # Compass + info
        self.compass_canvas = tk.Canvas(self.doa_frame, bg=PANEL_BG, highlightthickness=0)
        self.compass_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 6))

        self.hotmap_canvas = tk.Canvas(self.doa_frame, bg=PANEL_BG, highlightthickness=0)
        self.hotmap_canvas.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))

        self.lbl_doa = tk.Label(self.doa_frame, text="DOA: --- | SECTOR: --- | CONF: --- | LOCK: BUSCANDO",
                                font=("Consolas", 12, "bold"), bg=PANEL_BG, fg=TEXT_MAIN)
        self.lbl_doa.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))

        # BOTTOM STATUS (global)
        bottom = tk.Frame(self.root, bg=PANEL_BG, highlightbackground=FRAME_COLOR, highlightthickness=3)
        bottom.pack(fill="x", padx=18, pady=(0, 12))

        tk.Label(bottom, text="ESTADO OPERACIONAL:",
                 font=("Verdana", 18, "bold"),
                 bg=PANEL_BG, fg=TEXT_MAIN).pack(pady=8)

        self.lbl_status = tk.Label(
            bottom,
            text="CALIBRANDO... MANTÉN SILENCIO",
            font=("Verdana", 20, "bold"),
            bg=ACCENT_AMBER, fg="black"
        )
        self.lbl_status.pack(fill="x", ipady=18, padx=20, pady=(0, 12))

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
    # SERIAL PORTS
    # ------------------------------------------------------------
    def _list_serial_ports(self):
        ports = []
        try:
            for p in serial.tools.list_ports.comports():
                ports.append(p.device)
        except Exception:
            pass
        return ports

    def toggle_doa_connection(self):
        if self.doa_worker is not None:
            # desconectar
            self.disconnect_doa()
        else:
            # conectar
            port = self.cmb_ports.get().strip()
            if not port:
                # refrescar lista
                self.cmb_ports["values"] = self._list_serial_ports()
                self.lbl_status.config(text="Selecciona un COM para DOA", bg=ACCENT_AMBER)
                return
            self.connect_doa(port)

    def connect_doa(self, port):
        self.doa_worker = DOAWorker(self.msg_q, self.stop_event, port, self.doa_params)
        self.doa_worker.start()
        self.btn_doa.config(text="DESCONECTAR")

    def disconnect_doa(self):
        # basta con anular referencia; el stop_event cerrará en salida
        self.doa_worker = None
        self.doa_connected = False
        self.led_doa._led.config(bg=LED_OFF)
        self.btn_doa.config(text="CONECTAR")

    # ------------------------------------------------------------
    # CONFIG / CONTROLES
    # ------------------------------------------------------------
    def open_config(self):
        if self.config_win is not None and self.config_win.winfo_exists():
            self.config_win.lift()
            return

        win = tk.Toplevel(self.root)
        self.config_win = win
        win.title("Configuración rápida (MERGED)")
        win.configure(bg=BG_MAIN)
        win.resizable(False, False)

        pad = {"padx": 10, "pady": 6}

        # Detector
        self.var_margin = tk.StringVar(value=str(self.det_params.margin_low_db))
        self.var_hyst = tk.StringVar(value=str(self.det_params.hyst_db))
        self.var_calib = tk.StringVar(value=str(self.det_params.calib_seconds))
        self.var_use_ratios = tk.BooleanVar(value=self.det_params.use_ratios)
        self.var_r1 = tk.StringVar(value=str(self.det_params.r1_min_db))
        self.var_r2 = tk.StringVar(value=str(self.det_params.r2_min_db))
        self.var_use_snr = tk.BooleanVar(value=self.det_params.use_snr)
        self.var_snr = tk.StringVar(value=str(self.det_params.snr_min_db))
        self.var_peak = tk.BooleanVar(value=self.det_params.use_peak_stability)
        self.var_track = tk.BooleanVar(value=self.det_params.use_band_tracking)

        # DOA
        self.var_contrast = tk.StringVar(value=str(self.doa_params.contrast_min))
        self.var_lobe = tk.StringVar(value=str(self.doa_params.lobe_rel))
        self.var_alpha_min = tk.StringVar(value=str(self.doa_params.alpha_min))
        self.var_alpha_max = tk.StringVar(value=str(self.doa_params.alpha_max))

        r = 0
        tk.Label(win, text="DETECTOR (adaptativo)", bg=BG_MAIN, fg=TEXT_MAIN,
                 font=("Verdana", 11, "bold")).grid(row=r, column=0, columnspan=2, sticky="w", **pad)

        r += 1
        tk.Label(win, text="Margen THR (dB)", bg=BG_MAIN, fg=TEXT_MAIN,
                 font=("Verdana", 10, "bold")).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(win, textvariable=self.var_margin, width=12).grid(row=r, column=1, **pad)

        r += 1
        tk.Label(win, text="Histeresis OFF (dB)", bg=BG_MAIN, fg=TEXT_MAIN,
                 font=("Verdana", 10, "bold")).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(win, textvariable=self.var_hyst, width=12).grid(row=r, column=1, **pad)

        r += 1
        tk.Label(win, text="Calibración (s)", bg=BG_MAIN, fg=TEXT_MAIN,
                 font=("Verdana", 10, "bold")).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(win, textvariable=self.var_calib, width=12).grid(row=r, column=1, **pad)

        r += 1
        tk.Checkbutton(
            win, text="Usar ratios (E36-E13, E36-E69)",
            variable=self.var_use_ratios,
            bg=BG_MAIN, fg=TEXT_MAIN,
            selectcolor=BG_MAIN,
            activebackground=BG_MAIN, activeforeground=TEXT_MAIN
        ).grid(row=r, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 0))

        r += 1
        tk.Label(win, text="R1 min (dB)", bg=BG_MAIN, fg=TEXT_MAIN,
                 font=("Verdana", 10, "bold")).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(win, textvariable=self.var_r1, width=12).grid(row=r, column=1, **pad)

        r += 1
        tk.Label(win, text="R2 min (dB)", bg=BG_MAIN, fg=TEXT_MAIN,
                 font=("Verdana", 10, "bold")).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(win, textvariable=self.var_r2, width=12).grid(row=r, column=1, **pad)

        r += 1
        tk.Checkbutton(
            win, text="Usar SNR gating",
            variable=self.var_use_snr,
            bg=BG_MAIN, fg=TEXT_MAIN,
            selectcolor=BG_MAIN,
            activebackground=BG_MAIN, activeforeground=TEXT_MAIN
        ).grid(row=r, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 0))

        r += 1
        tk.Label(win, text="SNR min (dB)", bg=BG_MAIN, fg=TEXT_MAIN,
                 font=("Verdana", 10, "bold")).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(win, textvariable=self.var_snr, width=12).grid(row=r, column=1, **pad)

        r += 1
        tk.Checkbutton(
            win, text="Peak stability (armónicos/modulación)",
            variable=self.var_peak,
            bg=BG_MAIN, fg=TEXT_MAIN,
            selectcolor=BG_MAIN,
            activebackground=BG_MAIN, activeforeground=TEXT_MAIN
        ).grid(row=r, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 0))

        r += 1
        tk.Checkbutton(
            win, text="Band tracking (banda principal adaptativa)",
            variable=self.var_track,
            bg=BG_MAIN, fg=TEXT_MAIN,
            selectcolor=BG_MAIN,
            activebackground=BG_MAIN, activeforeground=TEXT_MAIN
        ).grid(row=r, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

        r += 1
        tk.Label(win, text="DOA (robustez)", bg=BG_MAIN, fg=TEXT_MAIN,
                 font=("Verdana", 11, "bold")).grid(row=r, column=0, columnspan=2, sticky="w", **pad)

        r += 1
        tk.Label(win, text="Contrast min", bg=BG_MAIN, fg=TEXT_MAIN,
                 font=("Verdana", 10, "bold")).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(win, textvariable=self.var_contrast, width=12).grid(row=r, column=1, **pad)

        r += 1
        tk.Label(win, text="Lobe rel (0..1)", bg=BG_MAIN, fg=TEXT_MAIN,
                 font=("Verdana", 10, "bold")).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(win, textvariable=self.var_lobe, width=12).grid(row=r, column=1, **pad)

        r += 1
        tk.Label(win, text="Alpha min", bg=BG_MAIN, fg=TEXT_MAIN,
                 font=("Verdana", 10, "bold")).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(win, textvariable=self.var_alpha_min, width=12).grid(row=r, column=1, **pad)

        r += 1
        tk.Label(win, text="Alpha max", bg=BG_MAIN, fg=TEXT_MAIN,
                 font=("Verdana", 10, "bold")).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(win, textvariable=self.var_alpha_max, width=12).grid(row=r, column=1, **pad)

        r += 1
        btns = tk.Frame(win, bg=BG_MAIN)
        btns.grid(row=r, column=0, columnspan=2, pady=12)

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
            self.det_params.margin_low_db = float(self.var_margin.get())
            self.det_params.hyst_db = float(self.var_hyst.get())
            self.det_params.calib_seconds = float(self.var_calib.get())

            self.det_params.use_ratios = bool(self.var_use_ratios.get())
            self.det_params.r1_min_db = float(self.var_r1.get())
            self.det_params.r2_min_db = float(self.var_r2.get())

            self.det_params.use_snr = bool(self.var_use_snr.get())
            self.det_params.snr_min_db = float(self.var_snr.get())

            self.det_params.use_peak_stability = bool(self.var_peak.get())
            self.det_params.use_band_tracking = bool(self.var_track.get())

            self.doa_params.contrast_min = float(self.var_contrast.get())
            self.doa_params.lobe_rel = float(self.var_lobe.get())
            self.doa_params.alpha_min = float(self.var_alpha_min.get())
            self.doa_params.alpha_max = float(self.var_alpha_max.get())
        except Exception:
            pass

        self.det_params.calib_seconds = clamp(self.det_params.calib_seconds, 1.0, 10.0)
        self.det_params.hyst_db = clamp(self.det_params.hyst_db, 0.5, 10.0)
        self.det_params.margin_low_db = clamp(self.det_params.margin_low_db, 3.0, 25.0)

        self.doa_params.lobe_rel = clamp(self.doa_params.lobe_rel, 0.50, 0.90)
        self.doa_params.alpha_min = clamp(self.doa_params.alpha_min, 0.01, 0.30)
        self.doa_params.alpha_max = clamp(self.doa_params.alpha_max, self.doa_params.alpha_min + 0.05, 0.60)

        # recalibra para aplicar nuevo margen de forma limpia
        self.recalibrate()

        if self.config_win is not None and self.config_win.winfo_exists():
            self.config_win.destroy()
            self.config_win = None

    def recalibrate(self):
        if self.alarm_active:
            self.rearm_alarm()
        self.lbl_status.config(text="CALIBRANDO... MANTÉN SILENCIO", bg=ACCENT_AMBER)
        self.lbl_calib.config(text="CALIBRANDO...", fg=ACCENT_AMBER)
        self.det_calibrating = True
        self.audio_worker.request_recalibration()

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
            self.lbl_status.config(text="ERROR: alerta.wav no reproducible (usa WAV PCM 16-bit)", bg=ACCENT_AMBER)

    # ------------------------------------------------------------
    # DIBUJO: Spectrum
    # ------------------------------------------------------------
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

        # marcar bandas (fijas) y banda principal (tracking)
        def fx(freq):
            return x0 + (freq - fmin) / (fmax - fmin) * (x1 - x0)

        # 1-3, 3-6, 6-9 (pero aquí dibujamos hasta 8k)
        for lo, hi, label in [(B1_LOW, B1_HIGH, "1-3k"), (B2_LOW, B2_HIGH, "3-6k")]:
            bx0 = fx(lo)
            bx1 = fx(min(hi, 8000.0))
            c.create_rectangle(bx0, y0, bx1, y1, outline=ACCENT_CYAN, width=1)
            c.create_text((bx0 + bx1) / 2, y0 + 8, text=label, fill=ACCENT_CYAN, font=("Verdana", 9, "bold"))

        # banda tracking
        lo = self.band_low
        hi = min(self.band_high, 8000.0)
        bx0 = fx(lo)
        bx1 = fx(hi)
        c.create_rectangle(bx0, y0, bx1, y1, outline=ACCENT_AMBER, width=2)
        c.create_text((bx0 + bx1) / 2, y0 + 22, text="MAIN", fill=ACCENT_AMBER, font=("Verdana", 9, "bold"))

    # ------------------------------------------------------------
    # DIBUJO: Energía que LLENA el panel (barra + escalas)
    # ------------------------------------------------------------
    def _draw_energy_panel(self):
        c = self.energy_canvas
        c.delete("all")

        w = max(c.winfo_width(), 340)
        h = max(c.winfo_height(), 300)

        m = 8
        x0, y0, x1, y1 = m, m, w - m, h - m
        c.create_rectangle(x0, y0, x1, y1, outline=GRID_DIM)

        # escala auto alrededor de thr_low si existe; si no, fallback
        if self.thr_low is not None:
            vmin = self.thr_low - 25.0
            vmax = self.thr_low + 35.0
        else:
            vmin, vmax = -60.0, 60.0

        if self.avg_db_main is not None and self.avg_db_main < vmin + 5:
            vmin = self.avg_db_main - 12.0
            vmax = vmin + 60.0

        if vmax <= vmin:
            vmax = vmin + 10

        def y_from_val(val):
            val = clamp(float(val), vmin, vmax)
            return y1 - (val - vmin) / (vmax - vmin) * (y1 - y0)

        # grid
        for i in range(1, 6):
            y = y0 + i * (y1 - y0) / 6
            c.create_line(x0, y, x1, y, fill=GRID_DIM)

        # Fill ocupa TODO el ancho; altura según energía
        if self.avg_db_main is not None:
            y_fill = y_from_val(self.avg_db_main)
            fill_color = ACCENT_DANGER if self.det_detected else (ACCENT_AMBER if self.det_raw else ACCENT_SAFE)
            c.create_rectangle(x0, y_fill, x1, y1, fill=fill_color, outline="")

        # THR
        if self.thr_low is not None:
            y_thr = y_from_val(self.thr_low)
            c.create_line(x0, y_thr, x1, y_thr, fill=ACCENT_AMBER, width=3)
            c.create_text(x1 - 6, y_thr - 12, text=f"THR {self.thr_low:.2f} dB",
                          fill=ACCENT_AMBER, font=("Verdana", 10, "bold"), anchor="e")

        # etiqueta min/max
        c.create_text(x0 + 6, y0 + 6, text=f"{vmax:.0f} dB", fill=TEXT_MAIN, font=("Verdana", 9), anchor="nw")
        c.create_text(x0 + 6, y1 - 6, text=f"{vmin:.0f} dB", fill=TEXT_MAIN, font=("Verdana", 9), anchor="sw")

        # Conf bar
        # barra vertical en el lateral derecho (llenando también)
        conf = clamp(self.conf_det, 0.0, 1.0)
        bx0 = x1 - 18
        bx1 = x1 - 6
        c.create_rectangle(bx0, y0, bx1, y1, outline=GRID_DIM)
        by = y1 - conf * (y1 - y0)
        c.create_rectangle(bx0, by, bx1, y1, fill=ACCENT_CYAN, outline="")
        c.create_text(bx0 - 6, y0 + 2, text="CONF", fill=ACCENT_CYAN, font=("Verdana", 9, "bold"), anchor="ne")

    # ------------------------------------------------------------
    # DOA: hotmap draw
    # ------------------------------------------------------------
    def _draw_hotmap(self, hm_uint8):
        c = self.hotmap_canvas
        c.delete("all")
        w = max(c.winfo_width(), 320)
        h = max(c.winfo_height(), 240)

        # fondo
        c.create_rectangle(0, 0, w, h, fill=HOTMAP_BG, outline="")

        if hm_uint8 is None:
            c.create_text(w/2, h/2, text="HOTMAP: ---", fill=TEXT_MAIN, font=FONT_MONO)
            return

        hm = hm_uint8.astype(np.float32)
        vmax = float(np.max(hm))
        if vmax <= 0:
            vmax = 1.0

        # celdas
        grid = 16
        m = 10
        x0, y0 = m, m
        x1, y1 = w - m, h - m

        cw = (x1 - x0) / grid
        ch = (y1 - y0) / grid

        # resaltar pico
        pi, pj = np.unravel_index(int(np.argmax(hm)), hm.shape)

        for i in range(grid):
            for j in range(grid):
                v = hm[i, j] / vmax
                # color simple: variación de intensidad sobre HOTMAP_LINE
                # (Tkinter no soporta alpha; hacemos gradiente aproximado)
                intensity = int(clamp(40 + 215 * v, 40, 255))
                col = f"#{0:02x}{intensity:02x}{int(120 + 80*v):02x}"  # verde-cian aproximado
                xA = x0 + j * cw
                yA = y0 + i * ch
                xB = xA + cw
                yB = yA + ch
                c.create_rectangle(xA, yA, xB, yB, fill=col, outline=GRID_DIM, width=1)

        # marca del pico
        xA = x0 + pj * cw
        yA = y0 + pi * ch
        c.create_rectangle(xA, yA, xA + cw, yA + ch, outline=ACCENT_AMBER, width=3)

    # ------------------------------------------------------------
    # DOA: compass draw
    # ------------------------------------------------------------
    def _draw_compass(self):
        c = self.compass_canvas
        c.delete("all")
        w = max(c.winfo_width(), 320)
        h = max(c.winfo_height(), 240)

        cx = w * 0.5
        cy = h * 0.52
        R = min(w, h) * 0.40

        # círculos
        for k in [1.0, 0.66, 0.33]:
            r = R * k
            c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=GRID_DIM, width=1)

        # ejes
        c.create_line(cx - R, cy, cx + R, cy, fill=GRID_DIM, width=1)
        c.create_line(cx, cy - R, cx, cy + R, fill=GRID_DIM, width=1)

        # N/E/S/W
        c.create_text(cx, cy - R - 12, text="PROA", fill=TEXT_MAIN, font=("Verdana", 10, "bold"))
        c.create_text(cx + R + 18, cy, text="ESTRIBOR", fill=TEXT_MAIN, font=("Verdana", 10, "bold"), angle=90)
        c.create_text(cx, cy + R + 12, text="POPA", fill=TEXT_MAIN, font=("Verdana", 10, "bold"))
        c.create_text(cx - R - 18, cy, text="BABOR", fill=TEXT_MAIN, font=("Verdana", 10, "bold"), angle=90)

        # estela
        if len(self._doa_trail) >= 2:
            for k, th in enumerate(list(self._doa_trail)[-60:]):
                rr = R * (0.30 + 0.70 * (k / max(1, min(60, len(self._doa_trail)) - 1)))
                x = cx + rr * math.sin(th)
                y = cy - rr * math.cos(th)
                c.create_oval(x-2, y-2, x+2, y+2, fill=ACCENT_CYAN, outline="")

        # sector y flecha
        if self.doa_theta is None:
            c.create_text(cx, cy, text="STANDBY", fill=ACCENT_AMBER, font=("Verdana", 16, "bold"))
            return

        th = float(self.doa_theta)
        # sector
        if self.doa_sector_width_deg is not None:
            wd = float(self.doa_sector_width_deg)
            wd = clamp(wd, 5.0, 180.0)
            start_deg = 90 - math.degrees(th) - wd/2
            extent = wd
            c.create_arc(cx - R, cy - R, cx + R, cy + R,
                         start=start_deg, extent=extent,
                         style="arc", outline=ACCENT_AMBER, width=5)

        # flecha
        x_end = cx + R * 0.95 * math.sin(th)
        y_end = cy - R * 0.95 * math.cos(th)
        c.create_line(cx, cy, x_end, y_end, fill=ACCENT_DANGER if self.det_detected else ACCENT_AMBER, width=4)

        # punto destino
        c.create_oval(x_end - 6, y_end - 6, x_end + 6, y_end + 6,
                     fill=ACCENT_DANGER if self.det_detected else ACCENT_AMBER, outline="")

        # lock indicator
        c.create_text(cx, cy + R + 14, text=f"LOCK: {self.doa_lock_state}",
                      fill=ACCENT_SAFE if self.doa_lock_state == "FIJADO" else ACCENT_AMBER,
                      font=("Verdana", 12, "bold"))

    # ------------------------------------------------------------
    # PRIMER DIBUJO
    # ------------------------------------------------------------
    def _first_draw(self):
        self._draw_spectrum_grid()
        self._draw_energy_panel()
        self._draw_compass()
        self._draw_hotmap(None)

    # ------------------------------------------------------------
    # WATCHDOGS
    # ------------------------------------------------------------
    def _watchdog_loop(self):
        # Micro watchdog
        if self.audio_worker.is_audio_dead():
            self.led_mic._led.config(bg=ACCENT_DANGER)
            self.lbl_status.config(text="MIC watchdog: sin audio. Reintentando...", bg=ACCENT_AMBER)
        self.after_ids.append(self.root.after(500, self._watchdog_loop))

    # ------------------------------------------------------------
    # DOA estimation from hotmap
    # ------------------------------------------------------------
    def _estimate_doa(self, hm_uint8):
        p = self.doa_params
        hm = hm_uint8.astype(np.float32)

        vmax = float(np.max(hm))
        vmean = float(np.mean(hm))
        contrast = vmax - vmean

        # rechazo por contraste
        if contrast < p.contrast_min:
            return None

        # normalización espacial
        med = float(np.median(hm))
        hn = hm - med
        hn[hn < 0] = 0
        # soft-threshold implícito vía gamma
        hn = np.power(hn, p.gamma)

        vmax2 = float(np.max(hn))
        if vmax2 <= 0:
            return None

        # lóbulo
        thr = p.lobe_rel * vmax2
        mask = hn >= thr
        count = int(np.sum(mask))
        if count < p.lobe_min_count:
            return None

        # bimodal: busca segundo pico separado
        flat = hn.flatten()
        idx1 = int(np.argmax(flat))
        v1 = float(flat[idx1])
        flat[idx1] = -1
        idx2 = int(np.argmax(flat))
        v2 = float(flat[idx2])
        flat[idx1] = v1

        i1, j1 = divmod(idx1, 16)
        i2, j2 = divmod(idx2, 16)
        sep = abs(i1 - i2) + abs(j1 - j2)
        ambiguous = False
        if sep >= p.peak_min_separation_cells and v2 >= p.bimodal_ratio * v1:
            ambiguous = True

        # ángulos y pesos del lóbulo
        angles = []
        weights = []
        sum_w = 0.0
        sum_vx = 0.0
        sum_vy = 0.0
        for i in range(16):
            for j in range(16):
                if not mask[i, j]:
                    continue
                w = float(hn[i, j])
                if w <= 0:
                    continue
                ang = _cell_angle_rad(i, j)
                angles.append(ang)
                weights.append(w)
                sum_w += w
                sum_vx += w * math.cos(ang)
                sum_vy += w * math.sin(ang)

        if sum_w <= 0:
            return None

        # media circular ponderada
        vx = sum_vx / sum_w
        vy = sum_vy / sum_w
        R = math.sqrt(vx * vx + vy * vy)  # coherencia 0..1
        theta = math.atan2(vy, vx)
        if theta < 0:
            theta += 2 * math.pi

        # sector por percentil de peso
        arc = _minimal_arc_containing_weight(angles, weights, frac=p.sector_weight_percent)
        if arc is None:
            sector_width = math.radians(120.0)
        else:
            _, _, width = arc
            sector_width = width

        # confianza (0..1): combina coherencia + contraste normalizado + radialidad
        # radialidad: si el lóbulo cae lejos del centro tiende a ser más direccional
        # aproximación: distancia promedio del centro de las celdas del lóbulo
        cx, cy = 7.5, 7.5
        rr = []
        for i in range(16):
            for j in range(16):
                if mask[i, j]:
                    rr.append(math.hypot(j - cx, i - cy))
        rmean = float(np.mean(rr)) if rr else 0.0
        rnorm = clamp(rmean / 7.5, 0.0, 1.0)

        c_norm = clamp((contrast - p.contrast_min) / (p.contrast_min * 2.0), 0.0, 1.0)
        conf = float(0.50 * clamp(R, 0.0, 1.0) + 0.30 * c_norm + 0.20 * rnorm)

        # si ambiguo, penaliza confianza y ensancha sector
        if ambiguous:
            conf *= 0.75
            sector_width = max(sector_width, math.radians(70.0))

        return theta, math.degrees(sector_width), conf, ambiguous, vmax, vmean, count, R

    def _update_lock_state(self):
        """
        BUSCANDO / INESTABLE / FIJADO
        """
        now = time.time()

        if not self.det_detected or self.doa_theta is None:
            self.doa_lock_state = "BUSCANDO"
            self._doa_good_streak_start = None
            return

        good = (self.doa_conf >= 0.65) and (self.doa_sector_width_deg is not None and self.doa_sector_width_deg <= 25.0)
        if good and not self.doa_ambiguous:
            if self._doa_good_streak_start is None:
                self._doa_good_streak_start = now
            if (now - self._doa_good_streak_start) >= 2.0:
                self.doa_lock_state = "FIJADO"
            else:
                self.doa_lock_state = "INESTABLE"
        else:
            self.doa_lock_state = "INESTABLE"
            self._doa_good_streak_start = None

    # ------------------------------------------------------------
    # LOOP: queue
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
                    _, bm, bs, tl = msg
                    self.baseline_mean = bm
                    self.baseline_std = bs
                    self.thr_low = tl
                    self.det_calibrating = False
                    self.lbl_calib.config(
                        text=f"BASE {bm:.2f} dB | THR {tl:.2f} dB | AUTO {int(self.det_params.auto_recal_every_s)}s",
                        fg=ACCENT_SAFE
                    )
                    self.lbl_status.config(text="LISTO. VIGILANCIA ACTIVA.", bg=ACCENT_SAFE)

                elif kind == "SPEC":
                    _, f, db = msg
                    self.led_dsp._led.config(bg=ACCENT_SAFE)
                    self._draw_spectrum(f, db)

                elif kind == "DET":
                    (
                        _,
                        det_final, det_raw,
                        E13, E36, E69,
                        R1, R2, SNR,
                        avg_db_main,
                        thr_low,
                        bm, bs,
                        band_low, band_high,
                        calibrating,
                        conf_det
                    ) = msg

                    self.det_detected = bool(det_final)
                    self.det_raw = bool(det_raw)

                    self.E13, self.E36, self.E69 = float(E13), float(E36), float(E69)
                    self.R1, self.R2, self.SNR = float(R1), float(R2), float(SNR)

                    self.avg_db_main = float(avg_db_main)
                    self.thr_low = thr_low if thr_low is None else float(thr_low)
                    self.baseline_mean = bm if bm is None else float(bm)
                    self.baseline_std = bs if bs is None else float(bs)

                    self.band_low = float(band_low)
                    self.band_high = float(band_high)

                    self.det_calibrating = bool(calibrating)
                    self.conf_det = float(conf_det)

                    # UI text + panel full-fill
                    thr_txt = "---" if self.thr_low is None else f"{self.thr_low:6.2f}"
                    self.lbl_energy.config(
                        text=f"E_MAIN: {self.avg_db_main:6.2f} dB | THR: {thr_txt} dB | MAIN[{self.band_low:0.0f}-{self.band_high:0.0f}] Hz"
                    )

                    self.lbl_features.config(
                        text=(
                            f"E13 {self.E13:6.2f} | E36 {self.E36:6.2f} | E69 {self.E69:6.2f} | "
                            f"R1 {self.R1:5.1f} | R2 {self.R2:5.1f} | SNR {self.SNR:5.1f} | "
                            f"CONF {self.conf_det:0.2f}"
                        )
                    )

                    self._draw_energy_panel()

                    # Estado global
                    if self.det_calibrating and self.det_params.adaptive:
                        self.lbl_calib.config(text="CALIBRANDO...", fg=ACCENT_AMBER)
                        self.lbl_status.config(text="CALIBRANDO... MANTÉN SILENCIO", bg=ACCENT_AMBER)
                    else:
                        if self.det_detected:
                            self.lbl_status.config(text="UAS DETECTADO (ROBUSTO: multi-banda + persistencia)", bg=ACCENT_AMBER)
                            if not self.alarm_active:
                                self.trigger_alarm()
                        else:
                            if not self.alarm_active:
                                self.lbl_status.config(text="UAS NO DETECTADO / FUERA DE PELIGRO", bg=ACCENT_SAFE)

                elif kind == "DOA_OK":
                    self.doa_connected = True
                    self.led_doa._led.config(bg=ACCENT_SAFE)
                    self.lbl_status.config(text=msg[1], bg=ACCENT_SAFE)

                elif kind == "DOA_ERR":
                    self.doa_connected = False
                    self.led_doa._led.config(bg=ACCENT_DANGER)
                    self.lbl_status.config(text=msg[1], bg=ACCENT_AMBER)

                elif kind == "HOTMAP":
                    _, hm, ts = msg
                    self.hotmap = hm
                    self.last_hotmap_ts = float(ts)

                    # gating por detección
                    if not self.det_detected or self.det_calibrating:
                        # standby DOA: no actualizar, mantiene último fiable
                        self.doa_theta = self._doa_last_good_theta
                        self.doa_sector_width_deg = 120.0 if self._doa_last_good_theta is not None else None
                        self.doa_conf = 0.0
                        self.doa_ambiguous = False
                    else:
                        est = self._estimate_doa(hm)
                        if est is None:
                            # frame malo -> no actualizar
                            # mantener último valor
                            self.doa_theta = self._doa_last_good_theta
                            self.doa_sector_width_deg = 120.0 if self._doa_last_good_theta is not None else None
                            self.doa_conf = 0.0
                            self.doa_ambiguous = False
                        else:
                            theta, sector_deg, conf, amb, vmax, vmean, count, R = est

                            # smoothing adaptativo por confianza
                            alpha = self.doa_params.alpha_min + conf * (self.doa_params.alpha_max - self.doa_params.alpha_min)
                            alpha = clamp(alpha, self.doa_params.alpha_min, self.doa_params.alpha_max)

                            v = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
                            self._doa_vf = (1.0 - alpha) * self._doa_vf + alpha * v
                            nrm = float(np.linalg.norm(self._doa_vf))
                            if nrm > 1e-9:
                                self._doa_vf /= nrm
                            theta_f = math.atan2(float(self._doa_vf[1]), float(self._doa_vf[0]))
                            if theta_f < 0:
                                theta_f += 2 * math.pi

                            self.doa_theta = theta_f
                            self.doa_sector_width_deg = float(sector_deg)
                            self.doa_conf = float(conf)
                            self.doa_ambiguous = bool(amb)

                            # guardar último bueno si conf aceptable
                            if conf >= 0.35:
                                self._doa_last_good_theta = theta_f
                                self._doa_trail.append(theta_f)

                    # dibujar DOA panel
                    self._draw_hotmap(self.hotmap)
                    self._draw_compass()
                    self._update_lock_state()

                    # texto DOA
                    if self.doa_theta is None:
                        doa_txt = "DOA: ---"
                        sec_txt = "SECTOR: ---"
                        conf_txt = "CONF: ---"
                        nav_txt = "SECTOR NAVAL: ---"
                    else:
                        deg = (math.degrees(self.doa_theta) % 360.0)
                        nav_txt = f"SECTOR NAVAL: {naval_sector_label(deg)}"
                        doa_txt = f"DOA: {deg:6.1f}°"
                        sec_txt = "SECTOR: ---" if self.doa_sector_width_deg is None else f"SECTOR: ±{self.doa_sector_width_deg/2:4.1f}°"
                        conf_txt = f"CONF: {self.doa_conf:0.2f}" + (" (AMBIGUO)" if self.doa_ambiguous else "")

                    self.lbl_doa.config(text=f"{doa_txt} | {sec_txt} | {conf_txt} | {nav_txt} | LOCK: {self.doa_lock_state}")

        except queue.Empty:
            pass

        self.after_ids.append(self.root.after(60, self._poll_queue))

    # ------------------------------------------------------------
    # LOOP: blink
    # ------------------------------------------------------------
    def _blink_loop(self):
        if self.alarm_active and self.blink_enabled:
            self.blink_on = not self.blink_on
            color = ACCENT_DANGER if self.blink_on else ACCENT_DANGER_DARK
            self.lbl_status.config(bg=color)
            self.led_alert._led.config(bg=color)
        self.after_ids.append(self.root.after(220, self._blink_loop))

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
    app = MergedUI(root)
    root.mainloop()
