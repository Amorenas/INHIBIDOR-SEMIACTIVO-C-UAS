import time
import math
import json
import threading
import queue
from collections import deque

import tkinter as tk

try:
    import serial
    import serial.tools.list_ports
except Exception:
    serial = None


# ============================================================
# CONFIG GENERAL
# ============================================================
APP_TITLE = "CONSOLA DOA - MA-USB8 (SIPEED 6+1)"
BG_MAIN = "#050B12"
FRAME_COLOR = "#1C252F"
PANEL_BG = "#243B4E"
TEXT_MAIN = "#E2F4FF"
STATUS_STRIP_BG = "#0B1118"
GRID_DIM = "#1E2E3A"
ACCENT_SAFE = "#00D68F"
ACCENT_AMBER = "#F5C542"
ACCENT_DANGER = "#FF3B3B"
LED_OFF = "#555555"
RADAR_LINE = "#00FF7F"  # flecha DOA

HOTMAP_HEADER = bytes([0xFF] * 16)
HOTMAP_FRAME_LEN = 16 + 256  # 272 bytes total (16 header + 256 payload)

DEFAULT_BAUD = 115200  # CDC-ACM suele ignorarlo, pero pyserial lo pide

# Suavizado temporal adicional (por seguridad). Media circular sobre últimos N azimuts.
AZ_SMOOTH_N = 5

# ============================================================
# DOA "SERIO" DESDE HOTMAP
# ============================================================
# Selección de lóbulo: se usan celdas con v >= LOBE_FRAC * vmax
LOBE_FRAC = 0.70

# Ignorar celdas demasiado cerca del centro (típicamente inestables)
R_MIN = 1.0  # en celdas (radio mínimo desde el centro 7.5,7.5)

# Cálculo sector a partir de desviación estándar circular:
# half_angle_deg = clamp(STD_MULT * sigma_deg, SECTOR_MIN, SECTOR_MAX)
STD_MULT = 2.2
SECTOR_MIN_HALF_DEG = 8.0
SECTOR_MAX_HALF_DEG = 70.0

# Si quieres forzar sector fijo, pon un número (por ej 20.0). Si None => estimación seria.
SECTOR_FIXED_HALF_ANGLE_DEG = None

# Umbral de "datos suficientes" para fiarte del lóbulo
MIN_LOBE_POINTS = 6
MIN_LOBE_WEIGHT = 50.0

# ============================================================
# UDP output (para merge posterior con Detector_Acustico)
# ============================================================
UDP_ENABLED_DEFAULT = True
UDP_IP_DEFAULT = "127.0.0.1"
UDP_PORT_DEFAULT = 5055

# Reintentos de conexión
RECONNECT_SEC = 2.0


def clamp(v, a, b):
    return max(a, min(b, v))


def deg_norm(a):
    return (a + 360.0) % 360.0


def az_to_canvas_xy(cx, cy, R, az_deg):
    """0°=N (arriba), horario. Convierte a coords canvas."""
    a = math.radians(az_deg)
    x = cx + R * math.sin(a)
    y = cy - R * math.cos(a)
    return x, y


def circular_mean_deg(angles_deg, weights=None):
    """Media circular (grados)."""
    if not angles_deg:
        return None
    if weights is None:
        weights = [1.0] * len(angles_deg)
    s = 0.0
    c = 0.0
    wsum = 0.0
    for a, w in zip(angles_deg, weights):
        r = math.radians(a)
        s += w * math.sin(r)
        c += w * math.cos(r)
        wsum += w
    if wsum <= 0:
        return None
    if abs(s) < 1e-12 and abs(c) < 1e-12:
        return deg_norm(angles_deg[-1])
    return deg_norm(math.degrees(math.atan2(s, c)))


def circular_resultant_and_std_rad(angles_deg, weights):
    """
    Devuelve:
      R (0..1) y sigma (radianes) usando aproximación sigma = sqrt(-2 ln R)
    """
    s = 0.0
    c = 0.0
    wsum = 0.0
    for a, w in zip(angles_deg, weights):
        r = math.radians(a)
        s += w * math.sin(r)
        c += w * math.cos(r)
        wsum += w
    if wsum <= 0:
        return 0.0, float("inf")

    R = math.hypot(s, c) / wsum
    R = clamp(R, 1e-6, 1.0)
    sigma = math.sqrt(max(0.0, -2.0 * math.log(R)))
    return float(R), float(sigma)


def find_candidate_com_ports():
    """
    Intenta localizar el MA-USB8 por heurística.
    Devuelve lista de (device, description).
    """
    out = []
    if serial is None:
        return out
    try:
        for p in serial.tools.list_ports.comports():
            desc = (p.description or "").lower()
            hwid = (p.hwid or "").lower()
            name = (p.name or "").lower()
            if ("usb" in desc or "usb" in hwid or "usb" in name) and (
                "sipeed" in desc or "sipeed" in hwid or
                "cdc" in desc or "acm" in desc or
                "serial" in desc or
                "bouffalo" in desc or "bl" in desc or
                "ma" in desc or "micarray" in desc
            ):
                out.append((p.device, p.description))
        if not out:
            for p in serial.tools.list_ports.comports():
                out.append((p.device, p.description))
    except Exception:
        pass
    return out


def hotmap_stats(hm256):
    if hm256 is None or len(hm256) != 256:
        return 0, 0.0, (0, 0)
    vmax = -1
    imax = 0
    s = 0
    for i, v in enumerate(hm256):
        if v > vmax:
            vmax = v
            imax = i
        s += v
    vmean = s / 256.0
    y = imax // 16
    x = imax % 16
    return int(vmax), float(vmean), (int(x), int(y))


def hotmap_to_doa_sector_conf(hm256):
    """
    DOA serio:
      - define lóbulo: v >= LOBE_FRAC*vmax y r>=R_MIN
      - calcula az como media circular ponderada por (v - vmean)+
      - sector = +/- STD_MULT*sigma_deg (sigma por resultante circular)
      - conf combina prominencia y concentración y radialidad

    Devuelve:
      az_deg (o None), sector_half_deg, conf(0..1), vmax, vmean, peak_xy, lobe_count, R
    """
    if hm256 is None or len(hm256) != 256:
        return None, None, 0.0, 0, 0.0, (0, 0), 0, 0.0

    vmax, vmean, peak_xy = hotmap_stats(hm256)
    if vmax <= 0:
        return None, None, 0.0, vmax, vmean, peak_xy, 0, 0.0

    cx = 7.5
    cy = 7.5

    thr = LOBE_FRAC * vmax

    angles = []
    weights = []
    rads = []
    lobe_count = 0

    for idx, v in enumerate(hm256):
        if v < thr:
            continue
        y = idx // 16
        x = idx % 16
        dx = x - cx
        dy = y - cy
        r = math.hypot(dx, dy)
        if r < R_MIN:
            continue

        # ángulo: 0=N, horario: atan2(dx, -dy)
        az = deg_norm(math.degrees(math.atan2(dx, -dy)))

        # peso: parte útil sobre el ruido medio
        w = max(0.0, float(v) - float(vmean))
        if w <= 0.0:
            continue

        angles.append(az)
        weights.append(w)
        rads.append(r)
        lobe_count += 1

    sumw = sum(weights) if weights else 0.0
    if lobe_count < MIN_LOBE_POINTS or sumw < MIN_LOBE_WEIGHT:
        # fallback: pico (pero con sector grande)
        px, py = peak_xy
        dx = px - cx
        dy = py - cy
        r = math.hypot(dx, dy)
        if r < R_MIN:
            return None, None, 0.0, vmax, vmean, peak_xy, lobe_count, 0.0
        az = deg_norm(math.degrees(math.atan2(dx, -dy)))
        sector_half = SECTOR_MAX_HALF_DEG
        # confianza baja: pico vs media y radialidad
        prom = clamp((vmax - vmean) / 255.0, 0.0, 1.0)
        radial = clamp(r / 10.6, 0.0, 1.0)
        conf = clamp(0.25 * prom * radial, 0.0, 1.0)
        return az, sector_half, conf, vmax, vmean, peak_xy, lobe_count, 0.0

    # DOA como media circular ponderada
    az_deg = circular_mean_deg(angles, weights)

    # Concentración y sigma
    R, sigma_rad = circular_resultant_and_std_rad(angles, weights)
    sigma_deg = math.degrees(sigma_rad)

    if SECTOR_FIXED_HALF_ANGLE_DEG is not None:
        sector_half = float(SECTOR_FIXED_HALF_ANGLE_DEG)
    else:
        sector_half = clamp(STD_MULT * sigma_deg, SECTOR_MIN_HALF_DEG, SECTOR_MAX_HALF_DEG)

    # radialidad media del lóbulo (si está cerca del centro => mala)
    r_mean = sum(rads) / max(1, len(rads))
    radial = clamp(r_mean / 10.6, 0.0, 1.0)

    # prominencia (pico vs media)
    prom = clamp((vmax - vmean) / 255.0, 0.0, 1.0)

    # confianza final: prom * concentración * radialidad
    # R cerca de 1 => lóbulo muy concentrado
    conf = clamp(0.20 * prom + 0.55 * prom * R + 0.25 * prom * radial, 0.0, 1.0)

    return az_deg, sector_half, conf, vmax, vmean, peak_xy, lobe_count, R


class SerialDOAWorker(threading.Thread):
    """
    Lee CDC-ACM (COM) del MA-USB8 y produce:
      ("STATUS", level, text)
      ("HOTMAP", hm256, az_deg, sector_half, conf, vmax, vmean, peak_xy, lobe_count, R)
    """
    def __init__(self, out_q: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.q = out_q
        self.stop_event = stop_event

        self.lock = threading.Lock()
        self.com_port = None
        self.baud = DEFAULT_BAUD

        self._ser = None
        self._buf = bytearray()

        self.az_hist = deque(maxlen=AZ_SMOOTH_N)
        self.sector_hist = deque(maxlen=AZ_SMOOTH_N)

        self._pending_cmd = None

    def push(self, msg):
        try:
            self.q.put_nowait(msg)
        except queue.Full:
            pass

    def set_port(self, com_port: str, baud: int = DEFAULT_BAUD):
        with self.lock:
            self.com_port = com_port
            self.baud = int(baud)

    def send_command(self, cmd: str):
        with self.lock:
            self._pending_cmd = cmd

    def _close_serial(self):
        try:
            if self._ser is not None:
                self._ser.close()
        except Exception:
            pass
        self._ser = None

    def _try_open(self):
        if serial is None:
            self.push(("STATUS", "ERR", "Falta pyserial: instala con 'py -m pip install pyserial'"))
            return False

        with self.lock:
            port = self.com_port
            baud = self.baud

        if not port:
            self.push(("STATUS", "WARN", "Selecciona un COM (o usa auto-detección)."))
            return False

        try:
            self._ser = serial.Serial(port, baudrate=baud, timeout=0.15)
            self._buf = bytearray()
            self.az_hist.clear()
            self.sector_hist.clear()
            self.push(("STATUS", "OK", f"Conectado a {port}"))
            return True
        except Exception as e:
            self._ser = None
            self.push(("STATUS", "ERR", f"No puedo abrir {port}: {e}"))
            return False

    def _read_bytes(self, n=256):
        if self._ser is None:
            return b""
        try:
            return self._ser.read(n)
        except Exception:
            return b""

    def _write_bytes(self, b: bytes):
        if self._ser is None:
            return
        try:
            self._ser.write(b)
        except Exception:
            pass

    def _extract_hotmap(self):
        chunk = self._read_bytes(512)
        if chunk:
            self._buf.extend(chunk)

        if len(self._buf) > 8192:
            self._buf = self._buf[-4096:]

        idx = self._buf.find(HOTMAP_HEADER)
        if idx < 0:
            return None

        end = idx + HOTMAP_FRAME_LEN
        if len(self._buf) < end:
            return None

        payload = self._buf[idx + 16: idx + 16 + 256]
        self._buf = self._buf[end:]

        if len(payload) != 256:
            return None

        return list(payload)

    def run(self):
        last_fail_ts = 0.0

        while not self.stop_event.is_set():
            if self._ser is None:
                now = time.time()
                if now - last_fail_ts < RECONNECT_SEC:
                    time.sleep(0.05)
                    continue
                ok = self._try_open()
                if not ok:
                    last_fail_ts = time.time()
                    time.sleep(0.10)
                    continue

            with self.lock:
                cmd = self._pending_cmd
                self._pending_cmd = None
            if cmd:
                self._write_bytes(cmd.encode("ascii", errors="ignore"))

            hm = self._extract_hotmap()
            if hm is None:
                continue

            az, sector_half, conf, vmax, vmean, peak_xy, lobe_count, R = hotmap_to_doa_sector_conf(hm)

            # suavizado temporal
            if az is not None:
                self.az_hist.append(az)
                az_s = circular_mean_deg(list(self.az_hist))
            else:
                az_s = None

            if sector_half is not None:
                self.sector_hist.append(sector_half)
                sector_s = sum(self.sector_hist) / max(1, len(self.sector_hist))
            else:
                sector_s = None

            self.push(("HOTMAP", hm, az_s, sector_s, conf, vmax, vmean, peak_xy, lobe_count, R))

        self._close_serial()


class DOAConsoleUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.configure(bg=BG_MAIN)
        self.root.geometry("1400x720")

        self.stop_event = threading.Event()
        self.msg_q = queue.Queue(maxsize=200)

        # runtime
        self.current_hotmap = None
        self.az_deg = None
        self.sector_half = None
        self.conf = 0.0
        self.vmax = 0
        self.vmean = 0.0
        self.peak_xy = (0, 0)
        self.lobe_count = 0
        self.R = 0.0

        # sector toggle
        self.sector_enabled = True

        # UDP
        self.udp_enabled = UDP_ENABLED_DEFAULT
        self.udp_ip = UDP_IP_DEFAULT
        self.udp_port = UDP_PORT_DEFAULT
        self._udp_sock = None

        # serial
        self.com_port = None
        self.baud = DEFAULT_BAUD

        # worker
        self.worker = SerialDOAWorker(self.msg_q, self.stop_event)
        self.worker.start()

        self._build_ui()

        self.after_ids = []
        self.after_ids.append(self.root.after(200, self._first_draw))
        self.after_ids.append(self.root.after(60, self._poll_queue))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._auto_detect_and_connect()

    # ---------------- UI BUILD ----------------
    def _build_ui(self):
        title_frame = tk.Frame(self.root, bg=BG_MAIN)
        title_frame.pack(pady=(5, 0), fill="x")

        tk.Label(
            title_frame, text="DOA UAS (MA-USB8 + SIPEED 6+1)",
            font=("Verdana", 26, "bold"),
            fg=TEXT_MAIN, bg=BG_MAIN
        ).pack(side="left", padx=10)

        tk.Button(
            title_frame, text="SECTOR ON/OFF",
            font=("Verdana", 10, "bold"),
            command=self.toggle_sector,
            bg="#29323C", fg="white",
            activebackground="#3C4A58", activeforeground="white",
            relief="raised", bd=2
        ).pack(side="right", padx=(0, 10))

        tk.Button(
            title_frame, text="CONECTAR",
            font=("Verdana", 10, "bold"),
            command=self.open_connection_dialog,
            bg="#29323C", fg="white",
            activebackground="#3C4A58", activeforeground="white",
            relief="raised", bd=2
        ).pack(side="right", padx=(0, 10))

        status_strip = tk.Frame(self.root, bg=STATUS_STRIP_BG)
        status_strip.pack(fill="x", padx=10, pady=(5, 10))

        self.lbl_port = tk.Label(
            status_strip, text="COM: ---",
            font=("Consolas", 11, "bold"),
            bg=STATUS_STRIP_BG, fg=TEXT_MAIN
        )
        self.lbl_port.grid(row=0, column=0, padx=10, sticky="w")

        self.lbl_status = tk.Label(
            status_strip, text="ESTADO: buscando MA-USB8...",
            font=("Consolas", 11, "bold"),
            bg=STATUS_STRIP_BG, fg=ACCENT_AMBER
        )
        self.lbl_status.grid(row=0, column=1, padx=10, sticky="w")

        self.led_link = self._create_led(status_strip, "ENLACE", LED_OFF)
        self.led_link.grid(row=0, column=2, padx=10, sticky="e")

        self.led_doa = self._create_led(status_strip, "DOA", LED_OFF)
        self.led_doa.grid(row=0, column=3, padx=10, sticky="e")

        center = tk.Frame(self.root, bg=BG_MAIN)
        center.pack(fill="both", expand=True, padx=20, pady=10)

        center.columnconfigure(0, weight=1)
        center.columnconfigure(1, weight=1)
        center.columnconfigure(2, weight=1)
        center.rowconfigure(0, weight=1)

        # Left: hotmap
        self.hot_frame = tk.Frame(center, bg=PANEL_BG,
                                  highlightbackground=FRAME_COLOR, highlightthickness=3)
        self.hot_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.hot_frame.rowconfigure(1, weight=1)
        self.hot_frame.columnconfigure(0, weight=1)

        tk.Label(self.hot_frame, text="HOTMAP (16×16) CAMPO ACÚSTICO",
                 font=("Verdana", 15, "bold"),
                 bg=PANEL_BG, fg=TEXT_MAIN).grid(row=0, column=0, sticky="ew", pady=(8, 4))
        self.hot_canvas = tk.Canvas(self.hot_frame, bg=PANEL_BG, highlightthickness=0)
        self.hot_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Center: compass
        self.comp_frame = tk.Frame(center, bg=PANEL_BG,
                                   highlightbackground=FRAME_COLOR, highlightthickness=3)
        self.comp_frame.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self.comp_frame.rowconfigure(1, weight=1)
        self.comp_frame.columnconfigure(0, weight=1)

        tk.Label(self.comp_frame, text="DIRECCIÓN (DOA) + SECTOR",
                 font=("Verdana", 15, "bold"),
                 bg=PANEL_BG, fg=TEXT_MAIN).grid(row=0, column=0, sticky="ew", pady=(8, 4))
        self.comp_canvas = tk.Canvas(self.comp_frame, bg=PANEL_BG, highlightthickness=0)
        self.comp_canvas.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

        self.lbl_az = tk.Label(
            self.comp_frame, text="AZ: --- °",
            font=("Consolas", 18, "bold"),
            bg=PANEL_BG, fg=TEXT_MAIN
        )
        self.lbl_az.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        # Right: info + udp
        self.info_frame = tk.Frame(center, bg=PANEL_BG,
                                   highlightbackground=FRAME_COLOR, highlightthickness=3)
        self.info_frame.grid(row=0, column=2, sticky="nsew", padx=8, pady=8)
        self.info_frame.columnconfigure(0, weight=1)

        tk.Label(self.info_frame, text="TELEMETRÍA / CONTROL",
                 font=("Verdana", 15, "bold"),
                 bg=PANEL_BG, fg=TEXT_MAIN).grid(row=0, column=0, sticky="ew", pady=(8, 4))

        self.lbl_conf = tk.Label(
            self.info_frame, text="CONF: ---",
            font=("Consolas", 14, "bold"),
            bg=PANEL_BG, fg=TEXT_MAIN
        )
        self.lbl_conf.grid(row=1, column=0, sticky="w", padx=12, pady=(8, 2))

        self.lbl_peak = tk.Label(
            self.info_frame, text="PEAK: ---",
            font=("Consolas", 14, "bold"),
            bg=PANEL_BG, fg=TEXT_MAIN
        )
        self.lbl_peak.grid(row=2, column=0, sticky="w", padx=12, pady=(2, 2))

        self.lbl_sector = tk.Label(
            self.info_frame, text="SECTOR: ---",
            font=("Consolas", 14, "bold"),
            bg=PANEL_BG, fg=ACCENT_AMBER
        )
        self.lbl_sector.grid(row=3, column=0, sticky="w", padx=12, pady=(8, 2))

        self.lbl_quality = tk.Label(
            self.info_frame, text="LOBE: --- | R: ---",
            font=("Consolas", 12, "bold"),
            bg=PANEL_BG, fg=ACCENT_AMBER
        )
        self.lbl_quality.grid(row=4, column=0, sticky="w", padx=12, pady=(2, 6))

        self.lbl_udp = tk.Label(
            self.info_frame, text=f"UDP: {'ON' if self.udp_enabled else 'OFF'}  {self.udp_ip}:{self.udp_port}",
            font=("Consolas", 12, "bold"),
            bg=PANEL_BG, fg=ACCENT_AMBER
        )
        self.lbl_udp.grid(row=5, column=0, sticky="w", padx=12, pady=(8, 8))

        btns = tk.Frame(self.info_frame, bg=PANEL_BG)
        btns.grid(row=6, column=0, sticky="ew", padx=10, pady=(10, 6))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)

        tk.Button(
            btns, text="UDP ON/OFF",
            font=("Verdana", 10, "bold"),
            command=self.toggle_udp,
            bg="#29323C", fg="white",
            activebackground="#3C4A58", activeforeground="white",
            relief="raised", bd=2
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))

        tk.Button(
            btns, text="CONFIG UDP",
            font=("Verdana", 10, "bold"),
            command=self.open_udp_dialog,
            bg="#29323C", fg="white",
            activebackground="#3C4A58", activeforeground="white",
            relief="raised", bd=2
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # Control rápido MA-USB8 (si lo soporta el firmware)
        ctrl = tk.Frame(self.info_frame, bg=PANEL_BG)
        ctrl.grid(row=7, column=0, sticky="ew", padx=10, pady=(10, 10))
        ctrl.columnconfigure(0, weight=1)
        ctrl.columnconfigure(1, weight=1)

        tk.Label(ctrl, text="THRESH (t/T)",
                 font=("Verdana", 10, "bold"),
                 bg=PANEL_BG, fg=TEXT_MAIN).grid(row=0, column=0, sticky="w", padx=6, pady=6)
        sub = tk.Frame(ctrl, bg=PANEL_BG)
        sub.grid(row=0, column=1, sticky="e", padx=6, pady=6)

        tk.Button(sub, text="t -",
                  font=("Verdana", 10, "bold"),
                  command=lambda: self.worker.send_command("t"),
                  bg="#29323C", fg="white",
                  activebackground="#3C4A58", activeforeground="white",
                  relief="raised", bd=2).pack(side="left", padx=4)

        tk.Button(sub, text="T +",
                  font=("Verdana", 10, "bold"),
                  command=lambda: self.worker.send_command("T"),
                  bg="#29323C", fg="white",
                  activebackground="#3C4A58", activeforeground="white",
                  relief="raised", bd=2).pack(side="left", padx=4)

    def _create_led(self, parent, text, color):
        block = tk.Frame(parent, bg=STATUS_STRIP_BG)
        led = tk.Label(block, bg=color, width=2, height=1, relief="sunken")
        led.pack(side="left", padx=(0, 4))
        tk.Label(block, text=text,
                 font=("Verdana", 8, "bold"),
                 bg=STATUS_STRIP_BG, fg=TEXT_MAIN).pack(side="left")
        block._led = led
        return block

    # ---------------- CONNECT / UDP / SECTOR ----------------
    def toggle_sector(self):
        self.sector_enabled = not self.sector_enabled

    def _auto_detect_and_connect(self):
        ports = find_candidate_com_ports()
        if ports:
            self.com_port = ports[0][0]
            self.lbl_port.config(text=f"COM: {self.com_port}")
            self.worker.set_port(self.com_port, self.baud)
        else:
            self.lbl_status.config(text="ESTADO: no detecto COM (abre CONECTAR)", fg=ACCENT_AMBER)

    def open_connection_dialog(self):
        if serial is None:
            self.lbl_status.config(text="ESTADO: instala pyserial (py -m pip install pyserial)", fg=ACCENT_DANGER)
            return

        win = tk.Toplevel(self.root)
        win.title("Conectar MA-USB8 (CDC-ACM)")
        win.configure(bg=BG_MAIN)
        win.resizable(False, False)

        pad = {"padx": 10, "pady": 6}
        ports = find_candidate_com_ports()

        tk.Label(win, text="Puerto COM:", bg=BG_MAIN, fg=TEXT_MAIN,
                 font=("Verdana", 10, "bold")).grid(row=0, column=0, sticky="w", **pad)

        var_com = tk.StringVar(value=self.com_port if self.com_port else (ports[0][0] if ports else ""))
        opt = tk.OptionMenu(win, var_com, *([p[0] for p in ports] if ports else [""]))
        opt.config(bg="#29323C", fg="white", relief="raised")
        opt.grid(row=0, column=1, sticky="ew", **pad)

        tk.Label(win, text="Baud:", bg=BG_MAIN, fg=TEXT_MAIN,
                 font=("Verdana", 10, "bold")).grid(row=1, column=0, sticky="w", **pad)
        var_baud = tk.StringVar(value=str(self.baud))
        tk.Entry(win, textvariable=var_baud, width=14).grid(row=1, column=1, sticky="w", **pad)

        def do_connect():
            self.com_port = var_com.get().strip()
            try:
                self.baud = int(float(var_baud.get()))
            except Exception:
                self.baud = DEFAULT_BAUD

            self.lbl_port.config(text=f"COM: {self.com_port if self.com_port else '---'}")
            self.worker.set_port(self.com_port, self.baud)
            win.destroy()

        tk.Button(win, text="Aplicar",
                  font=("Verdana", 10, "bold"),
                  command=do_connect,
                  bg="#29323C", fg="white",
                  activebackground="#3C4A58", activeforeground="white",
                  relief="raised", bd=2).grid(row=2, column=0, columnspan=2, pady=10)

    def toggle_udp(self):
        self.udp_enabled = not self.udp_enabled
        self.lbl_udp.config(text=f"UDP: {'ON' if self.udp_enabled else 'OFF'}  {self.udp_ip}:{self.udp_port}",
                            fg=ACCENT_AMBER if self.udp_enabled else GRID_DIM)
        if not self.udp_enabled:
            self._udp_sock = None

    def open_udp_dialog(self):
        win = tk.Toplevel(self.root)
        win.title("Configurar UDP (salida DOA)")
        win.configure(bg=BG_MAIN)
        win.resizable(False, False)
        pad = {"padx": 10, "pady": 6}

        tk.Label(win, text="IP destino:", bg=BG_MAIN, fg=TEXT_MAIN,
                 font=("Verdana", 10, "bold")).grid(row=0, column=0, sticky="w", **pad)
        var_ip = tk.StringVar(value=self.udp_ip)
        tk.Entry(win, textvariable=var_ip, width=18).grid(row=0, column=1, **pad)

        tk.Label(win, text="Puerto:", bg=BG_MAIN, fg=TEXT_MAIN,
                 font=("Verdana", 10, "bold")).grid(row=1, column=0, sticky="w", **pad)
        var_port = tk.StringVar(value=str(self.udp_port))
        tk.Entry(win, textvariable=var_port, width=18).grid(row=1, column=1, **pad)

        def apply_udp():
            self.udp_ip = var_ip.get().strip() or "127.0.0.1"
            try:
                self.udp_port = int(float(var_port.get()))
            except Exception:
                self.udp_port = 5055
            self.lbl_udp.config(text=f"UDP: {'ON' if self.udp_enabled else 'OFF'}  {self.udp_ip}:{self.udp_port}",
                                fg=ACCENT_AMBER if self.udp_enabled else GRID_DIM)
            win.destroy()

        tk.Button(win, text="Guardar",
                  font=("Verdana", 10, "bold"),
                  command=apply_udp,
                  bg="#29323C", fg="white",
                  activebackground="#3C4A58", activeforeground="white",
                  relief="raised", bd=2).grid(row=2, column=0, columnspan=2, pady=10)

    # ---------------- DRAW ----------------
    def _first_draw(self):
        self._draw_hotmap_grid(None)
        self._draw_compass(None, None, 0.0)

    def _draw_hotmap_grid(self, hm256):
        c = self.hot_canvas
        c.delete("all")
        w = max(c.winfo_width(), 320)
        h = max(c.winfo_height(), 320)

        m = 10
        x0, y0, x1, y1 = m, m, w - m, h - m
        c.create_rectangle(x0, y0, x1, y1, outline=GRID_DIM)

        cw = (x1 - x0) / 16.0
        ch = (y1 - y0) / 16.0

        if hm256 is None:
            for r in range(16):
                for col in range(16):
                    xx0 = x0 + col * cw
                    yy0 = y0 + r * ch
                    c.create_rectangle(xx0, yy0, xx0 + cw, yy0 + ch, outline=GRID_DIM)
            return

        vmax = max(hm256) if hm256 else 1
        vmax = max(vmax, 1)

        for r in range(16):
            for col in range(16):
                v = hm256[r * 16 + col]
                g = int(clamp((v / vmax) * 255, 0, 255))
                color = f"#{0:02x}{g:02x}{0:02x}"
                xx0 = x0 + col * cw
                yy0 = y0 + r * ch
                c.create_rectangle(xx0, yy0, xx0 + cw, yy0 + ch, outline=GRID_DIM, fill=color)

        px, py = self.peak_xy
        px = int(clamp(px, 0, 15))
        py = int(clamp(py, 0, 15))
        xx0 = x0 + px * cw
        yy0 = y0 + py * ch
        c.create_rectangle(xx0, yy0, xx0 + cw, yy0 + ch, outline=ACCENT_AMBER, width=3)

    def _draw_compass(self, az_deg, sector_half_deg, conf):
        c = self.comp_canvas
        c.delete("all")
        w = max(c.winfo_width(), 320)
        h = max(c.winfo_height(), 320)

        cx = w / 2
        cy = h / 2
        R = min(w, h) * 0.38

        c.create_oval(cx - R, cy - R, cx + R, cy + R, outline=GRID_DIM, width=2)
        c.create_oval(cx - R * 0.66, cy - R * 0.66, cx + R * 0.66, cy + R * 0.66, outline=GRID_DIM, width=1)
        c.create_oval(cx - R * 0.33, cy - R * 0.33, cx + R * 0.33, cy + R * 0.33, outline=GRID_DIM, width=1)

        c.create_line(cx - R, cy, cx + R, cy, fill=GRID_DIM, width=1)
        c.create_line(cx, cy - R, cx, cy + R, fill=GRID_DIM, width=1)

        c.create_text(cx, cy - R - 14, text="N", fill=TEXT_MAIN, font=("Verdana", 12, "bold"))
        c.create_text(cx + R + 14, cy, text="E", fill=TEXT_MAIN, font=("Verdana", 12, "bold"))
        c.create_text(cx, cy + R + 14, text="S", fill=TEXT_MAIN, font=("Verdana", 12, "bold"))
        c.create_text(cx - R - 14, cy, text="W", fill=TEXT_MAIN, font=("Verdana", 12, "bold"))

        # SECTOR serio (abanico)
        if self.sector_enabled and (az_deg is not None) and (sector_half_deg is not None):
            half = float(sector_half_deg)
            a0 = deg_norm(az_deg - half)
            a1 = deg_norm(az_deg + half)

            # Tk: start desde +X (derecha), CCW. Convert: theta_tk = 90 - az
            def to_tk_angle(az):
                return deg_norm(90.0 - az)

            # Queremos dibujar el abanico pequeño de a0->a1 alrededor de az
            start = to_tk_angle(a1)
            extent = (to_tk_angle(a0) - to_tk_angle(a1)) % 360.0
            # nos aseguramos de coger el arco corto
            if extent > 180.0:
                extent = 360.0 - extent
                start = to_tk_angle(a0)

            bbox = (cx - R * 0.98, cy - R * 0.98, cx + R * 0.98, cy + R * 0.98)
            c.create_arc(*bbox, start=start, extent=extent,
                         fill=ACCENT_AMBER, outline="", style="pieslice", stipple="gray50")

            p0 = az_to_canvas_xy(cx, cy, R * 0.98, a0)
            p1 = az_to_canvas_xy(cx, cy, R * 0.98, a1)
            c.create_line(cx, cy, p0[0], p0[1], fill=ACCENT_AMBER, width=2)
            c.create_line(cx, cy, p1[0], p1[1], fill=ACCENT_AMBER, width=2)

        # Flecha DOA
        if az_deg is not None:
            x_end, y_end = az_to_canvas_xy(cx, cy, R * 0.92, az_deg)
            width = int(2 + 6 * clamp(conf, 0.0, 1.0))
            c.create_line(cx, cy, x_end, y_end, fill=RADAR_LINE, width=width)

            head = 12 + 10 * clamp(conf, 0.0, 1.0)
            left = deg_norm(az_deg - 150)
            right = deg_norm(az_deg + 150)
            xl, yl = az_to_canvas_xy(cx, cy, head, left)
            xr, yr = az_to_canvas_xy(cx, cy, head, right)
            c.create_polygon(x_end, y_end, xl, yl, xr, yr, fill=RADAR_LINE, outline="")

        # barra confianza
        bar_w = w * 0.75
        bar_h = 16
        bx0 = (w - bar_w) / 2
        by0 = h - 30
        bx1 = bx0 + bar_w
        by1 = by0 + bar_h
        c.create_rectangle(bx0, by0, bx1, by1, outline=GRID_DIM)
        fill = bar_w * clamp(conf, 0.0, 1.0)
        c.create_rectangle(bx0, by0, bx0 + fill, by1, outline="", fill=ACCENT_AMBER)
        c.create_text(cx, by0 - 10, text=f"CONF {conf:0.2f}", fill=ACCENT_AMBER,
                      font=("Consolas", 12, "bold"))

    # ---------------- UDP ----------------
    def _send_udp(self):
        if not self.udp_enabled:
            return
        if self._udp_sock is None:
            import socket
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        msg = {
            "ts": time.time(),
            "az_deg": float(self.az_deg) if self.az_deg is not None else -1.0,
            "sector_half_deg": float(self.sector_half) if self.sector_half is not None else -1.0,
            "conf": float(self.conf),
            "vmax": int(self.vmax),
            "vmean": float(self.vmean),
            "peak_xy": [int(self.peak_xy[0]), int(self.peak_xy[1])],
            "lobe_count": int(self.lobe_count),
            "R": float(self.R),
        }
        try:
            self._udp_sock.sendto(json.dumps(msg).encode("utf-8"), (self.udp_ip, self.udp_port))
        except Exception:
            pass

    # ---------------- MAIN LOOP ----------------
    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_q.get_nowait()
                kind = msg[0]

                if kind == "STATUS":
                    _, level, text = msg
                    if level == "OK":
                        self.lbl_status.config(text=f"ESTADO: {text}", fg=ACCENT_SAFE)
                        self.led_link._led.config(bg=ACCENT_SAFE)
                    elif level == "WARN":
                        self.lbl_status.config(text=f"ESTADO: {text}", fg=ACCENT_AMBER)
                        self.led_link._led.config(bg=ACCENT_AMBER)
                    else:
                        self.lbl_status.config(text=f"ESTADO: {text}", fg=ACCENT_DANGER)
                        self.led_link._led.config(bg=ACCENT_DANGER)

                elif kind == "HOTMAP":
                    _, hm, az_s, sector_s, conf, vmax, vmean, peak_xy, lobe_count, R = msg
                    self.current_hotmap = hm
                    self.az_deg = az_s
                    self.sector_half = sector_s
                    self.conf = conf
                    self.vmax = vmax
                    self.vmean = vmean
                    self.peak_xy = peak_xy
                    self.lobe_count = lobe_count
                    self.R = R

                    good = (az_s is not None) and (conf >= 0.10) and (sector_s is not None)
                    self.led_doa._led.config(bg=ACCENT_SAFE if good else ACCENT_AMBER)

                    # labels
                    if self.az_deg is None:
                        self.lbl_az.config(text="AZ: --- °")
                        self.lbl_sector.config(text="SECTOR: ---")
                    else:
                        self.lbl_az.config(text=f"AZ: {self.az_deg:6.1f} °")
                        if self.sector_half is not None:
                            self.lbl_sector.config(text=f"SECTOR: {self.az_deg:5.1f}° ± {self.sector_half:4.1f}°")
                        else:
                            self.lbl_sector.config(text="SECTOR: ---")

                    self.lbl_conf.config(text=f"CONF: {self.conf:0.2f}")
                    self.lbl_peak.config(text=f"PEAK: {self.vmax:3d} | MEAN: {self.vmean:5.1f} | XY: {self.peak_xy}")
                    self.lbl_quality.config(text=f"LOBE: {self.lobe_count:2d} | R: {self.R:0.2f}")

                    # draw
                    self._draw_hotmap_grid(hm)
                    self._draw_compass(self.az_deg, self.sector_half, self.conf)

                    # udp
                    self._send_udp()

        except queue.Empty:
            pass

        self.after_ids.append(self.root.after(60, self._poll_queue))

    # ---------------- CLOSE ----------------
    def on_close(self):
        self.stop_event.set()
        for aid in self.after_ids:
            try:
                self.root.after_cancel(aid)
            except Exception:
                pass
        self.after_ids.clear()
        try:
            self.root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = DOAConsoleUI(root)
    root.mainloop()
