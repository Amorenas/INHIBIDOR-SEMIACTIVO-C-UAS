#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CIC ONEFILE - Umbral Adaptativo + DOA (Sipeed MA-USB8) + ML (DADS)

Un único fichero Python para:
  1) Preparar un PC Linux "desde cero" (apt + venv + pip)
  2) Descargar/usar dataset (DADS en parquet) y entrenar modelo
  3) Ejecutar GUI tipo CIC con:
        - dB2 + umbral adaptativo (calibración/recalibración)
        - ML (probabilidad dron)
        - DOA desde hotmap por serial (/dev/ttyACM* / /dev/serial/by-id/*)

USO:
  cd /home/ubuntu/proyectos/detector
  sudo -E python3 cic_onefile.py

Dataset:
  - Busca parquets en: data/dads/data/*.parquet
  - Si no existen, intentará descargar desde Hugging Face.
    Recomendado:
      export DADS_REPO="<repo_hf>"

"""
from __future__ import annotations

import os, sys, math, time, glob, random, queue, threading, subprocess, pathlib
from dataclasses import dataclass
from collections import deque
from typing import Optional, List

APP_TITLE = "CIC Unificado - Umbral + DOA + ML (OneFile)"
ROOT = pathlib.Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "drone_clf.joblib"
DATA_ROOT = ROOT / "data" / "dads" / "data"
LOG_PATH = ROOT / "logs" / "cic_onefile.log"

APT_PACKAGES = [
    "python3", "python3-venv", "python3-pip", "python3-tk",
    "portaudio19-dev", "alsa-utils", "udev", "ffmpeg",
    "build-essential", "git", "git-lfs"
]
PIP_PACKAGES = [
    "numpy", "scipy", "sounddevice", "pyserial", "pillow",
    "tqdm", "soundfile", "librosa", "scikit-learn", "joblib",
    "huggingface_hub", "datasets", "pyarrow"
]

TARGET_SR = 16000
CAPTURE_SR_DEFAULT = 48000
FRAME_S_DEFAULT = 1.0
BANDPASS_HZ = (3000.0, 6000.0)
BUTTER_ORDER = 5

CALIB_SEC = 8.0
RECALIB_EVERY_SEC = 180.0
RECALIB_MIN_STABLE_SEC = 8.0
DETECTION_HOLD_SEC = 2.0

ML_THRESH_DEFAULT = 0.60
TRAIN_MAX_ITEMS_DEFAULT = 12000

HOTMAP_HEADER = bytes([0xFF] * 16)
HOTMAP_FRAME_LEN = 16 + 256
DEFAULT_BAUD = 115200
RECONNECT_SEC = 2.0

AZ_SMOOTH_N = 5
LOBE_FRAC = 0.70
R_MIN = 1.0
STD_MULT = 2.2
SECTOR_MIN_HALF_DEG = 8.0
SECTOR_MAX_HALF_DEG = 70.0
SECTOR_FIXED_HALF_ANGLE_DEG = None
MIN_LOBE_POINTS = 6
MIN_LOBE_WEIGHT = 50.0

def _now() -> str:
    return time.strftime("%H:%M:%S")

def log(msg: str):
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass

def info(msg: str):
    print(msg, flush=True)
    log(msg)

def run_cmd(cmd: List[str], check: bool = True, env: Optional[dict] = None) -> int:
    info(f">>> {' '.join(cmd)}")
    try:
        p = subprocess.run(cmd, check=check, env=env)
        return int(p.returncode)
    except subprocess.CalledProcessError as e:
        info(f"[ERR] comando falló rc={e.returncode}: {' '.join(cmd)}")
        return int(e.returncode)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        info(f"[ERR] excepción ejecutando comando: {e}")
        return 1

def is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0

def have_internet_hint() -> bool:
    try:
        import socket
        socket.gethostbyname("archive.ubuntu.com")
        return True
    except Exception:
        return False

def ensure_apt():
    if not is_root():
        info("[WARN] No soy root; no puedo instalar por apt. Continúo.")
        return
    run_cmd(["apt-get", "update"], check=False)
    run_cmd(["apt-get", "install", "-y"] + APT_PACKAGES, check=False)

def ensure_user_dialout():
    if not is_root():
        return
    user = os.environ.get("SUDO_USER") or os.environ.get("USER") or "ubuntu"
    run_cmd(["usermod", "-aG", "dialout", user], check=False)
    run_cmd(["udevadm", "control", "--reload-rules"], check=False)
    run_cmd(["udevadm", "trigger"], check=False)
    run_cmd(["bash", "-lc", "ls -l /dev/ttyACM* 2>/dev/null || true"], check=False)
    run_cmd(["bash", "-lc", "chmod a+rw /dev/ttyACM* 2>/dev/null || true"], check=False)

def ensure_venv_and_reexec():
    py_in_venv = VENV_DIR / "bin" / "python3"
    if sys.prefix == str(VENV_DIR):
        return
    if not py_in_venv.exists():
        info("[BOOT] Creando venv...")
        run_cmd(["python3", "-m", "venv", str(VENV_DIR)], check=True)
    pip = VENV_DIR / "bin" / "pip"
    info("[BOOT] Actualizando pip/setuptools/wheel...")
    run_cmd([str(pip), "install", "--upgrade", "pip", "setuptools", "wheel"], check=False)
    info("[BOOT] Instalando dependencias Python...")
    run_cmd([str(pip), "install"] + PIP_PACKAGES, check=False)

    info("[BOOT] Re-ejecutando dentro del venv...")
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(VENV_DIR)
    env["PATH"] = str(VENV_DIR / "bin") + ":" + env.get("PATH", "")
    os.execve(str(py_in_venv), [str(py_in_venv), str(__file__)] + sys.argv[1:], env)

def lazy_imports():
    global np, scipy, sd, serial, serial_tools, librosa, joblib, pq, tk
    import numpy as np
    import scipy, scipy.signal
    import sounddevice as sd
    try:
        import serial
        import serial.tools.list_ports as serial_tools
    except Exception:
        serial = None
        serial_tools = None
    import librosa
    import joblib
    import pyarrow.parquet as pq
    import tkinter as tk
    return np, scipy

@dataclass
class FeatCfg:
    sr: int = TARGET_SR
    n_mels: int = 40
    n_fft: int = 1024
    hop: int = 256
    fmin: float = 300.0
    fmax: float = 8000.0

def butter_bandpass(scipy_signal, lo: float, hi: float, fs: float, order: int = 5):
    nyq = 0.5 * fs
    lo_n = max(1e-6, lo / nyq)
    hi_n = min(0.999999, hi / nyq)
    b, a = scipy_signal.butter(order, [lo_n, hi_n], btype="band")
    return b, a

def energy_db(x: "np.ndarray") -> float:
    eps = 1e-12
    rms = float((x.astype("float64")**2).mean()**0.5)
    return 20.0 * math.log10(rms + eps)

def resample_to(x: "np.ndarray", sr_in: int, sr_out: int, librosa_mod) -> "np.ndarray":
    if sr_in == sr_out:
        return x
    y = librosa_mod.resample(x.astype("float32"), orig_sr=sr_in, target_sr=sr_out, res_type="kaiser_fast")
    return y.astype("float32")

def extract_features(x_16k: "np.ndarray", cfg: FeatCfg, librosa_mod) -> "np.ndarray":
    S = librosa_mod.feature.melspectrogram(
        y=x_16k.astype("float32"), sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop,
        n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax, power=2.0
    )
    logS = librosa_mod.power_to_db(S + 1e-10, ref=1.0)
    mu = logS.mean(axis=1)
    sdv = logS.std(axis=1)
    return np.concatenate([mu, sdv], axis=0).astype("float32")

def find_parquets() -> List[str]:
    return sorted(glob.glob(str(DATA_ROOT / "*.parquet")))

def ensure_dataset():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if find_parquets():
        info(f"[OK] Dataset local presente en {DATA_ROOT}")
        return
    info("[BOOT] No encuentro parquets. Intentando descargar dataset (HF)...")
    if not have_internet_hint():
        raise SystemExit("No hay internet y no existe dataset local.")
    from datasets import load_dataset
    repo = os.environ.get("DADS_REPO", "").strip()
    candidates = [repo] if repo else []
    candidates += ["DroneAudioDetectionSamples/DADS", "drone-audio-detection-samples"]
    last = None
    for rid in candidates:
        if not rid:
            continue
        try:
            info(f"[BOOT] load_dataset('{rid}', split='train')")
            ds = load_dataset(rid, split="train")
            import pyarrow as pa
            import pyarrow.parquet as pq2
            shard = 0
            batch = []
            batch_size = 4096
            for ex in ds:
                batch.append({"audio": ex["audio"], "label": int(ex["label"])})
                if len(batch) >= batch_size:
                    pq2.write_table(pa.Table.from_pylist(batch), DATA_ROOT / f"train-{shard:05d}.parquet")
                    shard += 1
                    batch.clear()
            if batch:
                pq2.write_table(pa.Table.from_pylist(batch), DATA_ROOT / f"train-{shard:05d}.parquet")
            if find_parquets():
                info("[OK] Dataset descargado.")
                return
        except Exception as e:
            last = e
            info(f"[WARN] Falló '{rid}': {e}")
    raise SystemExit(f"No pude descargar dataset. Define DADS_REPO. Último error: {last}")

def iter_parquet_examples(parquet_files: List[str], rng: random.Random):
    files = parquet_files[:]
    rng.shuffle(files)
    for f in files:
        try:
            table = pq.read_table(f)
            pdf = table.to_pandas()
            idx = list(range(len(pdf)))
            rng.shuffle(idx)
            for i in idx:
                row = pdf.iloc[i]
                yield row.get("audio"), int(row.get("label"))
        except Exception as e:
            info(f"[WARN] Error leyendo {f}: {e}")

def train_model(max_items: int = TRAIN_MAX_ITEMS_DEFAULT, target_sr: int = TARGET_SR):
    lazy_imports()
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    ensure_dataset()
    parqs = find_parquets()
    if not parqs:
        raise RuntimeError("No hay parquets para entrenar")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler(with_mean=True, with_std=True)
    clf = SGDClassifier(loss="log_loss", alpha=1e-5, penalty="l2", max_iter=1, tol=None, random_state=123)
    classes = np.array([0, 1], dtype=int)

    def batch_weights(yb: "np.ndarray") -> "np.ndarray":
        eps = 1e-6
        c0 = float((yb == 0).sum()); c1 = float((yb == 1).sum())
        w0 = 1.0 / (c0 + eps); w1 = 1.0 / (c1 + eps)
        w = np.where(yb == 0, w0, w1)
        w = w * (len(yb) / (w.sum() + eps))
        return w.astype("float32")

    rng = random.Random(12345)
    Xb, yb = [], []
    B = 64
    n_ok = y0 = y1 = 0

    def flush(first: bool):
        nonlocal n_ok, y0, y1, Xb, yb
        if not Xb:
            return
        X = np.vstack(Xb).astype("float32")
        y = np.array(yb, dtype=int)
        scaler.partial_fit(X)
        Xs = scaler.transform(X)
        sw = batch_weights(y)
        if first:
            clf.partial_fit(Xs, y, classes=classes, sample_weight=sw)
        else:
            clf.partial_fit(Xs, y, sample_weight=sw)
        n_ok += len(y)
        y0 += int((y == 0).sum()); y1 += int((y == 1).sum())
        Xb, yb = [], []

    info(f"[ML] Entrenando: max_items={max_items}")
    for audio_obj, lab in iter_parquet_examples(parqs, rng):
        if n_ok >= max_items:
            break
        if not (isinstance(audio_obj, dict) and "array" in audio_obj):
            continue
        try:
            x = np.array(audio_obj["array"], dtype="float32")
            sr_in = int(audio_obj.get("sampling_rate", target_sr))
            if x.ndim > 1:
                x = x.mean(axis=1)
            n = int(sr_in * 2.0)
            if len(x) < n:
                x = np.pad(x, (0, n-len(x)))
            else:
                x = x[:n]
            x16 = resample_to(x, sr_in, target_sr, librosa)
            feat = extract_features(x16, FeatCfg(sr=target_sr), librosa)[None, :]
            Xb.append(feat); yb.append(int(lab))
            if len(Xb) >= B:
                flush(first=(n_ok == 0))
        except Exception:
            pass

    flush(first=(n_ok == 0))
    if n_ok < 300:
        raise RuntimeError(f"Entrenamiento insuficiente: n_ok={n_ok}, y0={y0}, y1={y1}")
    payload = {"created_ts": time.time(), "cfg": FeatCfg(sr=target_sr), "scaler": scaler, "clf": clf}
    import joblib as jb
    jb.dump(payload, MODEL_PATH)
    info(f"[ML] Guardado {MODEL_PATH} n_ok={n_ok} y0={y0} y1={y1}")

def load_model():
    lazy_imports()
    payload = joblib.load(MODEL_PATH)
    return payload.get("cfg", FeatCfg()), payload["scaler"], payload["clf"]

def clamp(v, a, b): return max(a, min(b, v))
def deg_norm(a): return (a + 360.0) % 360.0

def circular_mean_deg(angles_deg, weights=None):
    if not angles_deg: return None
    if weights is None: weights = [1.0]*len(angles_deg)
    s=c=ws=0.0
    for a,w in zip(angles_deg,weights):
        r=math.radians(a); s+=w*math.sin(r); c+=w*math.cos(r); ws+=w
    if ws<=0: return None
    return deg_norm(math.degrees(math.atan2(s, c)))

def circular_resultant_and_std_rad(angles_deg, weights):
    s=c=ws=0.0
    for a,w in zip(angles_deg,weights):
        r=math.radians(a); s+=w*math.sin(r); c+=w*math.cos(r); ws+=w
    if ws<=0: return 0.0, float("inf")
    R = math.hypot(s, c)/ws
    R = clamp(R, 1e-6, 1.0)
    sigma = math.sqrt(max(0.0, -2.0*math.log(R)))
    return float(R), float(sigma)

def hotmap_stats(hm256):
    vmax=-1; imax=0; s=0
    for i,v in enumerate(hm256):
        if v>vmax: vmax=v; imax=i
        s+=v
    vmean=s/256.0
    return int(vmax), float(vmean), (imax%16, imax//16)

def hotmap_to_doa_sector_conf(hm256):
    vmax, vmean, peak_xy = hotmap_stats(hm256)
    if vmax<=0: return None, None, 0.0, vmax, vmean, peak_xy, 0, 0.0
    cx=7.5; cy=7.5
    thr=LOBE_FRAC*vmax
    angles=[]; weights=[]; rads=[]; lobe_count=0
    for idx,v in enumerate(hm256):
        if v<thr: continue
        y=idx//16; x=idx%16
        dx=x-cx; dy=y-cy
        r=math.hypot(dx,dy)
        if r<R_MIN: continue
        az=deg_norm(math.degrees(math.atan2(dx, -dy)))
        w=max(0.0, float(v)-float(vmean))
        if w<=0: continue
        angles.append(az); weights.append(w); rads.append(r); lobe_count+=1
    sumw=sum(weights) if weights else 0.0
    if lobe_count<MIN_LOBE_POINTS or sumw<MIN_LOBE_WEIGHT:
        px,py=peak_xy; dx=px-cx; dy=py-cy
        r=math.hypot(dx,dy)
        if r<R_MIN: return None,None,0.0,vmax,vmean,peak_xy,lobe_count,0.0
        az=deg_norm(math.degrees(math.atan2(dx, -dy)))
        sector_half=SECTOR_MAX_HALF_DEG
        prom=clamp((vmax-vmean)/255.0,0,1)
        radial=clamp(r/10.6,0,1)
        conf=clamp(0.25*prom*radial,0,1)
        return az, sector_half, conf, vmax, vmean, peak_xy, lobe_count, 0.0
    az_deg=circular_mean_deg(angles,weights)
    R,sigma_rad=circular_resultant_and_std_rad(angles,weights)
    sigma_deg=math.degrees(sigma_rad)
    sector_half = float(SECTOR_FIXED_HALF_ANGLE_DEG) if SECTOR_FIXED_HALF_ANGLE_DEG is not None else clamp(STD_MULT*sigma_deg, SECTOR_MIN_HALF_DEG, SECTOR_MAX_HALF_DEG)
    r_mean=sum(rads)/max(1,len(rads))
    radial=clamp(r_mean/10.6,0,1)
    prom=clamp((vmax-vmean)/255.0,0,1)
    conf=clamp(0.20*prom + 0.55*prom*R + 0.25*prom*radial,0,1)
    return az_deg, sector_half, conf, vmax, vmean, peak_xy, lobe_count, R

def find_serial_candidates():
    out=[]
    out += sorted(glob.glob("/dev/serial/by-id/*"))
    for p in sorted(glob.glob("/dev/ttyACM*")):
        if p not in out: out.append(p)
    if serial_tools is not None:
        try:
            for p in serial_tools.comports():
                if p.device and p.device not in out:
                    out.append(p.device)
        except Exception:
            pass
    return out

class SerialDOAWorker(threading.Thread):
    def __init__(self, out_q: "queue.Queue", stop_event: threading.Event):
        super().__init__(daemon=True)
        self.q=out_q; self.stop_event=stop_event
        self.port=None; self.baud=DEFAULT_BAUD
        self._ser=None; self._buf=bytearray()
        self.az_hist=deque(maxlen=AZ_SMOOTH_N); self.sector_hist=deque(maxlen=AZ_SMOOTH_N)
        self._lock=threading.Lock()

    def set_port(self, port: str):
        with self._lock: self.port=port

    def push(self, msg):
        try: self.q.put_nowait(msg)
        except queue.Full: pass

    def _try_open(self):
        if serial is None:
            self.push(("STATUS","ERR","pyserial no disponible")); return False
        with self._lock:
            port=self.port; baud=self.baud
        if not port:
            self.push(("STATUS","WARN","Sin puerto DOA")); return False
        try:
            self._ser=serial.Serial(port, baudrate=baud, timeout=0.15)
            self._buf=bytearray(); self.az_hist.clear(); self.sector_hist.clear()
            self.push(("STATUS","OK",f"DOA conectado: {port}")); return True
        except Exception as e:
            self._ser=None
            self.push(("STATUS","ERR",f"No puedo abrir {port}: {e}")); return False

    def _read_bytes(self, n=512):
        if self._ser is None: return b""
        try: return self._ser.read(n)
        except Exception: return b""

    def _extract_hotmap(self):
        chunk=self._read_bytes(512)
        if chunk: self._buf.extend(chunk)
        if len(self._buf)>8192: self._buf=self._buf[-4096:]
        idx=self._buf.find(HOTMAP_HEADER)
        if idx<0: return None
        end=idx+HOTMAP_FRAME_LEN
        if len(self._buf)<end: return None
        payload=self._buf[idx+16:idx+16+256]
        self._buf=self._buf[end:]
        if len(payload)!=256: return None
        return list(payload)

    def run(self):
        last_fail=0.0
        while not self.stop_event.is_set():
            if self._ser is None:
                now=time.time()
                if now-last_fail<RECONNECT_SEC:
                    time.sleep(0.05); continue
                if not self._try_open():
                    last_fail=time.time(); time.sleep(0.10); continue
            hm=self._extract_hotmap()
            if hm is None: continue
            az, sector, conf, vmax, vmean, peak_xy, lobe_count, R = hotmap_to_doa_sector_conf(hm)
            az_s = circular_mean_deg(list(self.az_hist)+([az] if az is not None else [])) if az is not None else None
            if az is not None: self.az_hist.append(az)
            if sector is not None: self.sector_hist.append(sector)
            sector_s = (sum(self.sector_hist)/len(self.sector_hist)) if self.sector_hist else None
            self.push(("HOTMAP", hm, az_s, sector_s, conf))
        try:
            if self._ser: self._ser.close()
        except Exception:
            pass

class AudioWorker(threading.Thread):
    def __init__(self, out_q: "queue.Queue", stop_event: threading.Event, device: Optional[int], capture_sr: int, frame_s: float):
        super().__init__(daemon=True)
        self.q=out_q; self.stop_event=stop_event
        self.device=device; self.capture_sr=int(capture_sr); self.frame_s=float(frame_s)
        self._buf=deque(); self._buf_n=0; self._lock=threading.Lock()

    def push(self, msg):
        try: self.q.put_nowait(msg)
        except queue.Full: pass

    def run(self):
        lazy_imports()
        block=int(self.capture_sr*0.10)
        need=int(self.capture_sr*self.frame_s)

        def callback(indata, frames, t, status):
            x=indata[:,0].copy()
            with self._lock:
                self._buf.append(x); self._buf_n += len(x)

        try:
            with sd.InputStream(device=self.device, channels=1, samplerate=self.capture_sr, blocksize=block, callback=callback):
                self.push(("AUDIO_STATUS","OK",f"Audio OK dev={self.device} sr={self.capture_sr}")) 
                while not self.stop_event.is_set():
                    with self._lock:
                        if self._buf_n>=need:
                            chunks=[]; n=0
                            while self._buf and n<need:
                                c=self._buf[0]; take=min(len(c), need-n)
                                chunks.append(c[:take]); n+=take
                                if take==len(c): self._buf.popleft()
                                else: self._buf[0]=c[take:]
                            self._buf_n -= need
                        else:
                            chunks=None
                    if chunks:
                        frame=np.concatenate(chunks).astype("float32")
                        self.push(("AUDIO_FRAME", frame, self.capture_sr))
                    else:
                        time.sleep(0.02)
        except Exception as e:
            self.push(("AUDIO_STATUS","ERR",f"Audio error: {e}"))

UI_BG="#0b0f14"; PANEL_BG="#141c25"; PANEL_EDGE="#263445"
TXT="#dfefff"; ACCENT="#00D68F"; AMBER="#F5C542"; DANGER="#FF3B3B"

def fmt_sector(az, half):
    if az is None or half is None: return "--"
    return f"{az:5.1f}° ± {half:4.1f}°"

def az_to_sector_name(az):
    if az is None: return "--"
    a=deg_norm(az)
    if a>=315 or a<45: return "PROA"
    if 45<=a<135: return "ESTRIBOR"
    if 135<=a<225: return "POPA"
    return "BABOR"

class CICApp:
    def __init__(self, root_tk):
        lazy_imports()
        self.root=root_tk
        self.root.title(APP_TITLE)
        self.root.configure(bg=UI_BG)
        self.root.geometry("1200x700")
        self.stop_event=threading.Event()
        self.msg_q=queue.Queue(maxsize=500)

        self.state="ARRANQUE"
        self.db2=-120.0
        self.thr=None
        self.last_recalib=0.0
        self.last_det_ts=0.0
        self.calib_until=time.time()+CALIB_SEC
        self.calib_buf=[]
        self.stable_buf=deque(maxlen=int(RECALIB_MIN_STABLE_SEC/FRAME_S_DEFAULT)+1)

        self.ml_ok=False; self.ml_p=0.0; self.ml_det=False; self.ml_thresh=ML_THRESH_DEFAULT
        self.feat_cfg=None; self.scaler=None; self.clf=None

        self.az_deg=None; self.sector_half=None; self.doa_conf=0.0; self.hotmap=None

        self.audio_device=None
        self.capture_sr=CAPTURE_SR_DEFAULT
        self.frame_s=FRAME_S_DEFAULT

        self._b=None; self._a=None
        self._build_ui()
        self._load_model_or_train()
        self._start_workers()
        self.root.after(50, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        import tkinter as tk
        top=tk.Frame(self.root, bg=UI_BG); top.pack(fill="x", padx=12, pady=(10,6))
        tk.Label(top, text="CIC UNIFICADO", font=("Verdana",20,"bold"), fg=TXT, bg=UI_BG).pack(side="left")
        self.lbl_ml=tk.Label(top, text="ML: ...", font=("Consolas",11,"bold"), fg=ACCENT, bg=UI_BG)
        self.lbl_ml.pack(side="right")

        main=tk.Frame(self.root, bg=UI_BG); main.pack(fill="both", expand=True, padx=12, pady=12)
        main.columnconfigure(0, weight=0); main.columnconfigure(1, weight=1); main.rowconfigure(0, weight=1)

        left=tk.Frame(main, bg=PANEL_BG, highlightbackground=PANEL_EDGE, highlightthickness=2, width=280)
        left.grid(row=0, column=0, sticky="nsw", padx=(0,10)); left.grid_propagate(False)
        tk.Label(left, text="ESTADO", font=("Verdana",12,"bold"), fg=TXT, bg=PANEL_BG).pack(pady=(10,2))
        self.lbl_state=tk.Label(left, text="---", font=("Verdana",18,"bold"), fg=AMBER, bg=PANEL_BG); self.lbl_state.pack(pady=(0,12))
        self.lbl_db=tk.Label(left, text="db2: --", font=("Consolas",12,"bold"), fg=TXT, bg=PANEL_BG); self.lbl_db.pack(anchor="w", padx=12, pady=4)
        self.lbl_thr=tk.Label(left, text="thr: --", font=("Consolas",12,"bold"), fg=TXT, bg=PANEL_BG); self.lbl_thr.pack(anchor="w", padx=12, pady=4)
        self.lbl_ml_p=tk.Label(left, text="pML: --", font=("Consolas",12,"bold"), fg=TXT, bg=PANEL_BG); self.lbl_ml_p.pack(anchor="w", padx=12, pady=4)
        self.lbl_doa=tk.Label(left, text="DOA: --", font=("Consolas",12,"bold"), fg=TXT, bg=PANEL_BG); self.lbl_doa.pack(anchor="w", padx=12, pady=4)
        tk.Button(left, text="Recalibrar", command=self.request_recalib).pack(fill="x", padx=12, pady=(12,6))
        tk.Button(left, text="Listar dispositivos", command=self.show_devices).pack(fill="x", padx=12, pady=(0,6))

        right=tk.Frame(main, bg=PANEL_BG, highlightbackground=PANEL_EDGE, highlightthickness=2)
        right.grid(row=0, column=1, sticky="nsew"); right.rowconfigure(1, weight=1); right.columnconfigure(0, weight=1)
        tk.Label(right, text="SALIDA (texto)", font=("Verdana",12,"bold"), fg=TXT, bg=PANEL_BG).grid(row=0, column=0, sticky="ew", pady=(8,4))
        self.txt=tk.Text(right, height=10, bg="#0e141b", fg="#bfe4ff", font=("Consolas",10), wrap="none")
        self.txt.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0,10))

        bottom=tk.Frame(self.root, bg=UI_BG); bottom.pack(fill="x", padx=12, pady=(0,12))
        bottom.columnconfigure(0, weight=1); bottom.columnconfigure(1, weight=1)
        self.canvas_hot=tk.Canvas(bottom, bg=PANEL_BG, height=220, highlightbackground=PANEL_EDGE, highlightthickness=2)
        self.canvas_hot.grid(row=0, column=0, sticky="ew", padx=(0,10))
        self.canvas_comp=tk.Canvas(bottom, bg=PANEL_BG, height=220, highlightbackground=PANEL_EDGE, highlightthickness=2)
        self.canvas_comp.grid(row=0, column=1, sticky="ew")
        self._draw_hotmap(None); self._draw_compass(None, None, 0.0)

    def _append(self, s: str):
        try:
            self.txt.insert("end", f"[{_now()}] {s}\n"); self.txt.see("end")
        except Exception:
            pass

    def show_devices(self):
        lazy_imports()
        try:
            devs=sd.query_devices()
            self._append("Audio input devices:")
            for i,d in enumerate(devs):
                if int(d.get("max_input_channels",0))>0:
                    self._append(f"  {i:2d} | in={d['max_input_channels']:2d} | sr={d.get('default_samplerate')} | {d['name']}")
        except Exception as e:
            self._append(f"[ERR] listdev: {e}")

    def request_recalib(self):
        self._append("[UI] Recalibración solicitada.")
        self.calib_buf.clear(); self.calib_until=time.time()+CALIB_SEC; self.thr=None; self.state="CALIBRANDO"

    def _load_model_or_train(self):
        if not MODEL_PATH.exists():
            self._append("Modelo ML no existe. Entrenando...")
            train_model(max_items=TRAIN_MAX_ITEMS_DEFAULT, target_sr=TARGET_SR)
        try:
            self.feat_cfg, self.scaler, self.clf = load_model()
            self.ml_ok=True
            self.lbl_ml.config(text=f"ML: OK ({MODEL_PATH})", fg=ACCENT)
        except Exception as e:
            self.ml_ok=False
            self.lbl_ml.config(text=f"ML: ERROR ({e})", fg=DANGER)
            self._append(f"[ERR] Cargando modelo: {e}")

    def _auto_pick_audio_device(self):
        lazy_imports()
        try:
            devs=sd.query_devices(); cand=None
            for i,d in enumerate(devs):
                if int(d.get("max_input_channels",0))<=0: continue
                name=(d.get("name") or "").lower()
                ch=int(d.get("max_input_channels",0))
                if "sipeed" in name or "micarray" in name: cand=i; break
                if ch>=8 and cand is None: cand=i
            if cand is None:
                cand = sd.default.device[0] if isinstance(sd.default.device,(tuple,list)) else None
            self.audio_device=cand
        except Exception:
            self.audio_device=None

    def _start_workers(self):
        lazy_imports()
        self._auto_pick_audio_device()
        try:
            d = sd.query_devices(self.audio_device)
            self.capture_sr=int(float(d.get("default_samplerate", CAPTURE_SR_DEFAULT)))
        except Exception:
            self.capture_sr=CAPTURE_SR_DEFAULT

        self._b, self._a = butter_bandpass(scipy.signal, BANDPASS_HZ[0], BANDPASS_HZ[1], TARGET_SR, BUTTER_ORDER)
        self._append(f"Audio device: {self.audio_device} sr={self.capture_sr}")

        self.audio_worker=AudioWorker(self.msg_q, self.stop_event, self.audio_device, self.capture_sr, self.frame_s)
        self.audio_worker.start()

        self.doa_worker=SerialDOAWorker(self.msg_q, self.stop_event)
        cands=find_serial_candidates()
        if cands:
            self.doa_worker.set_port(cands[0])
            self._append(f"DOA serial: {cands[0]}")
        else:
            self._append("DOA serial: (no detectado)")
        self.doa_worker.start()

        self.state="CALIBRANDO"
        self.last_recalib=time.time()

    def _draw_hotmap(self, hm256):
        c=self.canvas_hot; c.delete("all")
        w=max(c.winfo_width(),420); h=max(c.winfo_height(),220)
        c.create_text(10,10,anchor="nw",fill=TXT,font=("Verdana",11,"bold"),text="HOTMAP 16×16 (DOA)")
        x0,y0=10,30
        size=max(160, min(h-40, w-20))
        cw=size/16.0; ch=size/16.0
        if hm256 is None:
            for r in range(16):
                for col in range(16):
                    c.create_rectangle(x0+col*cw, y0+r*ch, x0+(col+1)*cw, y0+(r+1)*ch, outline=PANEL_EDGE)
            return
        vmax=max(1, int(max(hm256)))
        for r in range(16):
            for col in range(16):
                v=hm256[r*16+col]
                g=int(clamp((v/vmax)*255,0,255))
                color=f"#{0:02x}{g:02x}{0:02x}"
                c.create_rectangle(x0+col*cw, y0+r*ch, x0+(col+1)*cw, y0+(r+1)*ch, outline=PANEL_EDGE, fill=color)

    def _draw_compass(self, az_deg, sector_half, conf):
        c=self.canvas_comp; c.delete("all")
        w=max(c.winfo_width(),420); h=max(c.winfo_height(),220)
        c.create_text(10,10,anchor="nw",fill=TXT,font=("Verdana",11,"bold"),text="DOA (brújula)")
        cx=w*0.5; cy=h*0.55; R=min(w,h)*0.35
        c.create_oval(cx-R, cy-R, cx+R, cy+R, outline=PANEL_EDGE, width=2)
        c.create_text(cx, cy-R-10, text="N", fill=TXT, font=("Verdana",10,"bold"))
        c.create_text(cx+R+10, cy, text="E", fill=TXT, font=("Verdana",10,"bold"))
        c.create_text(cx, cy+R+10, text="S", fill=TXT, font=("Verdana",10,"bold"))
        c.create_text(cx-R-10, cy, text="W", fill=TXT, font=("Verdana",10,"bold"))
        if az_deg is not None:
            a=math.radians(az_deg)
            x_end=cx+R*0.9*math.sin(a); y_end=cy-R*0.9*math.cos(a)
            width=int(2+6*clamp(conf,0,1))
            c.create_line(cx, cy, x_end, y_end, fill=ACCENT, width=width)
        bw=w*0.6; bh=14; bx0=(w-bw)/2; by0=h-30
        c.create_rectangle(bx0, by0, bx0+bw, by0+bh, outline=PANEL_EDGE)
        c.create_rectangle(bx0, by0, bx0+bw*clamp(conf,0,1), by0+bh, outline="", fill=AMBER)
        c.create_text(cx, by0-10, text=f"CONF {conf:0.2f}", fill=AMBER, font=("Consolas",11,"bold"))

    def _update_left_panel(self):
        if self.state=="CALIBRANDO": self.lbl_state.config(text="CALIBRANDO", fg=AMBER)
        elif self.state=="DETECCIÓN": self.lbl_state.config(text="DETECCIÓN", fg=DANGER)
        else: self.lbl_state.config(text="VIGILANDO", fg=ACCENT)
        self.lbl_db.config(text=f"db2: {self.db2:6.1f}")
        self.lbl_thr.config(text=f"thr: {self.thr:6.1f}" if self.thr is not None else "thr: --")
        self.lbl_ml_p.config(text=f"pML: {self.ml_p:0.3f} | detML: {self.ml_det}")
        self.lbl_doa.config(text=f"DOA: {fmt_sector(self.az_deg,self.sector_half)} ({az_to_sector_name(self.az_deg)})")

    def _process_audio_frame(self, frame, sr_in):
        x16 = resample_to(frame, sr_in, TARGET_SR, librosa)
        try:
            xb = scipy.signal.lfilter(self._b, self._a, x16).astype("float32")
        except Exception:
            xb = x16
        self.db2 = energy_db(xb)
        now=time.time()
        self.stable_buf.append(self.db2)

        if now < self.calib_until:
            self.calib_buf.append(self.db2); self.state="CALIBRANDO"; return

        if self.thr is None:
            base=float(np.median(np.array(self.calib_buf,dtype="float32"))) if self.calib_buf else self.db2
            self.thr=base+6.0; self.state="VIGILANDO"; self.last_recalib=now
            self._append(f"[CAL] thr inicial={self.thr:0.1f} base={base:0.1f}")
        if (now-self.last_recalib)>RECALIB_EVERY_SEC and (now-self.last_det_ts)>(RECALIB_MIN_STABLE_SEC+1.0):
            if len(self.stable_buf)>=5:
                sigma=float(np.std(np.array(self.stable_buf,dtype="float32")))
                if sigma<2.0:
                    base=float(np.median(np.array(self.stable_buf,dtype="float32")))
                    self.thr=base+6.0; self.last_recalib=now
                    self._append(f"[CAL] recalib auto thr={self.thr:0.1f} sigma={sigma:0.2f}")
        if self.ml_ok:
            try:
                feat=extract_features(x16, self.feat_cfg, librosa)[None,:]
                feat_s=self.scaler.transform(feat)
                proba=self.clf.predict_proba(feat_s)[0,1]
                self.ml_p=float(proba); self.ml_det=bool(self.ml_p>=self.ml_thresh)
            except Exception as e:
                self.ml_p=0.0; self.ml_det=False; self._append(f"[ERR] ML infer: {e}")
        det_thr=(self.thr is not None) and (self.db2>=self.thr)
        det=det_thr and self.ml_det
        if det:
            self.last_det_ts=now; self.state="DETECCIÓN"
            self._append(f"[DET] db2={self.db2:0.1f} thr={self.thr:0.1f} pML={self.ml_p:0.2f} DOA={fmt_sector(self.az_deg,self.sector_half)}")
        else:
            self.state = "DETECCIÓN" if (now-self.last_det_ts)<DETECTION_HOLD_SEC else "VIGILANDO"

    def _poll(self):
        try:
            while True:
                msg=self.msg_q.get_nowait()
                kind=msg[0]
                if kind=="AUDIO_STATUS":
                    _,lvl,text=msg; self._append(text if lvl=="OK" else "[ERR] "+text)
                elif kind=="AUDIO_FRAME":
                    _,frame,sr=msg; self._process_audio_frame(frame, sr)
                elif kind=="STATUS":
                    _,lvl,text=msg; self._append(f"[DOA] {lvl}: {text}")
                elif kind=="HOTMAP":
                    _,hm,az_s,sector_s,conf=msg
                    self.hotmap=hm; self.az_deg=az_s; self.sector_half=sector_s; self.doa_conf=conf
        except queue.Empty:
            pass
        self._update_left_panel()
        self._draw_hotmap(self.hotmap)
        self._draw_compass(self.az_deg, self.sector_half, self.doa_conf)
        self.root.after(80, self._poll)

    def on_close(self):
        self.stop_event.set()
        try: self.root.destroy()
        except Exception: pass

def main():
    info("=== CIC ONEFILE (Umbral + DOA + ML) ===")
    info(f"[INFO] Root: {ROOT}")
    ensure_apt()
    ensure_user_dialout()
    ensure_venv_and_reexec()
    lazy_imports()
    import tkinter as tk
    root=tk.Tk()
    CICApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
