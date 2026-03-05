import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import stft

# =========================
# CONFIGURACION
# =========================

DURACION = 10        # segundos de grabacion
FS = 16000           # frecuencia de muestreo
ARCHIVO_AUDIO = "grabacion_dron.wav"
ARCHIVO_IMAGEN = "espectrograma_dron.png"

print("\nGrabando audio durante", DURACION, "segundos...")
print("Haz volar el dron o reproduce el sonido ahora\n")

# =========================
# GRABAR AUDIO
# =========================

audio = sd.rec(int(DURACION * FS), samplerate=FS, channels=1)
sd.wait()

print("Grabacion finalizada")

# guardar audio
sf.write(ARCHIVO_AUDIO, audio, FS)

# =========================
# ANALISIS ESPECTRAL
# =========================

# Cargar audio con soundfile (sin librosa/numba)
y, sr = sf.read(ARCHIVO_AUDIO)
if y.ndim > 1:
    y = y[:, 0]  # convertir a mono si es necesario

# Calcular STFT con scipy
frecuencias, tiempos, Zxx = stft(y, fs=sr, nperseg=2048)

# Convertir a dB
S_db = 20 * np.log10(np.abs(Zxx) + 1e-10)

# =========================
# CREAR ESPECTROGRAMA
# =========================

plt.figure(figsize=(12, 6))

plt.pcolormesh(tiempos, frecuencias, S_db, shading='gouraud', cmap='magma')

plt.ylim(0, 10000)

plt.colorbar(label="dB")

plt.title("Espectrograma acústico UAV")
plt.xlabel("Tiempo (s)")
plt.ylabel("Frecuencia (Hz)")

plt.tight_layout()

plt.savefig(ARCHIVO_IMAGEN, dpi=300)

print("\nImagen guardada como:", ARCHIVO_IMAGEN)

plt.show()
