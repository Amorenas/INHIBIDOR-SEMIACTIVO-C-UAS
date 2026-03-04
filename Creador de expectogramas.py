import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

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

y, sr = librosa.load(ARCHIVO_AUDIO, sr=FS)

S = librosa.stft(y)

S_db = librosa.amplitude_to_db(np.abs(S))

# =========================
# CREAR ESPECTROGRAMA
# =========================

plt.figure(figsize=(12,6))

librosa.display.specshow(
    S_db,
    sr=sr,
    x_axis="time",
    y_axis="hz",
    cmap="magma"
)

plt.ylim(0,10000)

plt.colorbar(label="dB")

plt.title("Espectrograma acústico UAV")

plt.xlabel("Tiempo (s)")
plt.ylabel("Frecuencia (Hz)")

plt.tight_layout()

plt.savefig(ARCHIVO_IMAGEN, dpi=300)

print("\nImagen guardada como:", ARCHIVO_IMAGEN)

plt.show()
