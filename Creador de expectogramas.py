import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load("avata2.wav", sr=16000)

S = librosa.stft(y)
S_db = librosa.amplitude_to_db(abs(S))

plt.figure(figsize=(10,5))
librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz")
plt.ylim(0,10000)
plt.colorbar(label="dB")
plt.title("Espectrograma acústico DJI Avata 2")
plt.show()
