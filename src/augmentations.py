import numpy as np
import librosa

def noise(data, rng=np.random):
    """Adds random background static scaled to the audio's maximum volume."""
    noise_amp = 0.035 * rng.uniform() * np.amax(data)
    data = data + noise_amp * rng.normal(size=data.shape[0])
    return data

def stretch(data, rate1=0.8):
    """Alters the speed of the speech without affecting its natural pitch."""
    return librosa.effects.time_stretch(y=data, rate=rate1)

def shift(data, rng=np.random):
    """Rolls the audio signal forward or backward along the time axis."""
    shift_range = int(rng.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, n_steps=0.7):
    """Raises or lowers the tonal frequency of the voice."""
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=n_steps)
