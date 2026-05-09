import tensorflow as tf
import numpy as np

class JetsonIQProcessor:
    def __init__(self, chunk_size=1024):
        self.chunk_size = chunk_size

    @tf.function(jit_compile=True)
    def preprocess_batch(self, iq_batch):
        """
        Optimized for Jetson GPU via XLA. 
        Aligned 1:1 with BansheeSequence training logic.
        """
        # 1. Convert to Complex
        z = tf.complex(iq_batch[..., 0], iq_batch[..., 1])

        # --- CHANNEL 1: MAGNITUDE (Red) ---
        # frame_step=16 matches noverlap=112 (128-112=16)
        stfts = tf.signal.stft(z, frame_length=128, frame_step=16, fft_length=128)
        
        # CRITICAL FIX 1: Shift the Frequency axis (index 2), not Time (index 1)
        stfts = tf.signal.fftshift(stfts, axes=2)
        
        mag = tf.abs(stfts)
        # dB Calculation: 10 * log10(mag^2 + 1e-10)
        mag_db = 10.0 * (tf.math.log(tf.square(mag) + 1e-10) / tf.math.log(10.0))
        
        r = (tf.clip_by_value(mag_db, -35.0, 5.0) + 35.0) / 40.0
        
        # CRITICAL FIX 2: Transpose to [Batch, Freq, Time] to match SciPy/OpenCV orientation
        r = tf.transpose(r, perm=[0, 2, 1]) 
        r = tf.expand_dims(r, axis=-1)
        r = tf.image.resize(r, [224, 224], method='nearest')

        # --- CHANNEL 2: PHASE DIFFERENCE (Green) ---
        z_curr = z[:, 1:]
        z_prev = z[:, :-1]
        phase_diff = tf.math.angle(z_curr * tf.math.conj(z_prev))
        
        # Pad 1023 -> 1024
        paddings = [[0, 0], [0, 1]] 
        phase_padded = tf.pad(phase_diff, paddings)
        
        # Reshape to 32x32 structure
        g = tf.reshape(phase_padded, [-1, 32, 32, 1])
        g = (g + np.pi) / (2.0 * np.pi)
        g = tf.image.resize(g, [224, 224], method='nearest')

        # --- CHANNEL 3: INTERACTION (Blue) ---
        b = r * g

        # --- ASSEMBLE AND FINALIZE ---
        rgb = tf.concat([r, g, b], axis=-1)
        
        # This replaces MobileNet's preprocess_input(x * 255)
        # Mapping [0, 1] -> [-1, 1]
        final_input = (rgb * 2.0) - 1.0
        
        return final_input

def process_bladerf_buffer(raw_iq_buffer, processor):
    """
    raw_iq_buffer: (N, 2) int16
    """
    chunk_size = 1024
    num_chunks = len(raw_iq_buffer) // chunk_size
    
    # Vectorized chunking
    chunks = raw_iq_buffer[:num_chunks * chunk_size].reshape(-1, chunk_size, 2)
    
    # Upload to GPU and normalize CS16 to +/- 1.0
    chunks_tf = tf.cast(chunks, dtype=tf.float32) / 32768.0
    
    return processor.preprocess_batch(chunks_tf)