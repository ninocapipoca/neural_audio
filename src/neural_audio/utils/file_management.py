import struct
import numpy as np

def load_wav(file_path):
    """
    Load a WAV audio file and convert its raw data chunk into a normalized 
    float32 NumPy array.

    This function parses the standard 44-byte RIFF/WAVE header to extract 
    essential audio metadata (sampling rate, bit depth) before reading the 
    underlying binary audio payload.

    :param file_path: The path to the WAV file to be loaded.
    :type file_path: str
    
    :raise ValueError: If the file is not a valid RIFF/WAVE file, or if the 
        audio bit depth (sample width) is unsupported (e.g., 24-bit).
    :raise FileNotFoundError: If the file at `file_path` does not exist.

    :return: A tuple containing the waveform array and the sampling rate.
        - **waveform** (*np.ndarray*): A 1D float32 array containing the 
          audio samples.
        - **sampling_rate** (*int*): The sampling rate of the audio in Hz.
    :rtype: tuple(np.ndarray, int)

    .. note::
       **Assumptions:**
       
       - The WAV file follows the standard 44-byte header format. Extra 
         subchunks (like metadata or CUE points) preceding the data chunk 
         may cause incorrect offsets.
       - The audio data is PCM encoded (uncompressed).

       **Guarantees:**
       
       - The returned `waveform` array elements are always of type 
         :class:`np.float32`.
       - Audio amplitudes are strictly normalized to fall within the range 
         [-1.0, 1.0] for signed formats, and correctly centered for unsigned formats.
    """
    with open(file_path, 'rb') as audio_file:
        # Read the header to get audio file information
        header = audio_file.read(44)  # In WAV files, first 44 bytes are reserved for the header
        
        if header[:4] != b'RIFF' or header[8:12] != b'WAVE' or header[12:16] != b'fmt ':
            raise ValueError("Invalid WAV file structure or unsupported format.")
            
        # Extract relevant information from the header
        header_chunk_id = struct.unpack('4s', header[0:4])[0]
        header_chunk_size = struct.unpack('<I', header[4:8])[0]
        header_chunk_format = struct.unpack('4s', header[8:12])[0]
        format_chunk_id = struct.unpack('4s', header[12:16])[0]
        format_chunk_size = struct.unpack('<I', header[16:20])[0]
        format_code = struct.unpack('<H', header[20:22])[0]
        channels = struct.unpack('<H', header[22:24])[0]
        sampling_rate = struct.unpack('<I', header[24:28])[0]
        byte_rate = struct.unpack('<I', header[28:32])[0]
        block_align = struct.unpack('<H', header[32:34])[0]
        sample_width = struct.unpack('<H', header[34:36])[0]
        data_chunk_id = struct.unpack('4s', header[36:40])[0]
        data_chunk_size = struct.unpack('<I', header[40:44])[0]
        
        # Read the data from the file
        audio_file.seek(44)
        data = audio_file.read(data_chunk_size)
    
    # Map the bit depth (sample_width) to the appropriate NumPy data type
    if sample_width == 16:
        audio_dtype = np.int16
    elif sample_width == 32:
        audio_dtype = np.int32
    elif sample_width == 8:
        audio_dtype = np.uint8
    else:
        raise ValueError(
            f"Unsupported sample width: {sample_width}-bit. "
            f"Native NumPy reading only supports 8, 16, or 32-bit PCM."
        )

    # Convert the raw binary data into a NumPy array matching the file's bit depth
    waveform = np.frombuffer(data, dtype=audio_dtype)
    
    # Convert to float32 and normalize amplitude bounds to [-1.0, 1.0]
    waveform = waveform.astype(np.float32)
    
    if sample_width == 16:
        waveform /= 32768.0        # Normalized by max value of signed 16-bit int
    elif sample_width == 32:
        waveform /= 2147483648.0   # Normalized by max value of signed 32-bit int
    elif sample_width == 8:
        waveform = (waveform - 128.0) / 128.0  # 8-bit WAV is unsigned; center and scale
    
    return waveform, sampling_rate