import numpy as np
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torch

# Force single-threaded execution to prevent segfaults in loops
torch.set_num_threads(1)



# Global cache for the model
_MODEL = None

def separate(file, target_source=None, gpu=None):
    """
    Run Demucs source separation model on WAV files to isolate stems.
    
    Parameters
    ----------
    file :  str
            Path to the mixture WAV file to separate.
            
    target_source : str
            target source to separate. (optional, default = Drums)
    
    gpu : torch.device
          GPU device to use for processing. If None, CPU will be used.
          
    Returns
    -------
    target_source : np.ndarray
            Isolated vocal signal, converted to mono.
    
    accompaniments : np.ndarray
                    Isolated and combined signal, converted to mono.
                    
    sr : int
         Sample rate
    """
    global _MODEL
    
    # Load audio file (supports WAV, MP3, FLAC, AIFF, OGG, M4A)
    supported_ext = ('.wav', '.mp3', '.flac', '.aiff', '.ogg', '.m4a')
    if file.lower().endswith(supported_ext):
        waveform, sr = torchaudio.load(file)
    else:
        raise ValueError(f"Unsupported audio format: {file}. Supported: {supported_ext}")
    
    if _MODEL is None:
        # print("Loading Demucs model (this happens only once)...")
        if gpu: 
            _MODEL = get_model(name="htdemucs").cuda()
        else:
            _MODEL = get_model(name="htdemucs").cpu()
        _MODEL.eval()
        
    model = _MODEL

    model.eval()

    # Apply separation
    waveform = waveform.unsqueeze(0)
    sources = apply_model(model, waveform, split=True, overlap=0.25, progress=True)[0]

    # Map sources to names
    sources_dict = dict(zip(model.sources, sources))

    # Divide source and trarget source
    SOURCE = ['vocals', 'drums', 'bass', 'other']
    accompaniment = [s for s in SOURCE if s not in target_source]
    
    # Get each sources
    accompaniments = sources_dict[accompaniment[0]] + sources_dict[accompaniment[1]] + sources_dict[accompaniment[2]]
    target = sources_dict[target_source]
    
    # Convert from Tensor to Numpy
    target = target.detach().cpu().numpy().T
    accompaniments = accompaniments.detach().cpu().numpy().T
    
    # Convert to mono
    if len(target.shape) != 1:
        target = np.mean(target, axis=1)
    
    if len(accompaniments.shape) != 1:
        accompaniments = np.mean(accompaniments, axis=1)
    
    return target, accompaniments, sr
