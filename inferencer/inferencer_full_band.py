import librosa
import torch



def full_band_no_truncation(model, device, inference_args, noisy):
    """
    extract full_band spectra for inference, without truncation.
    """
    n_fft = inference_args["n_fft"]
    hop_length = inference_args["hop_length"]
    win_length = inference_args["win_length"]

    noisy_mag, noisy_phase = librosa.magphase(librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    noisy_mag = torch.tensor(noisy_mag, device=device)[None, None, :, :]  # [F, T] => [1, 1, F, T]
    print(f"input: {noisy_mag.shape}")
    noisy_mag = noisy_mag.permute(0,1,3,2) # 1, 1, f, t -> 1, 1, t, f
    print(f"input : {noisy_mag.shape}")

    enhanced_mag = model(noisy_mag)  # [1, 1, F, T] => [1, 1, F, T] 
    enhanced_mag = enhanced_mag.squeeze(0).squeeze(0).detach().cpu().numpy()  # [1, 1, F, T] => [F, T]
    # enhanced_mag = enhanced_mag.T
    noisy_phase_tensor = torch.tensor(noisy_phase, device=device).permute(1,0).numpy()
    # enhanced = librosa.istft(enhanced_mag * noisy_phase, hop_length=hop_length, win_length=win_length, length=len(noisy))
    enhanced = librosa.istft(enhanced_mag * noisy_phase_tensor.T, hop_length=hop_length, win_length=win_length, length=len(noisy))
    #noisy_phase 퍼뮤트 원랜없었음
    
    assert len(noisy) == len(enhanced)

    return noisy, enhanced
