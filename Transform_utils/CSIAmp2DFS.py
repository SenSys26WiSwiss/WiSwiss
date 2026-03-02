import torch
import torch.nn.functional as F


def csi2dfs_flexible_batched(csi_data, ori_packet_cnt, ori_samp_rate=1000):
    """
    A flexible, batched implementation that adapts to different input lengths.
    It uses a consistent n_fft parameter to prevent shape mismatch errors.

    Args:
        csi_data (torch.Tensor): Input CSI data, shape [B, Subcarriers, Packets].
        ori_packet_cnt (int): The original number of packets for scaling samp_rate.
        ori_samp_rate (int): The original sampling rate.
    """
    if not isinstance(csi_data, torch.Tensor):
        csi_data = torch.tensor(csi_data, dtype=torch.float32)

    device = csi_data.device
    batch_size = csi_data.shape[0]
    num_subcarriers_total = csi_data.shape[1]

    # --- 0. Permute input for time-series processing ---
    csi_data = csi_data.permute(0, 2, 1)
    
    # --- 1. Parameter and Reference Signal Setup ---
    num_packets = csi_data.shape[1]
    samp_rate = int(num_packets / (ori_packet_cnt / ori_samp_rate))
    samp_rate = (samp_rate // 2) * 2
    ant_cnt = 3
    subcarrier_cnt = num_subcarriers_total // ant_cnt
    
    csi_mean = torch.mean(csi_data, dim=1)
    csi_var = torch.var(csi_data, dim=1).sqrt()
    ratio = csi_mean / (csi_var + 1e-10)
    ratio_reshaped = ratio.view(batch_size, ant_cnt, subcarrier_cnt)
    idx = torch.argmax(torch.mean(ratio_reshaped, dim=2), dim=1)
    csi_antennas = csi_data.view(batch_size, num_packets, ant_cnt, subcarrier_cnt)
    one_hot_idx = F.one_hot(idx, num_classes=ant_cnt).view(batch_size, 1, ant_cnt, 1)
    ref_signal = (csi_antennas * one_hot_idx).sum(dim=2)
    csi_data_ref = ref_signal.repeat(1, 1, ant_cnt)
    alpha = torch.min(torch.where(csi_data > 0, csi_data, csi_data.max()), dim=1).values
    csi_data_adj = F.relu(csi_data - alpha.unsqueeze(1))
    beta = (1000 * alpha.sum(dim=1) / (ant_cnt * subcarrier_cnt)).view(batch_size, 1, 1)
    csi_data_ref_adj = csi_data_ref + beta
    conj_mult_full = csi_data_adj * csi_data_ref_adj
    ant_blocks = conj_mult_full.view(batch_size, num_packets, ant_cnt, subcarrier_cnt)
    mask_to_keep = (1 - F.one_hot(idx, num_classes=ant_cnt)).bool()
    conj_mult = ant_blocks.permute(2, 0, 1, 3)[mask_to_keep.T].view(ant_cnt-1, batch_size, num_packets, subcarrier_cnt).permute(1,0,2,3).reshape(batch_size, num_packets, -1)

    # --- 2. Bandpass Filtering ---
    nyquist_freq = samp_rate / 2
    upper_bound = min(60, nyquist_freq)
    freqs = torch.fft.fftfreq(num_packets, d=1/samp_rate, device=device)
    bandpass_mask = (torch.abs(freqs) >= 2) & (torch.abs(freqs) <= upper_bound)
    conj_mult_fft = torch.fft.fft(conj_mult, dim=1)
    conj_mult_fft_filtered = conj_mult_fft * bandpass_mask.view(1, -1, 1)
    conj_mult_filtered = torch.fft.ifft(conj_mult_fft_filtered, dim=1).real

    # --- 3. Stable PCA via Power Iteration (Adapted for Batch) ---
    # Centering the data remains the same.
    data_centered = conj_mult_filtered - conj_mult_filtered.mean(dim=1, keepdim=True)
    
    # The covariance matrix calculation remains the same.
    covariance_matrix = torch.bmm(data_centered.transpose(1, 2), data_centered)

    # --- Power Iteration to find the first principal component (dominant eigenvector) ---
    # This replaces the unstable `torch.linalg.eigh`.
    batch_size, n_features, _ = covariance_matrix.shape
    # 1. Initialize a random vector for each item in the batch.
    v = torch.randn(batch_size, n_features, 1, device=covariance_matrix.device)
    
    # 2. Iteratively multiply by the covariance matrix and normalize.
    # A small number of iterations is usually sufficient for convergence.
    for _ in range(10): 
        v_prime = torch.bmm(covariance_matrix, v)
        norm = torch.linalg.norm(v_prime, dim=1, keepdim=True)
        v = v_prime / (norm + 1e-10) # Add epsilon for stability
        
    # The resulting vector 'v' is our principal component.
    principal_component = v.squeeze(-1)
    
    # Project the data onto the principal component.
    conj_mult_pca = torch.bmm(data_centered, principal_component.unsqueeze(-1)).squeeze(-1)
    
    # --- 4. STFT (MODIFIED) ---
    win_length = min(num_packets-2, samp_rate // 4)
    win_length += 1 if win_length % 2 == 0 else 0

    # 📌 FIX: Set center=False to prevent padding error with short signals.
    fft_size = min(samp_rate, num_packets-1)
    # print(samp_rate, win_length, fft_size)
    stft_result = torch.stft(conj_mult_pca, n_fft=fft_size, hop_length=1, 
                               win_length=win_length, window=torch.hann_window(win_length, device=device),
                               center=True, return_complex=True)
    
    power_spectrum = stft_result.abs().pow(2)
    
    # --- 5. Final Processing ---
    freq_bins_stft = torch.fft.rfftfreq(fft_size, d=1/samp_rate, device=device)
    # print(freq_bins_stft, upper_bound)
    freq_lpf_sele = torch.abs(freq_bins_stft) <= upper_bound
    freq_time_prof = power_spectrum[:, freq_lpf_sele, :]
    norm_sum = torch.sum(freq_time_prof, dim=1, keepdim=True)
    freq_time_prof_normalized = freq_time_prof / (norm_sum + 1e-10)
    doppler_spectrum = torch.fft.fftshift(freq_time_prof_normalized, dim=1)
    
    return doppler_spectrum