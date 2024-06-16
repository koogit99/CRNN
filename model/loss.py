import torch
from torch.nn.utils.rnn import pad_sequence

def mse_loss_for_variable_length_data():
    def loss_function(target, ipt, n_frames_list, device):
        """
        Calculate the MSE loss for variable length dataset.

        ipt: [B, F, T]
        target: [B, F, T]
        """
        if target.shape[0] == 1:
            print("Target shape[0] == 1 이라 그냥 mse_loss 출력")
            return torch.nn.functional.mse_loss(target, ipt)
            
        ## 이 사이의 코드를 추가해주세요
        ipt = ipt.permute(0,2,1)
        target = target.permute(0,2,1) # 이렇게 하면 차원축이 다시 B x F x T로 모두 맞춰집니다
        ## 여기까지
        print("clean shape in loss.py : ", ipt.shape)
        print("enhanced shape in loss.py : ", target.shape)
        E = 1e-8
        with torch.no_grad():
            masks = []
            for n_frames in n_frames_list:
                masks.append(torch.ones(n_frames, target.size(1), dtype=torch.float32))  # the shape is (T_real, F)
            
            #
            print(f"ipt shape : {ipt.shape}")
            #
            binary_mask = pad_sequence(masks, batch_first=True).to(device).permute(0, 2, 1)  # ([T1, F], [T2, F]) => [B, T, F] => [B, F, T]
            #
            print(f"binary_mask shape : {binary_mask.shape}")
            #
            
        masked_ipt = ipt * binary_mask  # [B, F, T]
        masked_target = target * binary_mask
        return ((masked_ipt - masked_target) ** 2).sum() / (binary_mask.sum() + E)  # 不算 pad 部分的贡献，仅计算有效值

    return loss_function