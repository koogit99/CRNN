import torch.nn as nn
import torch.nn.functional as F
import torch

class CRNN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    def __init__(self):
        super(CRNN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 3), stride=(1, 2))
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 3), stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 3), stride=(1, 2))
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 3), stride=(1, 2))
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 3), stride=(1, 2))
        self.bn5 = nn.BatchNorm2d(num_features=256)

        # LSTM
        self.LSTM1 = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True)

        # Decoder
        self.convT1 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(2, 3), stride=(1, 2), padding=(1,0))
        self.bnT1 = nn.BatchNorm2d(num_features=128)
        self.convT2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(2, 3), stride=(1, 2), padding=(1,0))
        self.bnT2 = nn.BatchNorm2d(num_features=64)
        self.convT3 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(2, 3), stride=(1, 2), padding=(1,0))
        self.bnT3 = nn.BatchNorm2d(num_features=32)
        # output_padding为1，不然算出来是79
        self.convT4 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0), output_padding=(0,1))
        self.bnT4 = nn.BatchNorm2d(num_features=16)
        self.convT5 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(2, 3), stride=(1, 2), padding=(1,0))
        self.bnT5 = nn.BatchNorm2d(num_features=1)

    def forward(self, x):
        # conv
        # (B, in_c, T, F)
        # 시간축에 Zero-padding 추가함 #Z 붙여둔곳 T = 410을 보존하기 위함
        p1d = (0,0,1,0) # Zero-padding for time conservation #Z
        print(f"input : {x.shape}")
        if (x.ndim ==3): # validation 에만 한정
            x.unsqueeze_(1)
        else :
            pass # 
        print(f"unsqueezed input : {x.shape}")
        x_en = F.pad(x, p1d)
        x1 = F.elu(self.bn1(self.conv1(x_en)))
        print(f"x1 : {x1.shape}")
        
        x1_en = F.pad(x1, p1d)
        x2 = F.elu(self.bn2(self.conv2(x1_en)))
        print(f"x2 : {x2.shape}")
        
        x2_en = F.pad(x2, p1d)
        x3 = F.elu(self.bn3(self.conv3(x2_en)))
        print(f"x3 : {x3.shape}")
        
        x3_en = F.pad(x3, p1d)
        x4 = F.elu(self.bn4(self.conv4(x3_en)))
        print(f"x4 : {x4.shape}")
        
        x4_en = F.pad(x4, p1d)
        x5 = F.elu(self.bn5(self.conv5(x4_en)))
        print(f"x5 : {x5.shape}")
        # reshape
        out5 = x5.permute(0, 2, 1, 3)
        print(f"permuted x5 : {out5.shape}")
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        print(f"reshaped x5 : {out5.shape}")
        # lstm

        lstm, (hn, cn) = self.LSTM1(out5)
        # reshape
        output = lstm.reshape(lstm.size()[0], lstm.size()[1], 256, -1)
        output = output.permute(0, 2, 1, 3)
        print(f"output : {output.shape}")
        # ConvTrans
        res = torch.cat((output, x5), 1)
        print(f"res : {res.shape}")
        res_en = F.pad(res, p1d)
        res1 = F.elu(self.bnT1(self.convT1(res_en)))
        print(f"res1 : {res1.shape}") #문제의 라인
        res1 = torch.cat((res1, x4), 1)
        print(f"res1 : {res1.shape}") #문제의 라인
        res1_en = F.pad(res1, p1d)
        res2 = F.elu(self.bnT2(self.convT2(res1_en)))
        
        print(f"res2 shape : {res2.shape}")
        res2 = torch.cat((res2, x3), 1)
        res2_en = F.pad(res2, p1d)
        res3 = F.elu(self.bnT3(self.convT3(res2_en)))
        res3 = torch.cat((res3, x2), 1)
        print(f"res3 shape : {res3.shape}")
        
        res3_en = F.pad(res3, p1d)
        res4 = F.elu(self.bnT4(self.convT4(res3_en)))
        res4 = torch.cat((res4, x1), 1)
        print(f"res4 shape : {res4.shape}")
        # (B, o_c, T. F)
        
        res4_en = F.pad(res4, p1d)
        res5 = F.relu(self.bnT5(self.convT5(res4_en)))
        return res5.squeeze()