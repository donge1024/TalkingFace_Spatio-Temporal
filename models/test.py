import torch
from torch import nn
from torch.nn import functional as F
import math

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class Wav2Lip(nn.Module):
    def __init__(self):
        super(Wav2Lip, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)), # 96,96

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 48,48
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # 24,24
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 12,12
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),       # 6,6
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),     # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),),
            # nn.Sequential(Conv2d(1024, 512, kernel_size=1, stride=1, padding=0), ), # 加了r

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0), # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),), # 12, 12

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # 24, 24

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 48, 48

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),),]) # 96,96

        # self.face_decoder_blocks = nn.ModuleList([
        #     # nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),),
        #     nn.Sequential(Conv2d(1536, 1024, kernel_size=1, stride=1, padding=0), ),  # 加了r
        #
        #     nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),  # 3,3
        #                   Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),
        #
        #     nn.Sequential(Conv2dTranspose(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                   Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        #                   Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),  # 6, 6
        #
        #     nn.Sequential(Conv2dTranspose(512, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                   Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
        #                   Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True), ),  # 12, 12
        #
        #     nn.Sequential(Conv2dTranspose(384, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                   Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        #                   Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), ),  # 24, 24
        #
        #     nn.Sequential(Conv2dTranspose(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                   Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        #                   Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), ),  # 48, 48
        #
        #     nn.Sequential(Conv2dTranspose(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                   Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
        #                   Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), ), ])  # 96,96

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())

        # self.output_block = nn.Sequential(Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        #                                   nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
        #                                   nn.Sigmoid())

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences = (B, T=5, 1, 80, 16)
        # face_sequences = (B, 6, T=5, 96, 96) 6表示window和wrong_window的彩色特征图
        B = audio_sequences.size(0)

        # print("audio%s"%str(audio_sequences.size()))
        # print("face%s"%str(face_sequences.size()))

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0) # 将时间序列的维度消掉，放到batchsize的维度
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0) # 将时间序列的维度消掉，放到batchsize的维度
            # print("audio%s"%str(audio_sequences.size())) # BxT 1 80 16
            # print("face%s"%str(face_sequences.size())) # BxT 6 96 96

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        # # r加在这里
        # r = torch.randn(audio_embedding.size(0), 512, 1, 1 ).to(audio_embedding.device)
        # x = audio_embedding
        # # 连接x和r，加一层全连接层
        # x = torch.cat([x, r],dim=1) # b, 1024, 1, 1

        # 不加r
        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()

        # # 用音频去解码图像
        # # r加在这里
        # r = torch.randn(audio_embedding.size(0), 512, 1, 1).to(audio_embedding.device)
        # x = audio_embedding
        # # 连接x和audio_embedding, r，加一层全连接层
        # x = torch.cat([x, audio_embedding,r], dim=1)  # b, 1536, 1, 1
        # for f in self.face_decoder_blocks:
        #     x = f(x)

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x

        return outputs

class Wav2Lip_disc_qual(nn.Module):
    def __init__(self):
        super(Wav2Lip_disc_qual, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

        self.person_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(3, 16, kernel_size=7, stride=1, padding=3)),  # 96,96

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 48,48
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 24,24
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 12,12
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 6,6
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1, 1
                          Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)), # 48,96

            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2), # 48,48
            nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),    # 24,24
            nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),   # 12,12
            nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),       # 6,6
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),     # 3,3
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
            nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])


        self.full_pred = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=1, stride=1, padding=0),  # 输入层与第一隐层结点数设置，全连接结构
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01, inplace=True)  # 第一隐层激活函数采用sigmoid
            # nn.ReLU()  # 第一隐层激活函数采用sigmoid
        )

        # self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        # self.binary_pred = nn.Sequential(nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid()) # 合并后为1024
        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid()) # audio, person, 合并后为1536

        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2)//2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, audio_sequences, person_sequences, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        person_sequences = self.to_2d(person_sequences)

        audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))],
                                    dim=0)  # 将时间序列的维度消掉，放到batchsize的维度
        audio_embedding = self.audio_encoder(audio_sequences)  # B, 512, 1, 1

        person_embedding = person_sequences
        for f in self.person_encoder_blocks:
            person_embedding = f(person_embedding)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)


        # 将false_feats和audio_embedding和并
        x = torch.cat([false_feats, person_embedding, audio_embedding], dim=1)

        # 加入一层全连接层
        x = self.full_pred(x)

        # false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1),
        #                                 torch.ones((len(false_feats), 1)).cuda())
        # false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1),
        #                                          torch.ones((len(false_feats), 1)).to(next(self.parameters()).device))
        false_pred_loss = F.binary_cross_entropy(self.binary_pred(x).view(len(x), -1),
                                                 torch.ones((len(x), 1)).to(x.device))

        return false_pred_loss

    def forward(self, audio_sequences, person_sequences, face_sequences):
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)

        person_sequences = self.to_2d(person_sequences)

        audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))],
                                    dim=0)  # 将时间序列的维度消掉，放到batchsize的维度
        audio_embedding = self.audio_encoder(audio_sequences)  # B, 512, 1, 1

        person_embedding = person_sequences
        for f in self.person_encoder_blocks:
            person_embedding = f(person_embedding)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        # 将x和audio_embedding和并
        x = torch.cat([x, person_embedding, audio_embedding], dim=1)

        # 加入一层全连接层
        x = self.full_pred(x)

        return self.binary_pred(x).view(len(x), -1)

if __name__ == "__main__":
    print("?")