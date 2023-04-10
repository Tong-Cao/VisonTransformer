import torch
from torch import nn
import Leaves_Data


class ViT_Model(nn.Module):
    def __init__(self, image_size, patch_size, embed_size, num_classes,
                  num_layers, heads, mlp_dim, dropout=0., emb_dropout=0.):
        super().__init__()
        """
        image_size: 输入图片的尺寸
        patch_size: 切分图片的patch的尺寸
        embed_size: 词嵌入大小 vit中使用embed_size个卷积核将图片扩展成embed_size个通道并将这些通道值作为token的特征输入
        num_classes: 分类的类别数
        num_layers: encoder的层数
        heads: 多头注意力中的头数
        mlp_dim: feedforward隐藏层大小
        dropout: transformer中丢弃率
        emb_dropout: 词嵌入层的丢弃率
        """
        # 检查图像尺寸是否能被patch_size整除
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # 将图片(image_size,image_size)划分为num_patches个patches  每一个patch相当于一个token作为输入
        num_patches = (image_size // patch_size) ** 2
        # 保存参数
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 将图片卷积成(image_size //patch_size,image_size //patch_size)每一个点相当于是一个token
        # 一共用embed_size个卷积核输出embed_size个通道 这些通道值作为token的特征输入
        # (batch_size,3,image_size,image_size)
        # ->(batch_size,embed_size,image_size //patch_size,image_size //patch_size)
        self.patch_embeddings = nn.Conv2d(in_channels=3,
                                       out_channels=embed_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # 可学习的分类token
        self.classifer_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        # 设置可学习的位置编码信息，(1,196+1,786)
        self.position_embeddings = nn.Parameter(torch.zeros(1,
                                                            num_patches+1,
                                                            embed_size))
        # 词嵌入层dropout
        self.dropout = nn.Dropout(emb_dropout)
        # 使用transformer的encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads,
                                                        dropout=dropout, batch_first=True,
                                                        dim_feedforward=mlp_dim)

        self.transformer_encoder = nn.TransformerEncoder(
                                self.encoder_layer, num_layers=num_layers)

        # 使用分类token的特征作送入mlp_head进行分类
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size), nn.Linear(embed_size, num_classes))

    def forward(self, img):
        """
        img: (batch_size,3,image_size,image_size)
        """
        # 获取batch_size
        batch_size = img.shape[0]
        # 设置分类token：(batch_size,1,embed_size)
        cls_tokens = self.classifer_token.expand(batch_size, -1, -1)

        # 将图片切成patches (batch_size,embed_size,image_size //patch_size,image_size //patch_size)
        img = self.patch_embeddings(img)
        # 将维度2，3展开 (batch_size,embed_size,num_patches)  num_patches = (image_size // patch_size) ** 2
        img = img.flatten(2, 3)
        # 调整数据结构 (batch_size,num_patches,embed_size)
        img = img.transpose(-1, -2)
        # 将分类token和patches拼接 (batch_size,num_patches+1,embed_size)
        img = torch.cat((cls_tokens, img), dim=1)
        # 将图片的位置信息加入到图片中 (batch_size,num_patches+1,embed_size)
        img = img + self.position_embeddings
        # 词嵌入层dropout
        img = self.dropout(img)

        # transformer encoder (batch_size,num_patches+1,embed_size)
        img = self.transformer_encoder(img)

        # 将分类token的特征提取出来 (batch_size,embed_size)
        img = img[:, 0]

        # (batch_size,embed_size) -> (batch_size,num_classes)
        img = self.mlp_head(img)

        return img # (batch_size,num_classes)

'''模型测试'''
# x = torch.zeros(2,3, 224, 224)
# vib = ViT_Model(image_size=224, patch_size=16, embed_size=768, num_classes=10,
#                 num_layers=2, heads=2, mlp_dim=3072, dropout=0., emb_dropout=0.)
# y = vib(x)
# print(y.shape) # torch.Size([2, 10])

if __name__ == '__main__':

    # 1) 初始化
    torch.distributed.init_process_group(backend="nccl")

    # 2） 配置每个进程的gpu
    local_rank = torch.distributed.get_rank()
    print('local_rank',local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    batch_size, num_epochs = 2 , 120

    lr = 0.0001

    loss = torch.nn.CrossEntropyLoss()

    train_iter, test_iter,num_classes = Leaves_Data.get_data_iter(batch_size)



    net = ViT_Model(image_size=224, patch_size=16, embed_size=768, num_classes=num_classes,
                 num_layers=4, heads=12, mlp_dim=3072, dropout=0.5, emb_dropout=0.5)

    # 4) 封装之前要把模型移到对应的gpu
    net.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 5) 封装
        net = torch.nn.parallel.DistributedDataParallel(net,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank,
                                                        find_unused_parameters=True)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.001)


    #Leaves_Data.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

    








