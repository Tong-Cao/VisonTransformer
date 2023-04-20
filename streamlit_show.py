import torch
from torch import nn
import pandas as pd
from torchvision import transforms
import streamlit as st
from PIL import Image

# 模型
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

# 图片预处理
def image_preprocess(img):
    # 转换为PIL格式
    img = Image.open(img)
    # 转换为tensor
    img = transforms.ToTensor()(img)
    # resize大小为224
    img = transforms.Resize(224)(img)
    # 添加batch_size维度
    img = img.unsqueeze(0)
    return img

# 模型推理
def inference(model, img):
    # 设置为推理模式
    model.eval()
    # 读取class_to_num字典
    class_to_num = pd.read_csv('class_to_num.csv')
    # class_name = class_to_num.iloc[0,0]  # 切片读取 .iloc[i,j] i行j列  
    # print('名称',class_name)
    with torch.no_grad():
        logits = model(img)
        pred = torch.argmax(logits, dim=1)
        # 将预测结果转换为类别名称
        pred = class_to_num.iloc[pred,0].values[0]
        return pred



# construct UI layout
st.title("ViT demo")  # title

st.write(
    """使用ViT模型对树叶进行分类"""
)  # description and instructions

input_image = st.file_uploader("insert image")  # image upload widget

if st.button("Get Inference"):

    if input_image:

        # 展示原图片
        st.image(input_image, use_column_width=True)
        # 图片预处理
        input_image = image_preprocess(input_image)
        
        # 加载模型
        PATH = 'myvit.pth'
        model = ViT_Model(image_size=224, patch_size=16, embed_size=768, num_classes=176,
                        num_layers=2, heads=12, mlp_dim=3072, dropout=0.5, emb_dropout=0.5)
        model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))

        # 推理
        pred = inference(model, input_image)

        # 展示预测结果 并加粗放在正中间
        st.markdown(f"**Predicted class: {pred}**", unsafe_allow_html=True)
        

    else:
        # handle case with no image
        st.write("Insert an image!")
