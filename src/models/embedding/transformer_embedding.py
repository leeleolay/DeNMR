"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
from torch import nn
import torch
from .positional_encoding import PositionalEncoding
from .token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model).to(device)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)
        self.device = device

    def forward(self, x):

        x = x.to(self.device)  # 将 x 移动到设备
        tok_emb = self.tok_emb(x)  # 这里是嵌入层计算
        pos_emb = self.pos_emb(x)  # 这里是位置嵌入计算

        # 这里不需要再次移动到设备，因为 tok_emb 和 pos_emb 已经在设备上了
        return self.drop_out(tok_emb + pos_emb)

        # tok_emb = self.tok_emb(x).to(self.device)
        # pos_emb = self.pos_emb(x).to(self.device)