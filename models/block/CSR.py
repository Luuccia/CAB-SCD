import torch
import torch.nn as nn


class CSR(nn.Module):
    """
    Contrastive Semantic Reasoning Module
    """
    def __init__(self, in_dim=96):
        super(CSR, self).__init__()
        self.in_dim = in_dim
     
        self.query_conv1 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim //8, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim//8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=1)
        
        self.query_conv2 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim //8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim //8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=1)
        
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, feats1, feats2):
        m_batchsize, C, height, width = feats1[0].size()
        outputs1 = []
        outputs2 = []
        for x1, x2 in zip(feats1, feats2):
            q1 = self.query_conv1(x1).view(m_batchsize, -1, width*height).permute(0, 2, 1)
            k1 = self.key_conv1(x1).view(m_batchsize, -1, width*height)
            v1 = self.value_conv1(x1).view(m_batchsize, -1, width*height)
            
            q2 = self.query_conv2(x2).view(m_batchsize, -1, width*height).permute(0, 2, 1)
            k2 = self.key_conv2(x2).view(m_batchsize, -1, width*height)
            v2 = self.value_conv2(x2).view(m_batchsize, -1, width*height)
            
            energy1 = torch.bmm(q1, k2)
            attention1 = self.softmax(energy1)
            out1 = torch.bmm(v2, attention1.permute(0, 2, 1))
            out1 = out1.view(m_batchsize, C, height, width)
                    
            energy2 = torch.bmm(q2, k1)
            attention2 = self.softmax(energy2)
            out2 = torch.bmm(v1, attention2.permute(0, 2, 1))
            out2 = out2.view(m_batchsize, C, height, width)
            
            out1 = x1 + self.gamma1*out1
            out2 = x2 + self.gamma2*out2
            outputs1.append(out1)
            outputs2.append(out2)
        
        return outputs1, outputs2
