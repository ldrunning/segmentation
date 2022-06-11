import os

import cv2
import numpy as np
import torch
from gdn import GDN, BASELINE
from collections import OrderedDict
from tqdm import tqdm

net_a = GDN()
net_b = BASELINE()  # 别忘记传递必要的参数
net_b_dict = net_b.state_dict()
state_dict = torch.load("train_epoch_1200_final.pth")  # 加载预先训练好net-a的.pth文件
new_state_dict = OrderedDict()  # 不是必要的【from collections import OrderedDict】

new_state_dict = {k: v for k, v in state_dict.items() if k in net_b_dict}  # 删除net-b不需要的键
net_b_dict.update(new_state_dict)  # 更新参数
net_b.load_state_dict(net_b_dict)  # 加载参数
save_path = "gdn_solo.pth"
save_path2 = "gdn_solo2.pt"
for name, para in net_b.named_parameters():
    print(name, torch.max(para))
torch.save(net_b.state_dict(), save_path)
torch.save(net_b, save_path2)
