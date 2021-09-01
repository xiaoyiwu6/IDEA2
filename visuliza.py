import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 0->undefined   1->trees,  2->bldg    3->water   4->roads
palette = {0: (0, 0, 0),
           1: (0, 255, 0),
           2: (255, 0, 0),
           3: (0, 0, 255),
           4: (255, 255, 0)}

invert_palette = {v: k for k, v in palette.items()}


def convert_to_color(arr_2d, palette=palette):
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color(arr_3d, palette=invert_palette):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d


# prepare dataset

transform = transforms.Compose([
    transforms.ToTensor(),
])


class train_dataset(Dataset):
    def __init__(self, img_path, label_path, transform=None):
        super().__init__()
        self.dataset = os.listdir(img_path)
        self.labels = os.listdir(label_path)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def preprocess(self, img):
        img_nd = np.array(img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
            img_trans = img_nd.transpose((2, 0, 1))
            return img_trans

        # HWC to CHW
        return img_nd

    def __getitem__(self, idx):
        image = Image.open(os.path.join('./unet_train/bldg/src/', self.dataset[idx]))
        label = Image.open(os.path.join('./unet_train/bldg/label/', self.labels[idx]))

        mask = self.preprocess(label)

        if self.transform is not None:
            image = self.transform(image)

        return image, mask


bldg_dataset = train_dataset('./unet_train/bldg/src/', './unet_train/bldg/label/', transform=transform)
bldg_dataloader = torch.utils.data.DataLoader(bldg_dataset, batch_size=4, num_workers=0, shuffle=True)


# create Unet
class BasicConv2d(nn.Module):
    def __init__(self, inp, oup):
        super(BasicConv2d, self).__init__()
        self.conv1 = nn.Conv2d(inp, oup, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(oup, oup, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(oup)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = F.elu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn(x)
        return F.elu(x, inplace=True)


class Unet(nn.Module):
    def __init__(self, num_classes=1):
        super(Unet, self).__init__()
        self.conv1 = BasicConv2d(3, 32)
        self.conv2 = BasicConv2d(32, 64)
        self.conv3 = BasicConv2d(64, 128)
        self.conv4 = BasicConv2d(128, 256)
        self.conv5 = BasicConv2d(256, 512)

        self.conv6 = BasicConv2d(768, 256)
        self.conv7 = BasicConv2d(384, 128)
        self.conv8 = BasicConv2d(192, 64)
        self.conv9 = BasicConv2d(96, 32)

        self.MaxPool = nn.MaxPool2d(2, 2)

        self.conv10 = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        dec1 = self.conv1(x)
        pool_dec1 = self.MaxPool(dec1)

        dec2 = self.conv2(pool_dec1)
        pool_dec2 = self.MaxPool(dec2)

        dec3 = self.conv3(pool_dec2)
        pool_dec3 = self.MaxPool(dec3)

        dec4 = self.conv4(pool_dec3)
        pool_dec4 = self.MaxPool(dec4)

        center = self.conv5(pool_dec4)

        up6 = torch.cat([
            dec4, F.upsample_bilinear(center, dec4.size()[2:])], 1)
        dec6 = self.conv6(up6)

        up7 = torch.cat([
            dec3, F.upsample_bilinear(dec6, dec3.size()[2:])], 1)
        dec7 = self.conv7(up7)

        up8 = torch.cat([
            dec2, F.upsample_bilinear(dec7, dec2.size()[2:])], 1)
        dec8 = self.conv8(up8)

        up9 = torch.cat([
            dec1, F.upsample_bilinear(dec8, dec1.size()[2:])], 1)
        dec9 = self.conv9(up9)

        return self.conv10(dec9)


def PA(outputs, target):
    outputs = outputs.float()
    tmp = outputs == target
    return (torch.sum(tmp).float() / outputs.nelement())


# train network
from torch.optim import lr_scheduler

model = Unet()
model = torch.load('bldg_15.pth')

# predict
img = Image.open('./test/test_bldg/6466.png')
img = transform(img)
img = img.unsqueeze(0)
img = img.to('cuda', dtype=torch.float32)
model.eval()
with torch.no_grad():
    out = model(img)
    out = torch.sigmoid(out)
    probs = out.squeeze(0)
    probs = probs.cpu()
    full_mask = probs.squeeze().cpu().numpy()
    full_mask = full_mask > 0.3
    RGB = convert_to_color(full_mask)
    plt.imshow(RGB)
    plt.show()