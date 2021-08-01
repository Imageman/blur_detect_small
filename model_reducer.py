import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
# from torch.nn import DataParallel
from torch.autograd import Variable

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

img_tile_size = 328

def double_conv(in_channels, out_channels, drop_out_rate=0.0, ExitDropout=0.0):
    if ExitDropout > 0.001:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SELU(inplace=False),
            nn.Dropout(drop_out_rate),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.SELU(inplace=False),
            nn.Dropout(ExitDropout),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SELU(inplace=False),
            nn.Dropout(drop_out_rate),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.SELU(inplace=False),
        )


def half_conv(in_channels, out_channels, drop_out_rate=0.0, ExitDropout=0.0):
    if ExitDropout > 0.001:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SELU(inplace=False),
            nn.Dropout(ExitDropout),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            # nn.BatchNorm2d(out_channels , momentum=0.9999),
            nn.SELU(inplace=False),
            nn.Dropout(drop_out_rate),
        )


# для классификации - на выходе num_classes чисел
class reducer_conv(nn.Module):

    def __init__(self, num_classes, filter_levels=(6,8,26,22), AfterReducerChanels=3, adaptivepool_size=6, dropout_rate=0, ExitDropout = 0, extra_cat_lines=2, extra_lay= True, full_conv_block=False):
        super().__init__()

        # L1, L2, L3, L4 = 6, 8, 26, 22 # small

        # L1, L2, L3, L4 = 32, 64, 64, 128  # DeepFocus facebook-like (extra_lay=False, full_conv_block=False)
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6201886/

        L1, L2, L3, L4 = filter_levels[0], filter_levels[1], filter_levels[2], filter_levels[3]

        self.AfterReducerChanels = AfterReducerChanels
        self.adaptivepool_size = adaptivepool_size
        self.dropout_rate = dropout_rate
        self.ExitDropout = ExitDropout
        self.extra_cat_lines = extra_cat_lines
        self.extra_lay = extra_lay

        if full_conv_block:
            self.func_block = double_conv
        else:
            self.func_block = half_conv

        self.dconv_down1 = self.func_block(3, L1, drop_out_rate=self.dropout_rate)
        self.dconv_down2 = self.func_block(L1 + 3, L2, drop_out_rate=self.dropout_rate)
        self.dconv_down3 = self.func_block(L2 + self.extra_cat_lines, L3, drop_out_rate=self.dropout_rate)
        self.dconv_down4 = self.func_block(L3 + self.extra_cat_lines, L4, drop_out_rate=self.dropout_rate)
        self.dconv_5 = self.func_block(L4 + self.extra_cat_lines, L4, drop_out_rate=self.dropout_rate,
                                       ExitDropout=self.ExitDropout)

        self.reducer = nn.Conv2d(L4, self.AfterReducerChanels, 1)

        self.fc1 = nn.Sequential(
            nn.Linear(self.AfterReducerChanels * self.adaptivepool_size * self.adaptivepool_size, 64),
            nn.LeakyReLU(inplace=False),
            #nn.Dropout(0.031),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 16),
            nn.LeakyReLU(inplace=False),
            #nn.Dropout(0.06),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(16, num_classes),
            # nn.Sigmoid(),
            nn.Softmax(dim=-1),
        )

        self.maxpool = nn.MaxPool2d(2)  # classic
        self.avgpool = nn.AvgPool2d(2) # усреднение пикселей
        self.adaptivepool = nn.AdaptiveMaxPool2d(output_size=self.adaptivepool_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x2 = self.avgpool(x[:, 0:3, :, :])
        x = self.dconv_down1(x)
        x = self.maxpool(x)
        x = torch.cat([x, x2], dim=1)

        x2 = self.avgpool(x[:, 0:self.extra_cat_lines, :, :])
        x = self.dconv_down2(x)
        x = self.maxpool(x)
        x = torch.cat([x, x2], dim=1)

        x2 = self.avgpool(x[:, 0:self.extra_cat_lines, :, :])
        x = self.dconv_down3(x)
        x = self.maxpool(x)
        x = torch.cat([x, x2], dim=1)

        x2 = x[:, 0:self.extra_cat_lines, :, :]
        x = self.dconv_down4(x)

        if self.extra_lay:
            x = torch.cat([x, x2], dim=1)
            x = self.dconv_5(x)  # дополнительная обработка

        x = self.reducer(x)
        x = self.adaptivepool(x)

        x = x.view(-1, self.AfterReducerChanels * self.adaptivepool_size * self.adaptivepool_size)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


model_blur = None


def get_blur_predict_small(img):
    global model_blur

    if isinstance(img, str):
        img = Image.open(img)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.Tensor

    hr_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    data = hr_transform(img)
    # print(data.shape)
    data = torch.reshape(data, (1, data.shape[0], data.shape[1], data.shape[
        2]))  # добавляем ещё одно измерение для избежания ошибки too many indices for array
    # print(data.shape)

    if model_blur == None:
        model_blur = model_blur_load(device)

    data = data.type(Tensor)
    with torch.no_grad():
        data = Variable(data)
        data = data.to(device)
        output = model_blur(data)
    output = Tensor.numpy(Tensor.cpu(output).detach())
    result = output[0][0]  # 0 - это номер картинки, а второй - это прогноз значения для класса

    return result


"""
Split to slices and get average predict
"""
def get_blur_predict(img):
    if isinstance(img, str):
        img = Image.open(img)  # нужно загрузить картинку

    img = np.asarray(img)
    tile_size = img_tile_size
    if img.shape[0] < tile_size * 1.2 and img.shape[1] < tile_size * 1.2:
        return get_blur_predict_small(Image.fromarray(np.uint8(img)))

    delta0 = (img.shape[0] - tile_size) // (img.shape[0] // tile_size)
    if delta0 < tile_size * 0.9:
        delta0 = tile_size
    delta1 = (img.shape[1] - tile_size) // (img.shape[1] // tile_size)
    count = 0
    score = 0
    if delta1 < tile_size * 0.9:
        delta1 = tile_size
    for r in range(0, img.shape[0], delta0):
        for c in range(0, img.shape[1], delta1):
            if r + tile_size > img.shape[0]:
                continue
            if c + tile_size > img.shape[1]:
                continue
            count += 1
            score += get_blur_predict_small(Image.fromarray(np.uint8(img[r:r + tile_size, c:c + tile_size, :])))
    return score / count


def model_blur_load(device):
    SCRIPT_DIRECTORY = os.path.realpath(os.path.dirname(__file__))
    with torch.no_grad():
        model_blur = reducer_conv(num_classes=2, filter_levels=(6,8,26,22), AfterReducerChanels=3, adaptivepool_size=6, dropout_rate=0, ExitDropout = 0, extra_cat_lines=2, extra_lay= True, full_conv_block=False).to(device)
        model_blur.eval()
        checkpoint = torch.load(os.path.join(SCRIPT_DIRECTORY, r'model_blur.pth.tar'), map_location=device)
        model_blur.load_state_dict(checkpoint['state_dict'])

    # print('State loaded')
    return model_blur


import glob
import os


def get_filenames(patch_to_scan, wildcards):
    return [y for x in os.walk(patch_to_scan) for y in glob.glob(os.path.join(x[0], wildcards))]


def test_all(path):
    filenames = get_filenames(path, '*.jpg')
    for filename in filenames:
        print(f'file= {filename} result={get_blur_predict(filename) :3.3f}')


if __name__ == '__main__':
    test_all(r'.\test')
    print('Press enter....')
    input()
