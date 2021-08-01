import os
import shutil
import sys
import argparse
import time
import itertools

import numpy as np
import torch
import torch.nn as nn
import warnings
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
from torch.autograd import Variable
# from torch.nn import DataParallel
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from etaprogress.progress import ProgressBar
import glob

# сэмплер для несбалансированных данных (автоматически чаще выбирает угнетенный класс)
from torchsampler import ImbalancedDatasetSampler
# pip install https://github.com/ufoym/imbalanced-dataset-sampler/archive/master.zip

sys.path.append('./')
from utils.util import copy_file, write_json, dir_filename_join
from utils.FocalLoss import FocalLoss

import model_reducer

import utils.main_log

printL = utils.main_log.printL
utils.main_log.init('blur.log', 'blur')

plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='Training on Blur Dataset')
parser.add_argument('--batch_size', '-b', default=45, type=int, help='batch size')
parser.add_argument('--epochs', '-e', default=35, type=int, help='training epochs')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='use gpu or not')
parser.add_argument('--step_size', default=4, type=int, help='learning rate decay interval')
parser.add_argument('--gamma', default=0.2, type=float, help='learning rate decay scope')
parser.add_argument('--interval_freq', '-i', default=12, type=int, help='printing log frequence')
parser.add_argument('--data', '-d', default='./data/data_augu', help='path to dataset')
parser.add_argument('--prefix', '-p', default='classifier', type=str, help='save folder prefix')
parser.add_argument('--best_model_path', default='model_blur.pth.tar', help='best model saved filename')
parser.add_argument('--is_focal_loss', '-f', default=False, type=bool,
                    help='use focal loss or common loss(i.e. cross ectropy loss)(default: False)')

best_acc = 0.0


def main( filter_levels=(6,8,26,22), AfterReducerChanels=3, adaptivepool_size=6, dropout_rate=0, ExitDropout = 0, extra_cat_lines=2, extra_lay= True, full_conv_block=False):
    global args, best_acc
    best_acc = 0.0
    args = parser.parse_args()
    # save source script
    copy_file(args.prefix, __file__)
    import subprocess
    subprocess.call([r"rar.bat", r" u .\classifier\backup_py *.py"])
    # model = models.resnet18(pretrained=False, num_classes=2)
    model = model_reducer.reducer_conv(num_classes=2, filter_levels=filter_levels,
                                       AfterReducerChanels=AfterReducerChanels, adaptivepool_size=adaptivepool_size,
                                       dropout_rate=dropout_rate, ExitDropout=ExitDropout,
                                       extra_cat_lines=extra_cat_lines, extra_lay=extra_lay,
                                       full_conv_block=full_conv_block)
    if args.cuda:
        device = torch.device("cuda:0")
        model.to(device)
        # model = DataParallel(model).cuda()
    else:
        warnings.warn('There is no gpu')

    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = optim.Adamax(model.parameters(), lr=args.lr * 2)
    
    # accelerate the speed of training - вроде как в случае неизменных размеров тензоров
    # torch.backends.cudnn.benchmark = True

    train_loader, val_loader = load_dataset()
    # class_names=['LESION', 'NORMAL']
    class_names = train_loader.dataset.classes
    print(f'Class names: {class_names}')
    if args.is_focal_loss:
        printL('Try focal loss!')
        criterion = FocalLoss().cuda() if args.cuda else FocalLoss()
    else:
        # criterion = nn.CrossEntropyLoss().cuda() if args.cuda else nn.CrossEntropyLoss()
        # criterion = nn.L1Loss().cuda() if args.cuda else nn.L1Loss()
        criterion = nn.MSELoss().cuda() if args.cuda else nn.MSELoss()

    # learning rate decay per epochs
    # exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7, verbose=True)
    # exp_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs, verbose=True)
    exp_lr_scheduler = ReduceLROnPlateau(optimizer, factor=args.gamma, patience=args.step_size, cooldown=2, min_lr=1e-7, verbose=True)
    since = time.time()
    bar = ProgressBar(args.epochs, max_width=65)
    print('-' * 10)
    try:
        checkpoint = torch.load(r'.\classifier\checkpoint.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        printL('checkpoint ok')
    except:
        pass
    try:
        checkpoint = torch.load(r'.\classifier\model_blur.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        printL('Best state loaded')
    except:
        pass
    result_loss = 0.2
    best_loss = 999
    try:
        for epoch in range(args.epochs):
            bar.numerator = epoch+1
            # print(bar)
            train_loss = train(train_loader, model, optimizer, criterion, epoch, str(bar) + ' ')
            val_accuracy, val_loss = validate(model, val_loader, criterion)
            result_loss = (result_loss + 2 * val_loss) / 3
            is_best = val_accuracy > best_acc
            best_acc = max(val_accuracy, best_acc)
            if is_best: best_loss = val_loss
            exp_lr_scheduler.step((val_loss + train_loss*1001)/1002)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'model_reducer',
                'state_dict': model.state_dict(),
                'best_accuracy': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best)
            write_json({
                'epoch': epoch + 1,
                'best_accuracy': best_acc,
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'learing_rate': optimizer.param_groups[0]['lr'],
            }, dir_filename_join(args.prefix, 'best.txt' if is_best else 'curr_stat.txt'))
    except KeyboardInterrupt:
        print('-' * 39)
        printL('Exiting from training early')

    time_elapsed = time.time() - since
    printL('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # compute validate meter such as confusion matrix
    compute_validate_meter(model, dir_filename_join(args.prefix, args.best_model_path), val_loader)
    val_accuracy, val_loss = validate(model, val_loader, criterion)

    # save running parameter setting to json
    write_json(vars(args), dir_filename_join(args.prefix, 'paras.txt'))
    result = (best_loss*5 + result_loss) / 6
    print(f'result={result}')
    return result


def train(train_loader, model, optimizer, criterion, epoch, text_info=''):
    model.train(True)
    printL('{}Epoch {}/{}'.format(text_info, epoch, args.epochs))
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for idx, (inputs, labels) in enumerate(train_loader):
        # wrap them in Variable
        if args.cuda:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        # print(outputs.shape)
        # print(f'outpu={outputs}')
        # print(f'label={labels}')

        _, preds = torch.max(outputs.data, 1)

        loss = criterion(outputs[:, 1], labels.float())
        loss.backward()
        optimizer.step()
        if idx % args.interval_freq == 0:
            print('Train Epoch: {} [{:4}/{:4} ({:2.0f}%)] Loss: {:.5f}'.format(
                epoch + 1, idx * len(inputs), len(train_loader.dataset),
                100. * idx / len(train_loader), loss.item()), end='\r')

        # statistics
        running_loss += loss.item() * len(inputs)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.item() / len(train_loader.dataset)

    printL('Training Loss: {:.4f} Acc: {:.4f}                '.format(epoch_loss, epoch_acc))
    return epoch_loss


def compute_validate_meter(model, best_model_path, val_loader):
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    best_acc = checkpoint['best_accuracy']
    model.eval()
    print('best accuracy={:.4f}'.format(best_acc))
    pred_y = list()
    test_y = list()
    probas_y = list()
    with torch.no_grad():
        for data, target in val_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            probas_y.extend(output.data.cpu().numpy().tolist())
            pred_y.extend(output.data.cpu().max(1, keepdim=True)[1].numpy().flatten().tolist())
            test_y.extend(target.data.cpu().numpy().flatten().tolist())

    confusion = confusion_matrix(pred_y, test_y)
    plot_confusion_matrix(confusion,
                          classes=val_loader.dataset.classes,
                          title='Confusion matrix')
    plt_roc(test_y, probas_y)
    acc=0
    for i, _ in enumerate(pred_y):
        if pred_y[i]==test_y[i]: acc +=1
    print( f'In compute_validate_meter accurate={100*acc/len(pred_y)}')


def plt_roc(test_y, probas_y, plot_micro=False, plot_macro=False):
    assert isinstance(test_y, list) and isinstance(probas_y, list), 'the type of input must be list'
    skplt.metrics.plot_roc(test_y, probas_y, plot_micro=plot_micro, plot_macro=plot_macro)
    plt.savefig(dir_filename_join(args.prefix, 'roc_auc_curve.png'))
    plt.close()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    refence:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(dir_filename_join(args.prefix, 'confusion_matrix.png'))
    plt.close()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # save training state after each epoch
    torch.save(state, dir_filename_join(args.prefix, filename))
    if is_best:
        shutil.copyfile(dir_filename_join(args.prefix, filename),
                        dir_filename_join(args.prefix, args.best_model_path))


def load_dataset():
    if args.data == './data/data_augu':
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        # Normalization parameters for pre-trained PyTorch models
        #mean = np.array([0.485, 0.456, 0.406])
        #std = np.array([0.229, 0.224, 0.225])
        mean = model_reducer.mean
        std = model_reducer.std

        normalize = transforms.Normalize(mean, std)
        train_transforms = transforms.Compose([
            # transforms.RandomChoice([transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20)),transforms.RandomCrop(16*random.randint(2,20))]),
            transforms.RandomCrop(model_reducer.img_tile_size),  # 224
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.CenterCrop(model_reducer.img_tile_size),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = ImageFolder(traindir, train_transforms)
        val_dataset = ImageFolder(valdir, val_transforms)
        print('load data-augumentation dataset successfully!!!')
    else:
        raise ValueError("parameter 'data' that means path to dataset must be in "
                         "['./data/data_augu']")

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              sampler=ImbalancedDatasetSampler(train_dataset),
                              num_workers=4 if args.cuda else 1,
                              pin_memory=True if args.cuda else False)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size ,
                            shuffle=False,
                            num_workers=3 if args.cuda else 1,
                            pin_memory=True if args.cuda else False)
    return train_loader, val_loader


def validate(model, val_loader, criterion):
    #print(model)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += criterion(output[:, 1], target.float()).item() * len(data)
            # print(f'output={output}')
            # print(f'target={target}')
            # get the index of the max log-probability
            # pred = output.data.max(1, keepdim=True)[1]
            _, pred = torch.max(output.data, 1)
            # correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
            correct += torch.sum(pred == target.data).item()

    test_loss /= len(val_loader.dataset)
    test_acc = 100. * correct / len(val_loader.dataset)
    printL('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), test_acc))
    return test_acc, test_loss


def get_filenames(patch_to_scan, wildcards):
    return [y for x in os.walk(patch_to_scan) for y in glob.glob(os.path.join(x[0], wildcards))]


def test_all(path):
    filenames = get_filenames(path, '*.jpg')
    for filename in filenames:
        printL(f'file= {filename} result={model_reducer.get_blur_predict(filename) :3.3f}')


if __name__ == '__main__':
    #test_all(r'.\test')
    _ = main()
