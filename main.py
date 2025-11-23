import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn


class Block(nn.Module):
    def __init__(self, in_channels, out_1, out_2, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_1 = out_1
        self.out_2 = out_2
        self.cnn1 = self.cnn_factory(in_channels, out_1, stride=stride)
        self.cnn2 = self.cnn_factory(out_1, out_2, 1)
        self.cnn = nn.Sequential(self.cnn1, self.cnn2)
        self.short_cut = nn.Sequential()
        if stride != 1 or in_channels != out_2:
            self.short_cut = nn.Sequential(
                nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_2,
                kernel_size=1,
                stride=stride,
            ),
            nn.BatchNorm2d(out_2),
            )

    def cnn_factory(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        out = self.cnn(x)
        out += self.short_cut(x)
        out = nn.GELU()(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.start = nn.Sequential(
            nn.Conv2d(
                in_channels=3, 
                out_channels=64, 
                kernel_size=3, 
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.block1 = Block(64, 64, 64, 1)
        self.block2 = Block(64, 128, 128, 2)
        self.block3 = Block(128, 256, 256, 2)
        self.block4 = Block(256, 512, 512, 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 100),
        )

    def forward(self, x):
        x = self.start(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


def train(model, criterion, optimizer, dataloader, device):
    model.train()
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.forward(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(model, device, dataloader):
    model.eval()
    correct = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pre = output.argmax(axis=1, keepdims=True)
        correct += pre.eq(target.view_as(pre)).sum().item()
    print(f"testing acc {correct * 100 / train_dataset.__len__()}%")
    return correct / train_dataset.__len__()


if __name__ == "__main__":
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(0.25),
        ]
    )

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    train_dataset = datasets.CIFAR100(
        root="./DATA", train=True, download=True, transform=transform_train
    )

    test_dataset = datasets.CIFAR100(
        root="./DATA", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=8,
    )
    model = ResNet()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    criterion = torch.nn.CrossEntropyLoss()

    device = "mps"
    model.to(device)

    best_acc = 0
    for x in range(100):
        train(model, criterion, optimizer, train_loader, device)
        acc = test(model, device, test_loader)
        scheduler.step(acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "./best_model.pth")
            print("保存最佳模型")
        print(f"epoch {x} finished")
    print(f"best acc is {best_acc*100}%")
