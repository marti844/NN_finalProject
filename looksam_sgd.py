import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt

logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Defination of ResNet-18
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
testloader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

class LookSAM(torch.optim.Optimizer):

    def __init__(self, k, alpha, model, base_optimizer, criterion, rho=0.05, **kwargs):

        defaults = dict(alpha=alpha, rho=rho, **kwargs)
        self.model = model
        super(LookSAM, self).__init__(self.model.parameters(), defaults)

        self.k = k
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.criterion = criterion

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.criterion = criterion
        

    @staticmethod
    def normalized(g):
        return g / g.norm(p=2)

    def step(self, t, samples, targets, zero_grad=False):
        if not t % self.k:
            group = self.param_groups[0]
            scale = group['rho'] / (self._grad_norm() + 1e-7)

            for index_p, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                self.state[p]['old_p'] = p.data.clone()
                self.state[f'old_grad_p_{index_p}']['old_grad_p'] = p.grad.clone()

                with torch.no_grad():
                    e_w = p.grad * scale.to(p)
                    p.add_(e_w)

            self.criterion(self.model(samples), targets).backward()

        group = self.param_groups[0]
        for index_p, p in enumerate(group['params']):
            if p.grad is None:
                continue
            if not t % self.k:
                old_grad_p = self.state[f'old_grad_p_{index_p}']['old_grad_p']
                g_grad_norm = LookSAM.normalized(old_grad_p)
                g_s_grad_norm = LookSAM.normalized(p.grad)
                self.state[f'gv_{index_p}']['gv'] = torch.sub(p.grad, p.grad.norm(p=2) * torch.sum(
                    g_grad_norm * g_s_grad_norm) * g_grad_norm)

            else:
                with torch.no_grad():
                    gv = self.state[f'gv_{index_p}']['gv']
                    p.grad.add_(self.alpha.to(p) * (p.grad.norm(p=2) / (gv.norm(p=2) + 1e-8) * gv))

            p.data = self.state[p]['old_p']

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device) for group in self.param_groups for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )

        return norm


def calculate_accuracy(loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():  
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def calculate_loss(loader, model, criterion, device):
    total_loss = 0.0
    total_samples = 0

    model.eval()  
    with torch.no_grad():  
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    average_loss = total_loss / total_samples
    return average_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18().to(device)
num_epochs = 100  
criterion = torch.nn.CrossEntropyLoss()
base_optimizer = torch.optim.SGD
optimizer = LookSAM(k=10, alpha=0.7, model=model, base_optimizer=base_optimizer, criterion=criterion, rho=0.05, lr=0.001, momentum=0.9)


train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for train_index, data in enumerate(trainloader, 0):
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        def closure():
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            return loss

        loss = closure()
        running_loss += loss.item()
        optimizer.step(t=train_index, samples=inputs, targets=labels, zero_grad=True)

    avg_train_loss = running_loss / len(trainloader)
    train_accuracy = calculate_accuracy(trainloader, model, device)
    test_accuracy = calculate_accuracy(testloader, model, device)
    avg_test_loss = calculate_loss(testloader, model, criterion, device) 

    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
          f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")



plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue', linestyle='-', marker='o')
plt.plot(test_losses, label='Test Loss', color='red', linestyle='-', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='green', linestyle='-', marker='o')
plt.plot(test_accuracies, label='Test Accuracy', color='orange', linestyle='-', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.savefig('training_results.png')
torch.save(model.state_dict(), 'model_weights.pth')
