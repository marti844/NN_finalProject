import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
import random

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

class ESAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05,beta=1.0,gamma=1.0,adaptive=False,**kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.beta = beta
        self.gamma = gamma

        defaults = dict(rho=rho,adaptive=adaptive, **kwargs)
        super(ESAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7) / self.beta
            for p in group["params"]:
                p.requires_grad = True 
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w * 1)  
                self.state[p]["e_w"] = e_w



        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]: continue
                p.sub_(self.state[p]["e_w"])  
                self.state[p]["e_w"] = 0

                if random.random() > self.beta:
                    p.requires_grad = False

        self.base_optimizer.step()  

        if zero_grad: self.zero_grad()

    def step(self):
        inputs,targets,loss_fct,model,defined_backward = self.paras
        assert defined_backward is not None, "Sharpness Aware Minimization requires defined_backward, but it was not provided"

        model.require_backward_grad_sync = False
        model.require_forward_param_sync = True


        logits = model(inputs)
        loss = loss_fct(logits,targets)

        l_before = loss.clone().detach()
        predictions = logits
        return_loss = loss.clone().detach()
        loss = loss.mean()
        defined_backward(loss)

        self.first_step(True)


        with torch.no_grad():
            l_after = loss_fct(model(inputs),targets)
            instance_sharpness = l_after-l_before

            prob = self.gamma
            if prob >=0.99:
                indices = range(len(targets))
            else:
                position = int(len(targets) * prob)
                cutoff,_ = torch.topk(instance_sharpness,position)
                cutoff = cutoff[-1]

                indices = [instance_sharpness > cutoff] 

        model.require_backward_grad_sync = True
        model.require_forward_param_sync = False



        loss = loss_fct(model(inputs[indices]), targets[indices])
        loss = loss.mean()
        defined_backward(loss)
        self.second_step(True)

        self.returnthings = (predictions,return_loss)
 

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
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
optimizer = ESAM(model.parameters(), base_optimizer=torch.optim.Adam, lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
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
        optimizer.first_step(zero_grad=True)

        closure()
        optimizer.second_step(zero_grad=True)

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
