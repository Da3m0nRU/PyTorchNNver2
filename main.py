import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import csv
import onnx
import numpy as np
from onnx2pytorch import ConvertModel


# Установка устройства (CPU или GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Определение преобразований для нормализации данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Загрузка набора данных MNIST
train_dataset = datasets.MNIST(
    root='data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(
    root='data', train=False, transform=transform, download=True)

# Определение загрузчиков данных
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Определение класса сети


class Subnetwork(nn.Module):
    def __init__(self):
        super(Subnetwork, self).__init__()
        self.fc = nn.Linear(784, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MultimodularNetwork(nn.Module):
    def __init__(self, subnetworks):
        super(MultimodularNetwork, self).__init__()
        self.subnetworks = nn.ModuleList(subnetworks)

    def forward(self, x):
        outputs = []
        for subnetwork in self.subnetworks:
            outputs.append(subnetwork(x))
        return torch.cat(outputs, dim=1)


def save_network_as_csv(filepath):
    # Загрузка сохраненной мультимодульной сети
    multimodular_network = MultimodularNetwork(
        [Subnetwork() for _ in range(10)]).to(device)
    multimodular_network.load_state_dict(torch.load(filepath))

    # Извлечение и сохранение параметров в формате CSV
    parameters = multimodular_network.state_dict()
    with open('multimodular_network.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in parameters.items():
            writer.writerow([key, value.numpy()])


def save_network_as_onnx(filepath):
    # Загрузка сохраненной мультимодульной сети
    multimodular_network = MultimodularNetwork(
        [Subnetwork() for _ in range(10)]).to(device)
    multimodular_network.load_state_dict(torch.load(filepath))
    multimodular_network.eval()

    # Создание входного примера для генерации графа
    example_input = torch.randn(1, 1, 28, 28).to(device)

    # Экспорт модели в ONNX
    torch.onnx.export(multimodular_network, example_input,
                      'multimodular_network.onnx')


def save_torchscript(filepath):
    # Load the saved multimodular network
    multimodular_network = MultimodularNetwork(
        [Subnetwork() for _ in range(10)]).to(device)
    multimodular_network.load_state_dict(torch.load(filepath))
    multimodular_network.eval()

    # Trace the model to obtain the TorchScript representation
    example_input = torch.randn(1, 1, 28, 28).to(device)
    traced_model = torch.jit.trace(multimodular_network, example_input)

    # Save the TorchScript to a file
    torch.jit.save(traced_model, 'multimodular_network.pt')


def train_subnetworks(num_epochs):
    subnetworks = []
    criterion = nn.BCEWithLogitsLoss()

    for digit in range(10):
        subnetwork = Subnetwork().to(device)
        subnetworks.append(subnetwork)

        # Создание экземпляра оптимизатора
        optimizer = optim.Adam(subnetwork.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            subnetwork.train()
            for images, labels in train_loader:
                images = images.to(device)
                # Изменение формы меток для соответствия размеру выхода
                labels = (labels == digit).float().unsqueeze(1).to(device)

                optimizer.zero_grad()
                outputs = subnetwork(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(
                f"Сеть {digit} | Эпоха [{epoch + 1}/{num_epochs}] | Потери: {loss.item():.4f}")

    # Создание мультимодульной сети
    multimodular_network = MultimodularNetwork(subnetworks).to(device)

    # Тестирование мультимодульной сети
    multimodular_network.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = multimodular_network(images)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Модульная нейронная сеть | Точность: {accuracy:.2f}%")

    # Сохранение мультимодульной сети
    torch.save(multimodular_network.state_dict(), 'multimodular_network.pth')

    save_torchscript("multimodular_network.pth")

    user_input = input(
        "Хотите сохранить модель в формате CSV? '1' (да), '0' (нет): ")
    if user_input == '1':
        save_network_as_csv('multimodular_network.pth')

    user_input = input(
        "Хотите сохранить модель в формате onnx? '1' (да), '0' (нет): ")
    if user_input == '1':
        save_network_as_onnx('multimodular_network.pth')


def test_saved_network(filename, filetype):
    # Загрузка сохраненной мультимодульной сети
    if filetype == "pth":
        multimodular_network = MultimodularNetwork(
            [Subnetwork() for _ in range(10)]).to(device)
        multimodular_network.load_state_dict(
            torch.load(filename))
        # Тестирование мультимодульной сети
        multimodular_network.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = multimodular_network(images)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Мультимодульная сеть | Точность: {accuracy:.2f}%")
    elif filetype == "csv":
        return
    elif filetype == "onnx":
        return
    else:
        print("Неподдерживаемый тип файла")
        return


# Пользовательский ввод для тренировки или тестирования
print("Команды:\n'1': Обучение нейронной сети\n'2': Тестирование нейронной сети\n'3': Выгрузка нейронной сети в .csv формат\n'4': Выгрузка нейронной сети в .onnx формат\n'5': Выгрузка нейронной сети в .pt формат\n'6': Выход из программы\n")
while True:
    user_input = input("Введите вашу команду: ")
    if user_input.lower() == "1":
        num_epoch = int(input("Введите количество эпох: "))
        train_subnetworks(num_epoch)
    elif user_input.lower() == "2":
        print("Выберите формат модели для тестирования:\n'1': pth\n'2': csv\n'3': onnx\n")
        user_input = input("Введите ваш выбор: ")
        if user_input.lower() == "1":
            test_saved_network("multimodular_network.pth", "pth")
        elif user_input.lower() == "2":
            test_saved_network("multimodular_network.csv", "csv")
        elif user_input.lower() == "3":
            test_saved_network("multimodular_network.onnx", "onnx")
    elif user_input.lower() == "3":
        save_network_as_csv("multimodular_network.pth")
    elif user_input.lower() == "4":
        save_network_as_onnx('multimodular_network.pth')
    elif user_input.lower() == "5":
        save_torchscript('multimodular_network.pth')
    elif user_input.lower() == "6":
        break
    else:
        print("Неверный ввод. Пожалуйста, введите '1' или '2'.")
