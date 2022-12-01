import models.py
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def train(dataloader, model, nr_epochs=10, process=lambda x: x, model_name="RGB"):

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    size = len(dataloader.dataset)

    train_acc = []
    train_loss = []

    val_acc = []
    val_loss = []

    for epoch in range(nr_epochs):
      print(f"Training epoch: {epoch}")
      model.train()
      for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = process(X)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        #   loss, current = loss.item(), batch * len(X)
        #   print(f"[batch: {current:>5d}/{size:>5d}]")
      acc_train, loss_train = test(trainloader, model, loss_fn, process)
      acc_val, loss_val = test(testloader, model, loss_fn, process)
      train_acc.append(acc_train)
      train_loss.append(loss_train)
      val_acc.append(acc_val)
      val_loss.append(loss_val)

      torch.save(model, "./models/"+model_name+"_model_"+str(epoch))
    return (train_acc, train_loss, val_acc, val_loss)

def test(dataloader, model, loss_fn, process=lambda x: x, transform=lambda x: x):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = transform(X)
            X = process(X)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct, test_loss      

if __name__ == "__main__":
    batch_size = 64

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    nr_epochs = 75

    rgb_model = RGBNet()
    rgb_model.to(device)
    rgb_acc, rgb_loss, rgb_val_acc, rgb_val_loss = train(trainloader, rgb_model, nr_epochs, model_name="RGB")

    gray_model = GrayNet()
    gray_model.to(device)
    gray_acc, gray_loss, gray_val_acc, gray_val_loss = train(trainloader, gray_model, nr_epochs, process=gray_scale, model_name="Grayscale")

    combined_model = CombinedNet()
    combined_model.to(device)
    combined_acc, combined_loss, combined_val_acc, combined_val_loss = train(trainloader, combined_model, nr_epochs, model_name="Combined")

    plt.plot(range(nr_epochs), rgb_val_acc, label = "rgb_val")
    plt.plot(range(nr_epochs), gray_val_acc, label = "gray_val")
    plt.plot(range(nr_epochs), combined_val_acc, label = "combined_val")
    plt.plot(range(nr_epochs), rgb_acc, label = "rgb")
    plt.plot(range(nr_epochs), gray_acc, label = "gray")
    plt.plot(range(nr_epochs), combined_acc, label = "combined")

    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()

    plt.show()
