import torch
import train

model_names = [
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "vgg11_bn",
    "vgg16_bn",
    "mobilenetv2_x0_5",
    "mobilenetv2_x0_75",
    "mobilenetv2_x1_0",
    "mobilenetv2_x1_4",
    "shufflenetv2_x0_5",
    "shufflenetv2_x1_0",
    "shufflenetv2_x1_5",
    "shufflenetv2_x2_0",
    "repvgg_a0",
    "repvgg_a1",
    "repvgg_a2"
]

for model_name in model_names:
  print(f"Testing of the model: {model_name}")

  model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_"+model_name, pretrained=True)

  print("Test on normal")
  test(testloader, model, nn.CrossEntropyLoss())

  print("Test on GRB processing")
  test(testloader, model, nn.CrossEntropyLoss(), transform=GRB)

  print("Test on RBG processing")
  test(testloader, model, nn.CrossEntropyLoss(), transform=RBG)

  print("Test on BGR processing")
  test(testloader, model, nn.CrossEntropyLoss(), transform=BGR)

  print("Test on GBR processing")
  test(testloader, model, nn.CrossEntropyLoss(), transform=GBR)
