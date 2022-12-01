for model_nr in [19, 29, 49]:
  print(f"Model: {model_nr}")
  rgb_model = torch.load(".models/RGB_model_" + str(model_nr), map_location=torch.device('cpu'))
  gray_model = torch.load(".models/Grayscale_model_" + str(model_nr), map_location=torch.device('cpu'))
  combined_model = torch.load(".models/Combined_model_" + str(model_nr), map_location=torch.device('cpu')) 

  print("Test on normal")
  test(testloader, rgb_model, nn.CrossEntropyLoss())
  test(testloader, gray_model, nn.CrossEntropyLoss(), gray_scale)
  test(testloader, combined_model, nn.CrossEntropyLoss())

  print("Test on GRB processing")
  test(testloader, rgb_model, nn.CrossEntropyLoss(), transform=GRB)
  test(testloader, gray_model, nn.CrossEntropyLoss(), gray_scale, transform=GRB)
  test(testloader, combined_model, nn.CrossEntropyLoss(), transform=GRB)

  print("Test on RBG processing")
  test(testloader, rgb_model, nn.CrossEntropyLoss(), transform=RBG)
  test(testloader, gray_model, nn.CrossEntropyLoss(), gray_scale, transform=RBG)
  test(testloader, combined_model, nn.CrossEntropyLoss(), transform=RBG)

  print("Test on BGR processing")
  test(testloader, rgb_model, nn.CrossEntropyLoss(), transform=BGR)
  test(testloader, gray_model, nn.CrossEntropyLoss(), gray_scale, transform=BGR)
  test(testloader, combined_model, nn.CrossEntropyLoss(), transform=BGR)

  print("Test on GBR processing")
  test(testloader, rgb_model, nn.CrossEntropyLoss(), transform=GBR)
  test(testloader, gray_model, nn.CrossEntropyLoss(), gray_scale, transform=GBR)
  test(testloader, combined_model, nn.CrossEntropyLoss(), transform=GBR)
