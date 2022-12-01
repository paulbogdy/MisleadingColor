def GRB(X):
  red = X[:,0,:,:].unsqueeze(1)
  green = X[:,1,:,:].unsqueeze(1)
  blue = X[:,2,:,:].unsqueeze(1)
  return torch.cat((green, red, blue), 1)

def RBG(X):
  red = X[:,0,:,:].unsqueeze(1)
  green = X[:,1,:,:].unsqueeze(1)
  blue = X[:,2,:,:].unsqueeze(1)
  return torch.cat((red, blue, green), 1)

def BGR(X):
  red = X[:,0,:,:].unsqueeze(1)
  green = X[:,1,:,:].unsqueeze(1)
  blue = X[:,2,:,:].unsqueeze(1)
  return torch.cat((blue, green, red), 1)

def GBR(X):
  red = X[:,0,:,:].unsqueeze(1)
  green = X[:,1,:,:].unsqueeze(1)
  blue = X[:,2,:,:].unsqueeze(1)
  return torch.cat((green, blue, red), 1)

def gray_scale(x):
  return ((x[:,0,:,:] + x[:,1,:,:] + x[:,2,:,:])/3).unsqueeze(1)
