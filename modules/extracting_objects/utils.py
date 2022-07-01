import numpy as np
import cv2
import torch 
import matplotlib.pyplot as plt

def get_gradcam(model, image, label, size):
  label.backward(retain_graph=True)
  gradients = model.get_activations_gradient()
  pooled_gradients = torch.mean(gradients, dim=[0,2,3])
  activations = model.get_activations(image).detach()

  for i in range(activations.shape[1]):
    activations[:,i,:,:]*=pooled_gradients[i]
  heatmap=torch.mean(activations, dim=1).squeeze().cpu()
  heatmap = np.maximum(heatmap, 0)


  heatmap /= torch.max(heatmap)

  heatmap_exp = cv2.resize(heatmap.numpy(), (size, size))
  return heatmap_exp, heatmap.numpy()


def plot_heatmap(denorm_image, actual, displaying, heatmap, pred):

    fig, (ax1, ax2) = plt.subplots(figsize=(20,20), ncols=2)


    fig.suptitle(f'Predicted class - {classes[actual]}, Displaying class - {classes[displaying]}')
    fig.tight_layout()
    ps = torch.nn.Softmax(dim = 1)(pred).cpu().detach().numpy()
    ax1.imshow(denorm_image)



    ax2.imshow(denorm_image)
    ax2.imshow(heatmap, cmap='magma', alpha=0.7)

# Regions:
def dfs(grid, r, c, color, size, threshold):

  rowNbr = [0, 0, 1, -1]
  colNbr = [1, -1, 0, 0]

  grid[r][c] = color
  for i in range(4):
    tr = r+rowNbr[i]
    tc = c+colNbr[i]

    if(tr<0 or tr>=size or tc < 0 or tc >= size or grid[tr][tc] < threshold or grid[tr][tc] > 1):
      continue
    grid = dfs(grid, tr, tc, color, size, threshold)

  return grid
def roi(sm_heatmap, size):
  regions = []

  rowNbr = [0, 0, 1, -1]
  colNbr = [1, -1, 0, 0]

  color = 2
  color_map = {}
  threshold = sm_heatmap.mean()
  for i in range(size):
    for j in range(size):
      if(sm_heatmap[i][j] > 0.2 and sm_heatmap[i][j] < 2):
        sm_heatmap = dfs(sm_heatmap, i, j, color, size, threshold)
        color_map[color] = [i, j]
        color+=1
  
  sm_heatmap = sm_heatmap.astype(int)
#   print(sm_heatmap)  # To get a send of the output
  for i in range(2, color):
    start = color_map[i]

    cells = []
    cells.append(start)

    for j in range(4):
      tx = start[0]+rowNbr[j]
      ty = start[1]+colNbr[j] 

      if(tx<0 or tx>=size or ty < 0 or ty >= size):
        continue
      if(sm_heatmap[tx][ty] == i):
        cells.append([tx, ty])
    
    # top left and bottom right

    ml, mr, mt, mb = 14,-1,14,-1
    for c in cells:
      if(c[0] < ml): ml = c[0]
      if(c[0] > mr): mr = c[0]
      if(c[1] < mt): mt = c[1]
      if(c[1] > mb): mb = c[1]

    regions.append([[ml, mt], [mr, mb]])

  # Convert them to pixel coordinates
  for r in regions:
    r[0] = [x*16 for x in r[0]]
    r[1] = [(x+1)*16-1 for x in r[1]]
  # print(regions)
  return regions