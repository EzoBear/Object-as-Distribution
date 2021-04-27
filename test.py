from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from sources.mappers import COCOMapper
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from sources import add_default_config
from detectron2.utils.visualizer import Visualizer

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def build_train_loader(cfg):
        if "coco" == cfg.DATASETS.NAME:
            mapper = COCOMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_default_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def split_axis_from_polygons(polygons): 
    x_axis = [] 
    y_axis = [] 
    for i in polygons[::2]: 
        x_axis.append(i) 
    for i in polygons[1::2]:
        y_axis.append(i) 
    return (np.array(x_axis),np.array(y_axis))

def bivariate_normal_pdf(
            domain_x_min, domain_x_max,      
            domain_y_min, domain_y_max,                           
            domain_x, domain_y,
    ):
    mean_x, std_x = domain_x.mean(), domain_x.std()
    mean_y, std_y = domain_y.mean(), domain_y.std()

    X = torch.arange(domain_x_min, domain_x_max, 1.0)
    Y = torch.arange(domain_y_min, domain_y_max, 1.0)
    X, Y = torch.meshgrid(X, Y)
    POS = torch.empty(X.shape + (2,))
    POS[:, :, 0] = X
    POS[:, :, 1] = Y
    
    P = np.cov(domain_x, domain_y, bias=True)[0][1]/(std_x*std_y)
    Z = (POS[:, :, 0] - mean_x)**2 / std_x**2 \
         - (2*P*(POS[:, :, 0] - mean_x)*(POS[:, :, 1] - mean_y)) / (std_x * std_y) \
         + (POS[:, :, 1] - mean_y)**2 / std_y**2 
    PDF = (1.0 / 2.0 * np.pi * std_x * std_y * np.sqrt(1-P**2)) * np.exp(-Z/(2*(1-P**2)))
    return np.flip(np.rot90(PDF.numpy(),k=1,axes=(0,1)),axis=0)/PDF.max()

def show_result(image, heatmap):
    #processing for formatting
    v = Visualizer(image)
    out = v.overlay_instances(boxes=instances.gt_boxes)
    base_image = out.get_image()/255
    heatmap = heatmap.repeat(3,1,1).permute(1,2,0).numpy()
    x, y, z = base_image, heatmap, (base_image+heatmap)/2

    #print image
    fig = plt.figure(figsize=(5, 5))
    grid = ImageGrid(fig, 111,nrows_ncols=(1, 3), axes_pad=0.1, label_mode="L",)
    for ax, im in zip(grid, [x, y, z]):
        ax.imshow(im, origin="upper", vmin=0, vmax=1)
    plt.show()
 
if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    data_loader = iter(build_train_loader(cfg))

    data = next(data_loader)
    image = data[0]['image'].permute(1,2,0).contiguous()
    instances = data[0]['instances']
    
    heatmap = None
    for polygon in instances.gt_masks.polygons:
        points = split_axis_from_polygons(polygon[0])
        pdf = bivariate_normal_pdf(
            domain_x_min = 0,
            domain_x_max = 513, 
            domain_y_min = 0,
            domain_y_max = 513, 
            domain_x = points[0], 
            domain_y = points[1]
        )
        if heatmap is None: heatmap = pdf
        else: heatmap = np.maximum(heatmap,pdf)
  
    show_result(image, heatmap)
    