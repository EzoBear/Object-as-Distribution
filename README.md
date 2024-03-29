# Object-as-Distribution
Mobilepose-style implementation of object as distribution. Not parameterized method, we use heatmap based method. Additionally, although not specified in the paper, we conduct to normalization of the heatmap. This result is more similar to result of the paper.

<img src="demo/Object-as-Distribution.png">

**[Object As Distribution](https://arxiv.org/abs/1907.12929)** 

**[Mobilepose](https://arxiv.org/abs/2003.03522)**

## Project structure based on Detectron2

```
* Object-as-Distribution
  + configs
      + default_config.yaml
  - datasets(please add your repository of datasets)
      - coco
      - voc
      ...
  + sources
      + mappers
          + coco_mapper.py
      + utils
      ...
      + config.py
  + test.py
  
```

## Usage

```
python test.py --config configs/default_config.yaml --eval-only --num-gpus 2
```

## Demo
<img src="demo/Demo_1.png">
<img src="demo/Demo_2.png">
<img src="demo/Demo_3.png">
