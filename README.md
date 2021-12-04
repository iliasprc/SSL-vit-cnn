# Ablation study on self-supervised training of CNNs and Vision Transformers (ViT)

Tested self-supervised technique Exploring simple siamese representation learning ov various CNN and ViT architectures



| Image size    | Pretrain | Dataset  | Validation acc | Test acc       |
|---------------|----------|----------|----------------|----------------|
| $32\times 32$ | CIFAR-10 | CIFAR-10 | 88.59          | 88.89          |
| $32\times 32$ | Random   | CIFAR-10 | 83.87          | 83.12          |
| $32\times 32$ | No       | CIFAR-10 | 75.68          | 75.23          |



| Model           | Pretrained weights | Test acc                  |
|-----------------|--------------------|---------------------------|
| ResNet-18       | No                 | 65.73             |
|                 | STL-10             | 70.23                     |
|                 | ImageNet           | 89.82                     |
| EfficientNet-B0 | No                 | 65.30             |
|                 | STL-10             | 69.84                     |
|                 | ImageNet           | 95.03                     |
| ViT             | No                  | 53.74            |
|                 | STL-10             | 60.45                     |
|                 | ImageNet           | 96.26                     |
| PiT             | No                 | 58.14             |
|                 | STL-10             | 64.21                     |
|                 | ImageNet           | 87.31                     |

