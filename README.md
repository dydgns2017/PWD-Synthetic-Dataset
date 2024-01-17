# PWD-Synthetic-Dataset
PWD(Pine Wilt Disease) synthesis data generated using 3D rendering tools

![title.png](./images/title.png)  

## Environments

- python  version : 3.11.4
- pytorch version :  2.0.1
- GPU : 2080Ti*8EA

```bash
## conda env setup
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Dataset

... Coming soon (download link may be google drive)

### Desciption Table for Dataset

|R|S|Spr1|Spr2|Spr3|
|:-:|:-:|:-:|:-:|:-:|
|Real Dataset|Synthetic Dataset|Synthetic Image Translation Dataset (one)|Synthetic Image Translation Dataset (two)| Synthetic Image Translation (three)|

\+ meaning is mixed dataset

## Pre-Trained Model

... Coming soon

## Training

```bash
## Train Synthetic, Real, Mix Dataset
## already pre-trained model exists, if you want just inference skip this section(training) 
... Coming soon
```

## Results with Inference

```bash
## Inference Code
... Coming soon
```

## Authors and Citation

Authors : Yonghoon Jung, Sanghyun Byun, Bumsoo Kim, Sareer Ul Amin, Sanghyun Seo

```
... Coming soon
```

## Acknowledgements

We are very grateful to the CSLAB researchers at Chung-Ang University (Prof. Park Sang-Oh, Prof. Lee Jae-Hwan, Dr. Nam Sang-Hyuk, Dr. Cho Min-Gyu, M.S. Lee Yo-Seb, and M.S. Kim Dong-Hyeon) and Prof. Kang Dong-Wann at Seoul National University of Science and Technology for their great help in collecting the real dataset. We are also very grateful to M.S. Yoo Jae-Seok, Won-Seop Shin and students Lee Jeong-Ha, Lee Won-Byung, and Oh Chang-Jin for data labeling. Finally, we'd like to thank Assoicate Seung-Yong Ji (Monitoring & Analysis Department, Korea Forestry Promotion Institute) for responding to so many requests.

This study was carried out with the support of R&D Program for Forest Science Technology (Project No.2021338C10-2123-CD02) provided by Korea Forest Service (Korea Forestry Promotion Institute).

## References

- https://github.com/hankyul2/EfficientNetV2-pytorch for implementation EfficientNetv2
- I2I techniques ([one](https://github.com/taesungp/contrastive-unpaired-translation), [two](https://github.com/sapphire497/query-selected-attention), [three](https://github.com/Mid-Push/Decent))
