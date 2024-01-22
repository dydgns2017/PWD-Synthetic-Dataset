# PWD-Synthetic-Dataset
PWD(Pine Wilt Disease) synthesis data generated using 3D rendering tools

![title.png](./images/title.png)  

## Highlights

- Pine Wilt Disease requires early detection due to its severity and lack of cure.
- Our synthetic dataset creation outperforms traditional PWD data collection process.
- Real and synthetic data combination improves PWD F1 Score to 92.88%.
- Synthetic data method aids forest preservation, applies to other agriculture.

## Environments

- python  version : 3.11.4
- pytorch version :  2.0.1
- GPU : 2080Ti*8EA

```bash
## conda env setup
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Dataset & Pre-Trained Model Downloads

![title.png](./images/data_const.png)  

Fill out the form through that [link](https://docs.google.com/forms/d/e/1FAIpQLSc1FoXqo3Rg39M_A8Q6VT07oeJTf7BK7faU9uh2N9cvG_NC3A/viewform?usp=sf_link) and we'll give you access to the dataset.

Pre-trained model [downloads link](https://drive.google.com/drive/folders/1K-XfZnuhrESKyhlb9z9zK1cmyWAyUggU?usp=drive_link)



### Desciption Table for Dataset and Pre-Trained Model

|R|S|S<sub>pr1|S<sub>pr2|S<sub>pr3|
|:-:|:-:|:-:|:-:|:-:|
|Real Dataset|Synthetic Dataset|Synthetic Image Translation Dataset (one)|Synthetic Image Translation Dataset (two)| Synthetic Image Translation (three)|

\+ meaning is mixed dataset


## Training and Inference

traininig and Inference code reference and execute [this code](./train.py)

testset is available [this folder](./test_dataset)


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
