
## Dynamic Degradation Decomposition Network for All-in-One Image Restoration

PyTorch implementation for All-In-One Image Restoration for Unknown Corruption (D$^3$Net)

## 🔧Dependencies and Installation

- Python == 3.8.12 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- Pytorch ==2.0.0
- Option: Linux

### Installation

You can install the environment dependencies for D$^3$net using the following command.

1. Clone repo

   ```bash
   git clone https://github.com/Anonymous0515/D3Net
   cd D3Net
   ```
2. Install dependent packages

   ```bash
   pip install -r requirements.txt
   ```

## 🏰Datasets

You could find the dataset we used in the paper at following:

Denoising: [BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing), [WED](https://ece.uwaterloo.ca/~k29ma/exploration/)

Deraining: [Rain200L](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)

Dehazing: [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) (OTS)

Deblurring: [GoPro](https://seungjunnah.github.io/Datasets/gopro)

Low-light enhancement: [LOL](https://daooshee.github.io/BMVC2018website/)

## 💻Training

We provide the training code for D$^3$Net (used in our paper). You could improve it according to your own needs.

**Procedures**

1. Dataset preparation
2. Download training dataset. Put them in the `data/train` folder.
3. Training

   If you want to train our model, you can use the following command:

   ```
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --epoch 0 --n_epochs 2000 --train_datasets your_Datasets --model_folder your_model_folder
   ```

   You also can refer to more arguments in `tran.py` to modify the training command according to your needs. In addition, we default to training under five image degradation conditions. If you want to add more image degradations, please modify `utils/dataloader.py`.

## ⚡Testing

If you want to test our model and obtain PSNR and SSIM, you first need to fill in `your_testImagePath`, `your_savePath`, `your_test_Epoch_Number`, `your_model_folder`, and `your_GT_ImagePath`. Then, you can use the following command:

```
python test.py --image_path your_testImagePath --save_path your_savePath --epoch your_test_Epoch_Number --model_folder your_model_folder --target_data_dir your_GT_ImagePath
```

## 🤗 Acknowledgement

We borrow some code from [basicSR](https://github.com/XPixelGroup/BasicSR/tree/master), [RDN](https://github.com/yulunzhang/RDN), and [HorNet](https://github.com/raoyongming/HorNet). We gratefully thank the authors for their excellent works！
