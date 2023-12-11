
Download datasets from the following links and place them in `data/train`. Your directory tree should look like this

```
Denoise
  ├──BSD
  └──WED

GOPRO
  ├──train
  └──test

LOL
  ├──eval15
  └──our485

Rain200L
  ├──train
  └──test

RESIDE
  ├──OTS
     ├──clear
     |	└──clear
     └──haze
	├──part1
	├──part2
	├──part3
	└──part4
```

In the paper, we set up the training dataset according to the following directory structure. If you want to adjust the degradation combinations or add more degradation types, please modify `utils/dataloader.py`.
