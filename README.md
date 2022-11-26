# Road Damage Detection

Road damage detection using YOLOv7 ([GitHub Repo](https://github.com/WongKinYiu/yolov7), [Paper](https://arxiv.org/abs/2207.02696)) for the Norwegian dataset in the [Crowdsensing-based Road Damage Detection Challenge (CRDDC2022)](https://crddc2022.sekilab.global/).

## Model training

The following command was used in the `yolov7` directory to train the final model.
```sh
python -m torch.distributed.launch \
    --nproc_per_node 2 --master_port 9527 train_aux.py \
    --workers 14 \
    --device 0,1 \
    --sync-bn \
    --batch-size 16 \
    --data data/norway-nitti-ll-1856.yaml \
    --cfg cfg/training/yolov7-w6-norway.yaml \
    --weights "" \
    --img 1856 1856 \
    --name w6-nitti-p6-1856 \
    --hyp data/hyp.scratch.p6.custom.yaml \
    --epoch 200 \
    --cache-images \
    --label-smoothing 0.1
```

## Detection

Code to run detection using the trained model and produce a final file suitable for submission in CRDDC2022 can be found in the [`detect.ipynb`](detect.ipynb)-notebook.
