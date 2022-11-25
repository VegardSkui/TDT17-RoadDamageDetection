# Road Damage Detection

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
