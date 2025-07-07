KITTI 2D 데이터셋을 다운 받았습니다.
12gb 용량이고, 왼쪽 카메라 학습 이미지 (training/image_2)와 라벨 (training/label_2)를 받았습니다.

TFRecord 변환을 했습니다.
EfficientDet 코드가 TFRecord 포멧만 지원하기 때문에, create_kitti_tf_record.py 스크립트를 통해 .png + .txt 를 *.record로 변환했습니다.

문제 발생
변환 스크립트 내부에서 from object_detection.utils import dataset_util를 사용하는데, TensorFlow-Models Object Detection API가 설치되어 있지 않아서, git clone https://github.com/tensorflow/models.git 를 통해 model/research 를 다운받았습니다.

하지만 PYTHONPATH 설정을 실패했고, .bashrc와 activate.d 설정 값도 실패를 해서, create_kitti_tf_record.py 최상단에 다음과 같은 코드를 추가했습니다.
```
import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'models', 'research'))
sys.path.insert(0, os.path.join(ROOT, 'models', 'research', 'slim'))
```

이 코드는 스크립트가 실행되는 순간, models/research 와 models/research/slim 디렉터리를 파이썬 모듈 검색 경로(sys.path)로 추가하게 됩니다.

OOM 문제가 발생해서, batch_size를 4로 낮췄습니다.

fin-tuning을 진행했더니 box_loss가 0으로 고정되는 문제가 발생했습니다. 그래서 이를 확인하는 코드(inspect_tfrecord.py)를 실행했더니, efficientdet은 박스 좌표가 [0.0,1.0] 범위로 정규화되어 있어야 하는데 제가 사용한 방법은 px 단위로 박스 위치가 되어 있어서 손실 값이 0으로 고정되는 현상이 발생했습니다.

따라서 create_kitti_tf_record.py 내부에 코드 중 width와 height로 나눠서 0,1 범위로 변환하고자 합니다.

d4 모델로 fine-tuning을 시도했으나 배치 사이즈를 2로 설정했음에도 불구하고 OOM 문제가 발생하였습니다.
이에 d2 모델로 낮추고 배치 사이즈도 d0 모델의 절반인 2로 설정했을 때 메모리를 최대로 사용하면서 fine-tuning이 가능했습니다.

fine-tuning 첫 1000 step의 경우 loss=0.25, box_loss=0.001, cls_loss=0.1, det_loss=0.15 수준이었지만, step이 점점 늘어날 수록 loss 값이 증가하는 현상을 확인했습니다.
2000 step은 l=0.46, b_l=0.003, c_l=0.18, d_l=0.3 
3000 step은 l=0.38, b_l=0.002, c_l=0.17, d_l=0.29 
4000 step은 l=0.77 b_l=0.005, c_l=0.30, d_l=0.56 
5000 step은 l=0.46, b_l=0.002, c_l=0.25, d_l=0.36
6000 step은 l=0.40, b_l=0.001, c_l=0.21, d_l=0.3

따라서 Learning rate 또는 warmup_steps를 줄여서 실험하고자 합니다.

추가적으로 배치정규화를 frozen 시켜서 학습시키고자 합니다. det_model_fn.py에서 is_training_bn=False로 고정시키거나 hparams에 freeze_bn=True를 사용할 수 있습니다.

현재 사용하고 있는 GPU 모델은 RTX 3060 TI 8GB 으로, 심각한 VRAM 부족 문제를 겪고 있습니다. 이를 완화하기 위해 입력 해상도를 낮추거나 Gradient Checkpointing을 사용할 수 있습니다.

Gradient Checkpointing은 순전파의 activaton을 모두 저장하지 않고 몇몇의 체크포인트 레이어의 출력만 저장해둔 뒤, 역전파 시점에서 다시 순전파하여 activation을 재생성하는 기법입니다. 메모리 사용량을 O(n)에서 O(sqrt(n))으로 줄일 수 있습니다.

Train 데이터셋을 tfrecord로 변환하기
```
python create_kitti_tf_record.py --kitti_root=./KIT/ --output_path=./finTfrecords/train.record --label_map_path=./label_map.txt --subset=training
```

Validation 데이터셋을 tfrecord로 변환하기
```
python create_kitti_tf_record.py --kitti_root=./KIT/ --output_path=./finTfrecords/val.record --label_map_path=./label_map.txt --subset=validation
```

텐서보드로 결과 확인하기
```
tensorboard --logdir=./savedmodel/exp_kitti_d0
```

학습+평가시키기
```
python main.py \
  --mode=train_and_eval \
  --model_name=efficientdet-d0 \
  --train_file_pattern=./finTfrecords/train.record \
  --val_file_pattern=./finTfrecords/val.record \
  --model_dir=./savedmodel/exp_kitti_d1/ \
  --ckpt=./efficientdet-d0/model \
  --train_batch_size=4 \
  --eval_batch_size=2 \
  --num_epochs=5 \
  --hparams="mixed_precision=True"
```

  평가만 하기
  ```
python main.py \
  --mode=eval \
  --model_name=efficientdet-d0 \
  --val_file_pattern=./finTfrecords/val.record \
  --model_dir=./savedmodel/exp_kitti_d1/ \
  --eval_batch_size=2
```

중단된 체크포인트부터 학습
```
python main.py \
  --mode=train_and_eval \
  --model_name=efficientdet-d0 \
  --train_file_pattern=./finTfrecords/train.record \
  --val_file_pattern=./finTfrecords/val.record \
  --model_dir=./savedmodel/exp_kitti_d0/ \
  --train_batch_size=4 \
  --eval_batch_size=2 \
  --num_epochs=10 \
  --num_examples_per_epoch=5985 \
  --hparams="mixed_precision=True"
```
