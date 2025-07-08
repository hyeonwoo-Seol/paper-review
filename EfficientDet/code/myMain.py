import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2
import os
from PIL import Image
import numpy as np
from torchvision.ops import box_iou
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Swish Acvitation Function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Depthwise Separable Convolution
class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias = False)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = True)
        self.bn = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 1e-3)
        self.swish = Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.swish(x)
        return x


# BiFPN Layer
class BiFPNLayer(nn.Module):
    def __init__(self, num_channels, epsilon = 1e-4):
        super().__init__()
        self.epsilon = epsilon

        self.w1 = nn.Parameter(torch.ones(2, dtype = torch.float32), requires_grad = True)
        self.w1_relu = nn.ReLU()
        self.w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2_relu = nn.ReLU()
        self.w3 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w3_relu = nn.ReLU()
        self.w4 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w4_relu = nn.ReLU()

        self.w5 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.w5_relu = nn.ReLU()
        self.w6 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.w6_relu = nn.ReLU()
        self.w7 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.w7_relu = nn.ReLU()
        self.w8 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w8_relu = nn.ReLU()

        self.conv_up = nn.ModuleList([SeparableConvBlock(num_channels, num_channels) for _ in range(4)])
        self.conv_down = nn.ModuleList([SeparableConvBlock(num_channels, num_channels) for _ in range(4)])

        self.upsample = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.downsample = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, inputs):
        P3_in, P4_in, P5_in, P6_in, P7_in = inputs

        # Top down
        w1 = self.w1_relu(self.w1)
        w1 = w1 / (torch.sum(w1, dim = 0) + self.epsilon)
        P6_td = self.conv_up[0](w1[0] * P6_in + w1[1] * self.upsample(P7_in))

        w2 = self.w2_relu(self.w2); w2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)
        P5_td = self.conv_up[1](w2[0] * P5_in + w2[1] * self.upsample(P6_td))
        
        w3 = self.w3_relu(self.w3); w3 = w3 / (torch.sum(w3, dim=0) + self.epsilon)
        P4_td = self.conv_up[2](w3[0] * P4_in + w3[1] * self.upsample(P5_td))
        
        w4 = self.w4_relu(self.w4); w4 = w4 / (torch.sum(w4, dim=0) + self.epsilon)
        P3_out = self.conv_up[3](w4[0] * P3_in + w4[1] * self.upsample(P4_td))
        
        # Bottom up
        w5 = self.w5_relu(self.w5)
        w5 = w5 / (torch.sum(w5, dim=0) + self.epsilon)
        P4_out = self.conv_down[0](w5[0] * P4_in + w5[1] * P4_td + w5[2] * self.downsample(P3_out))

        w6 = self.w6_relu(self.w6); w6 = w6 / (torch.sum(w6, dim=0) + self.epsilon)
        P5_out = self.conv_down[1](w6[0] * P5_in + w6[1] * P5_td + w6[2] * self.downsample(P4_out))
        
        w7 = self.w7_relu(self.w7); w7 = w7 / (torch.sum(w7, dim=0) + self.epsilon)
        P6_out = self.conv_down[2](w7[0] * P6_in + w7[1] * P6_td + w7[2] * self.downsample(P5_out))
        
        w8 = self.w8_relu(self.w8); w8 = w8 / (torch.sum(w8, dim=0) + self.epsilon)
        P7_out = self.conv_down[3](w8[0] * P7_in + w8[1] * self.downsample(P6_out))
        return [P3_out, P4_out, P5_out, P6_out, P7_out]


# BiFPN
class BiFPN(nn.Module):
    def __init__(self, num_channels, num_repeats, backbone_out_channels):
        super().__init__()
        self.num_repeats = num_repeats

        self.in_convs = nn.ModuleList()
        for out_channels in backbone_out_channels:
            self.in_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, num_channels, kernel_size = 1, stride = 1, padding = 0), 
                    nn.BatchNorm2d(num_channels, momentum = 0.01, eps = 1e-3)
                    )
                )

        self.p6_conv = nn.Sequential(
            nn.Conv2d(backbone_out_channels[-1], num_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3)
        )
        self.p7_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3)
        )

        self.bifpn_layers = nn.ModuleList([BiFPNLayer(num_channels) for _ in range(num_repeats)])

        self.p6_downsample = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.p7_downsample = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

    def forward(self, backbone_feats):
        p3, p4, p5 = backbone_feats

        p3_in = self.in_convs[0](p3)
        p4_in = self.in_convs[1](p4)
        p5_in = self.in_convs[2](p5)

        p6_temp = self.p6_conv(p5)
        p6_in = self.p6_downsample(p6_temp)

        p7_temp = self.p7_conv(p6_in)
        p7_in = self.p7_downsample(p7_temp)

        features = [p3_in, p4_in, p5_in, p6_in, p7_in]

        for i in range(self.num_repeats):
            features = self.bifpn_layers[i](features)

        return features

# BoxNet
class BoxNet(nn.Module):
    def __init__(self, in_channels, num_layers, num_anchors = 9):
        super().__init__()
        self.num_layers = num_layers

        # 반복되는 공유 컨볼루션 레이어
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels) for _ in range(num_layers)]
            )

        # 최종 박스 예측을 위한 헤더 컨볼루션
        self.header = nn.Conv2d(in_channels, num_anchors * 4, kernel_size = 3, padding = 1)

    def forward(self, inputs):
        feats = []

        for feat in inputs:
            for conv in self.conv_list:
                feat = conv(feat)

            # 헤더를 통해 예측하기
            box_pred = self.header(feat)
            # 출력을 [batch, h, w, c] 형태로 변경하기
            box_pred = box_pred.permute(0, 2, 3, 1)
            # [batch, h*w*num_anchors, 4] 형태로 최종 변환하기
            feats.append(box_pred.reshape(box_pred.shape[0], -1, 4))

        feats = torch.cat(feats, dim = 1)

        return feats

    
# ClassNet
class ClassNet(nn.Module):
    def __init__(self, in_channels, num_layers, num_classes, num_anchors = 9):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # 반복되는 공유 컨볼루션 레이어
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels) for _ in range(num_layers)]
        )

        # 최종 클래스 예측을 위한 헤더 컨볼루션
        self.header = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        feats = []
        for feat in inputs:
            for conv in self.conv_list:
                feat = conv(feat)

            # 헤더를 통해 예측
            class_pred = self.header(feat)
            
            # 출력을 [Batch, H, W, C] 형태로 변경
            class_pred = class_pred.permute(0, 2, 3, 1)
            # [Batch, H*W*num_anchors, num_classes] 형태로 최종 변환
            feats.append(class_pred.reshape(class_pred.shape[0], -1, self.num_classes))

        # 모든 레벨의 예측 결과를 하나로 합치기
        feats = torch.cat(feats, dim = 1)
        # 시그모이드 활성화 함수를 적용하기
        feats = self.sigmoid(feats)

        return feats


# EfficientDet
class EfficientDet(nn.Module):
    def __init__(self, num_classes, D_bifpn, W_bifpn, D_class, model_name='efficientnet_b0'):
        super().__init__()
        
        # EfficientDet-D0의 P3, P4, P5 출력 채널
        backbone_out_channels = [40, 80, 192]
        
        self.backbone = EfficientNetBackbone(model_name=model_name)
        self.bifpn = BiFPN(num_channels=W_bifpn, 
                           num_repeats=D_bifpn, 
                           backbone_out_channels=backbone_out_channels)
        self.class_net = ClassNet(in_channels=W_bifpn, 
                                  num_layers=D_class, 
                                  num_classes=num_classes)
        self.box_net = BoxNet(in_channels=W_bifpn, 
                                num_layers=D_class)

    def forward(self, x):
        # 1. Backbone
        features = self.backbone(x)
        
        # 2. BiFPN
        features = self.bifpn(features)
        
        # 3. Prediction Head
        class_outputs = self.class_net(features)
        box_outputs = self.box_net(features)
        
        return class_outputs, box_outputs


# EfficientNet Backbone
class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super().__init__()

        # torchvision의 EfficientNet 불러오기
        backbone = getattr(torchvision.models, model_name)(weights = 'IMAGENET1K_V1' if pretrained else None)

        # efficientNet-b0의 구조를 출력했을 때, 3, 4, 6이 논문에서 나오는 구조에 적합하다.
        self.p3_layer = backbone.features[3]
        self.p4_layer = backbone.features[4]
        self.p5_layer = backbone.features[6]

        # Backbone의 전체 레이어 저장
        self.features = backbone.features


    def forward(self, x):
        outputs = []

        for i, layer in enumerate(self.features):
            x = layer(x)

            if i == 3:
                outputs.append(x)
            elif i == 4:
                outputs.append(x)
            elif i == 6:
                outputs.append(x)

        return outputs


# DataSet (KITTI)
class KittiDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        # 이미지 파일 목록을 정렬해서 순서를 보장하기
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

        # 클래스 이름을 정수 인덱스로 매핑하기
        self.class_to_idx = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 'Person_sitting': 4,
                             'Cyclist': 5, 'Tram': 6, 'Misc': 7}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 이미지 불러오기
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        image = np.array(image)

        # 라벨 불러오기
        label_name = img_name.replace('.png', '.txt')
        label_path = os.path.join(self.label_dir, label_name)

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split(' ')
                    class_name = parts[0]
                    if class_name in self.class_to_idx:
                        x1, y1, x2, y2 = [float(p) for p in parts[4:8]]
                        boxes.append([x1, y1, x2, y2])
                        labels.append(self.class_to_idx[class_name])
        if self.transform:
            # image, bboxes, labels를 함께 전달해서 변환을 동기화하기
            transformed = self.transform(image = image, bboxes = boxes, labels = labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
                                          
        target = {}
        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor([int(l) for l in labels], dtype=torch.int64)
        else: # 빈 텐서를 전달합니다.
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            
        return image, target

    
# Collate FN
def collate_fn(batch):
    # 배치 안에 있는 샘플의 크기가 다를 때 이를 처리한다.
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # 이미지들을 하나의 텐서로 스택하기
    images = torch.stack(images, 0)
    return images, targets


# Anchors
class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()
        self.pyramid_levels = pyramid_levels if pyramid_levels is not None else [3, 4, 5, 6, 7]
        self.strides = strides if strides is not None else [2 ** x for x in self.pyramid_levels]
        self.sizes = sizes if sizes is not None else [2 ** (x + 2) for x in self.pyramid_levels]
        self.ratios = ratios if ratios is not None else np.array([0.5, 1, 2])
        self.scales = scales if scales is not None else np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        all_anchors = np.zeros((0, 4)).astype(np.float32)
        for idx, p in enumerate(self.pyramid_levels):
            anchors = self._generate_anchors_at_level(self.sizes[idx], self.ratios, self.scales)
            shifted_anchors = self._shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
        all_anchors = np.expand_dims(all_anchors, axis=0)
        return torch.from_numpy(all_anchors.astype(np.float32)).to(image.device)

    def _generate_anchors_at_level(self, base_size=16, ratios=None, scales=None):
        num_anchors = len(ratios) * len(scales)
        anchors = np.zeros((num_anchors, 4))
        anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
        areas = anchors[:, 2] * anchors[:, 3]
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
        return anchors

    def _shift(self, shape, stride, anchors):
        shift_x = (np.arange(0, shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, shape[0]) + 0.5) * stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        A = anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        return all_anchors


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, classifications, regressions, anchors, annotations):
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]
        dtype = anchors.dtype

        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            bbox_annotation = annotations[j]['boxes'].to(dtype)
            labels_annotation = annotations[j]['labels']

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().to(classifications.device))
                classification_losses.append(torch.tensor(0).float().to(classifications.device))
                continue

            iou = box_iou(anchor, bbox_annotation)
            iou_max, iou_argmax = torch.max(iou, dim=1)

            targets = torch.ones(classification.shape, dtype=dtype) * -1
            targets = targets.to(classifications.device)
            targets[torch.lt(iou_max, 0.4), :] = 0
            positive_indices = torch.ge(iou_max, 0.5)
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[iou_argmax, :]
            targets[positive_indices, :] = 0
            targets[positive_indices, labels_annotation[iou_argmax[positive_indices]].long()] = 1

            alpha_factor = torch.ones_like(targets) * self.alpha
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            bce = -(targets * torch.log(torch.clamp(classification, 1e-6, 1.0)) + \
                    (1.0 - targets) * torch.log(torch.clamp(1.0 - classification, 1e-6, 1.0)))
            
            cls_loss = focal_weight * bce
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros_like(cls_loss))
            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            if num_positive_anchors > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]
                anchor_widths = anchor[:, 2] - anchor[:, 0]
                anchor_heights = anchor[:, 3] - anchor[:, 1]
                anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
                anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x[positive_indices]) / anchor_widths[positive_indices]
                targets_dy = (gt_ctr_y - anchor_ctr_y[positive_indices]) / anchor_heights[positive_indices]
                targets_dw = torch.log(gt_widths / anchor_widths[positive_indices])
                targets_dh = torch.log(gt_heights / anchor_heights[positive_indices])

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh)).t()
                regression_diff = torch.abs(targets - regression[positive_indices, :])
                
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().to(classifications.device))

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True)

# RUN
if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    # EfficientDet-D0 설정값
    W_BIFPN = 64  # BiFPN 너비(채널 수)
    D_BIFPN = 3   # BiFPN 반복 횟수
    D_CLASS = 3   # Class/Box Net 깊이
    
    NUM_CLASSES = 8
    INPUT_SIZE = 512

    EPOCHS = 50
    BATCH_SIZE = 10
    EARLY_STOPPING_PATIENCE = 10
    epochs_no_improve = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터 준비 및 Albumentation 증강
    train_transform = A.Compose([
        A.Resize(height = INPUT_SIZE, width = INPUT_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params = A.BboxParams(format='pascal_voc', label_fields=['labels']))

    val_transform = A.Compose([
        A.Resize(height = INPUT_SIZE, width = INPUT_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    # 데이터셋 인스턴스 생성
    train_dataset = KittiDataset(image_dir='./KIT/training/image_2', label_dir='./KIT/training/label_2', transform = train_transform)
    val_dataset = KittiDataset(image_dir='./KIT/validation/image_2', label_dir='./KIT/validation/label_2', transform=val_transform)

    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn = collate_fn, num_workers = 4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers = 4)

    # 모델, 손실함수, 옵티마이저 정의
    model = EfficientDet(num_classes=NUM_CLASSES, W_bifpn=W_BIFPN, D_bifpn=D_BIFPN, D_class=D_CLASS).to(device)

    # 앵커 생성 및 손실 함수 정의
    anchors = Anchors().to(device)
    criterion = FocalLoss().to(device)

    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-5)


    # 스케줄러 정의하기
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor = 0.5, patience = 2, verbose = True)
    

    

    # 체크포인트 불러오기
    start_epoch = 0
    best_val_loss = float('inf')

    # 저장할 디렉터리 생성
    save_dir = "mySave"
    os.makedirs(save_dir, exist_ok = True)

    checkpoint_path = os.path.join(save_dir, "best_checkpoint.pth")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}. Best validation loss so far: {best_val_loss:.4f}")
    else:
        print("No checkpoint found. Starting from scratch.")
        

    print("Setup complete. Starting training...")
    # 학습 검증
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_train_loss = 0
        
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            class_preds, box_preds = model(images)
            anchor_boxes = anchors(images)
            cls_loss, reg_loss = criterion(class_preds, box_preds, anchor_boxes, targets)
            loss = cls_loss + reg_loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN or Inf loss detected. Skipping update.")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if (i + 1) % 250 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], "
                      f"Total Loss: {loss.item():.4f}, Cls Loss: {cls_loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}")

        avg_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_loss:.4f}")

        # 검증 단계
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                class_preds, box_preds = model(images)
                anchor_boxes = anchors(images)
                cls_loss, reg_loss = criterion(class_preds, box_preds, anchor_boxes, targets)
                loss = cls_loss + reg_loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Average Validation Loss: {avg_val_loss:.4f}")


        # 스케줄러 실행하기
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving best model...")
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, checkpoint_path)

    print("Training finished.")
