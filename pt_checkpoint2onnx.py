import os
import torch
import train.models as models

pt_checkpoint_path = "train/checkpoints/FeatherNet54/2020-09-23-16_57_36__17_best.pth.tar"
onnx_path = "train/checkpoints/FeatherNet54/2020-09-23-16_57_36__17_best.pth.tar.onnx"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.__dict__["FeatherNet54"]()

# python train_FeatherNet.py --b 32 --lr 0.01 --every-decay 60 
# refer to train/cfgs/FeatherNet.yaml
optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9,
                                weight_decay=0.0001)

checkpoint = torch.load(pt_checkpoint_path)
epoch = checkpoint['epoch']
arch = checkpoint['arch']
best_prec1 = checkpoint['best_prec1']
optimizer.load_state_dict(checkpoint['optimizer'])
model.load_state_dict(checkpoint['state_dict'])

model.eval()

batch_size = 1
input_shape = (3, 224, 224)
input_data_shape = torch.randn(batch_size, *input_shape, device=device)

torch.onnx.export(model, input_data_shape, onnx_path, verbose=True)
