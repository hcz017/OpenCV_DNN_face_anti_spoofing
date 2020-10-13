import os
import torch

pt_model_path = "train/checkpoints/FeatherNet54/2020-09-23-16_57_36__17_best.pth"
onnx_path = "train/checkpoints/FeatherNet54/2020-09-23-16_57_36__17_best.pth.onnx"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(pt_model_path, map_location=device)

model.eval()

batch_size = 1
input_shape = (3, 224, 224)
input_data_shape = torch.randn(batch_size, *input_shape, device=device)

torch.onnx.export(model, input_data_shape, onnx_path, verbose=True)
