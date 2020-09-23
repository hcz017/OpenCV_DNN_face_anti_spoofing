import torch
import os

save_path = "train/checkpoints/FeatherNet54"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.load('filename.pth').to(device)
model_path=os.path.join(save_path, "2020-09-15-12_01_14__10_best.pth")
model = torch.load(model_path, map_location=device)
model.eval()
batch_size = 1
input_shape = (3, 224, 224)

input_data_shape = torch.randn(batch_size, *input_shape, device=device)

save_onnx_dir = os.path.join(save_path, "best.pth.onnx")
torch.onnx.export(model, input_data_shape, save_onnx_dir, verbose=True)