import torch

ckpt = torch.load("runs/train/yolov5s_nc4/weights/best.pt", map_location="cpu", weights_only=False)
model = ckpt['model']
anchors = model.model[-1].anchors  # Detect layer

anchors_pixel = anchors.clone()
for i, s in enumerate([8, 16, 32]):
    anchors_pixel[i] *= s

print("Extracted anchors:")
print(anchors_pixel)
