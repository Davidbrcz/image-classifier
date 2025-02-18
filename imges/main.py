from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset
from PIL import Image

image = Image.open('/home/dcome/source/tmp/imges/dataset/2020/IMG_20200620_115605.jpg')

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
inputs = processor(image, return_tensors="pt")

model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

with torch.no_grad():
    outputs = model(**inputs)
    # print(outputs)
    logits = outputs.logits

predicted_label = logits.argmax(-1)
print("label item = ", predicted_label.item())

probas = torch.nn.functional.softmax(logits, dim=1)
# print(probas)
topValues, topIndex = torch.topk(probas, 10)
print("topValues=", topValues , ";;topIdx=", topIndex)

for (p,i) in zip(topValues[:][0],topIndex[:][0]):
    print(f"is {model.config.id2label[i.item()]} [p={p}]")
