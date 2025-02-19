#!/usr/bin/env python

from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from PIL import Image
import sys
import os
from pathlib import Path

index = {
    337,    295,    357,    223,    285,    281,    202, 283, 285, 209, 234
}
files = [os.path.join(dirpath, f) for (dirpath, dirnames, filenames)
         in os.walk(sys.argv[1]) for f in filenames]

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")


def test(f,  debug=False):
    image = Image.open(f)
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_label = logits.argmax(-1)

    probas = torch.nn.functional.softmax(logits, dim=1)
    topValues, topIndex = torch.topk(probas, 10)

    for (p, i) in zip(topValues[:][0], topIndex[:][0]):
        # if debug:
        #     print(f"{f} has a PET in it [{model.config.id2label[i.item()]}] ;; {i}")
        path = Path(f)
        if i.item() in index:
            # print(f"{f} has a PET in it [{model.config.id2label[i.item()]}]")
            petsDir = os.path.join(path.parent, "pets")
            if not Path(petsDir).exists():
                os.mkdir(petsDir)
            dest = os.path.join(petsDir, path.name)
            os.rename(f, dest)
            print(f"{f}-> {dest}")
            break


# test(sys.argv[1],  True)

for f in files:
    if f.endswith("jpg"):
        test(f)
