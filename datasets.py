import torch
import os
import glob
import json
from PIL import Image
from utils.utils import SAMPLE_MAP

class SmallDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, annotations_file, transforms=None):
        self.data_folder = os.path.normpath(data_folder)
        self.annotations_file = os.path.normpath(annotations_file)
        self.transforms = transforms

        self.imgs = list(sorted(glob.glob(os.path.join(self.data_folder, "*.jpg"))))

        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        assert len(self.imgs) == len(self.annotations), "Number of images and annotations do not match"
    
    def relative_path(self, index):
        return self.imgs[index].removeprefix(self.data_folder + os.sep)

    def __getitem__(self, index):

        img = Image.open(self.imgs[index]).convert("RGB")
        objects = self.annotations[self.relative_path(index)]

        boxes = []
        labels = []
        for i in range(len(objects)):
            labels.append(SAMPLE_MAP[objects[i][0]])

            xmin = int(objects[i][3])
            xmax = int(objects[i][4])
            ymin = int(objects[i][5])
            ymax = int(objects[i][6])
            boxes.append([xmin, ymin, xmax, ymax])
        
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    dataset = SmallDataset("/home/kamranzolfonoon/eagle-test-bucket", 
    "/home/kamranzolfonoon/eagle-test-bucket/image_labels.json")
    img, target = dataset[2]

    # Draw bounding boxes on the image using pil
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    for box in target["boxes"]:
        draw.rectangle(box.tolist(), outline="red")
    img.save("test.jpg")


        