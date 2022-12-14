import torch
import os
import glob
import json
from PIL import Image
from utils.constants import SAMPLE_MAP
from utils.transforms import get_transform

def collate_fn(batch):
    return tuple(zip(*batch))

class BeamDataset(torch.utils.data.Dataset):
    """
    Dataset class for the beam dataset created on gcloud
    """
    def __init__(self, data_folder, annotations_file, transforms=None):
        self.data_folder = os.path.normpath(data_folder)
        self.annotations_file = os.path.normpath(annotations_file)
        self.transforms = transforms
        
        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.imgs = list(sorted(self.annotations.keys()))

    def __getitem__(self, index):

        img = Image.open(os.path.join(self.data_folder, self.imgs[index])).convert("RGB")
        objects = self.annotations[self.imgs[index]]

        boxes = []
        labels = []
        for data in objects:

            labels.append(data["label"])

            xmin = int(data["x_min"])
            xmax = int(data["x_max"])
            ymin = int(data["y_min"])
            ymax = int(data["y_max"])
            boxes.append([xmin, ymin, xmax, ymax])
        
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class SmallDataset(torch.utils.data.Dataset):
    """
    Daatset class created for dataset from project proposal
    """
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
        
        # TODO: Handle transforms on target dict
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    dataset = BeamDataset("/home/kamranzolfonoon/dev/eagle-images/beam_large_batch", 
    "/home/kamranzolfonoon/dev/eagle-images/beam_large_batch/annotations/annotations.json", get_transform(train=False))
    img, target = dataset[2]
    print("done")
    # Draw bounding boxes on the image using pil
    # from PIL import ImageDraw
    # img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    # draw = ImageDraw.Draw(img)
    # for box in target["boxes"]:
    #     draw.rectangle(box.tolist(), outline="red")
    # img.save("test.jpg")


        