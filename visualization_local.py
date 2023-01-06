import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np

# If checkpoint is saved, load it:
# model = LitCNN.load_from_checkpoint('/home/kamranzolfonoon/dev/eaglescnn/lightning_logs/version_0/checkpoints/epoch=29-step=3540.ckpt')
# img = image

def visualize_image_local(
    model,
    img,
    imgName,
    target=None,
    prediction_path=None,
    ground_truth_path=None,
    threshold=0.5
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)
    model.eval()

    img_cuda = img.to(device)
    prediction = model([img_cuda])[0]

    predictionData = {"Labels_Num": prediction['labels'].tolist(), "Scores": prediction['scores'].tolist()}
    predictionData = pd.DataFrame(predictionData)

    #error handling step if no predictions made
    if len(predictionData) == 0:
        predictionData['Labels_Num'] = 0
        predictionData['Scores'] = 0
        predictionData = pd.DataFrame(predictionData,index=np.arange(1))

    #convert labels to meaningful labels
    predictionData.loc[predictionData['Labels_Num'] == 1, 'Labels_Full'] = 'Eagle_Adult' 
    predictionData.loc[predictionData['Labels_Num'] == 2, 'Labels_Full'] = 'Eagle_Chick' 
    predictionData.loc[predictionData['Labels_Num'] == 3, 'Labels_Full'] = 'Eagle_Juvenile'
    predictionData.loc[predictionData['Labels_Num'] == 4, 'Labels_Full'] = 'Food'

    #add image name
    predictionData['Image_Name'] = imgName

    labels = []
    boxes = []
    for i in range(len(prediction["labels"])):
        if prediction["scores"][i] > threshold:
            labels.append(
                str(prediction["labels"][i].item())
                + " "
                + str(round(prediction["scores"][i].item(), 3))
            )
            boxes.append(prediction["boxes"][i].cpu())
    if boxes != []:
        boxes = torch.stack(boxes)
    else:
        boxes = torch.tensor([])

    # Draw predictions:
    prediction = draw_bounding_boxes(
        torch.tensor(img * 255, dtype=torch.uint8),
        boxes=boxes,
        labels=labels,
        colors="red",
        width=4,
        font="/usr/local/texlive/2016/texmf-dist/fonts/truetype/intel/clearsans/ClearSans-Bold.ttf",
        font_size=20
    )
    im = to_pil_image(prediction.detach())
    im.save(prediction_path)

    if ground_truth_path is not None:
        labels = []
        boxes = []
        for i in range(len(target["labels"])):
            labels.append(str(target["labels"][i].item()))
            boxes.append(target["boxes"][i].cpu())
        if boxes != []:
            boxes = torch.stack(boxes)
        else:
            boxes = torch.tensor([])

        ground_truth = draw_bounding_boxes(
            torch.tensor(img * 255, dtype=torch.uint8),
            boxes=boxes,
            labels=labels,
            colors="blue",
            width=4,
            font="/usr/local/texlive/2016/texmf-dist/fonts/truetype/intel/clearsans/ClearSans-Bold.ttf",
            font_size=20
        )
        im = to_pil_image(ground_truth.detach())
        im.save(ground_truth_path)
    
    return(predictionData)