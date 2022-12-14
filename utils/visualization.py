import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

# If checkpoint is saved, load it:
# model = LitCNN.load_from_checkpoint('/home/kamranzolfonoon/dev/eaglescnn/lightning_logs/version_0/checkpoints/epoch=29-step=3540.ckpt')


def visualize_image(
    model, img, target=None, prediction_path="prediction.png", ground_truth_path=None
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)
    model.eval()

    # good example: 12
    # tricky: 5, 80
    img_cuda = img.to(device)
    prediction = model([img_cuda])[0]

    labels = []
    boxes = []
    for i in range(len(prediction["labels"])):
        if prediction["scores"][i] > 0.5:
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
        )
        im = to_pil_image(ground_truth.detach())
        im.save(ground_truth_path)
