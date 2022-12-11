import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fcos_resnet50_fpn,
    FCOS_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(
    num_classes: int,
    frozen: bool = False,
    model_name: str = "fasterrcnn_resnet50_fpn",
    weights_path=None,
):

    if model_name == "fcos_resnet50_fpn":
        if weights_path is not None:
            pass
        else:
            model = fcos_resnet50_fpn(
                weights=FCOS_ResNet50_FPN_Weights.DEFAULT, num_classes=num_classes
            )
    elif model_name == "fasterrcnn_resnet50_fpn":
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        if frozen:
            # TODO: Investigate more re: what to freeze
            for param in model.parameters():
                param.requires_grad = False
        num_classes = num_classes + 1  # +1 for background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # TODO: Implement v2 with vision transformers
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    return model


if __name__ == "__main__":
    model = get_model(num_classes=10)
    print(model)
