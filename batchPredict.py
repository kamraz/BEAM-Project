from lightning import LitCNN
from utils.visualization_local import visualize_image_local
from utils.transforms import get_transform
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import glob

def batchPredict(modelChkpt, imagesPath,saveImages, savePredictions):

    #new df to save results
    saveDF= pd.DataFrame()

    #get list of all images from HD
    imageList = glob.glob(imagesPath + '/**/*', recursive=True)
    #remove subdirectories
    imageList = imageList[7:]

    #for loop to gather images
    for image in imageList:
        
        img_path = image
        print(img_path)
        img_path_split= img_path.split(sep="/")
        img_path_len = len(img_path_split)-1
        img_name_1= img_path_split[img_path_len]
        img_name_1_split = img_name_1.split(sep=".")
        image_name = img_name_1_split[0]
        output_path = saveImages + "/"+image_name+"_labelled.png"

        transforms = get_transform(train=False)
        image,_ = transforms(Image.open(img_path).convert("RGB"), {})

        # define model - use full path!
        model_checkpoint = modelChkpt
        model = LitCNN.load_from_checkpoint(model_checkpoint)

        newPredictData = visualize_image_local(model=model, img=image, imgName=image_name, prediction_path=output_path, threshold=0.50)

        saveDF=saveDF.append(newPredictData)
    
    saveDF.to_csv(savePredictions+"eagle_nest_cam_predictions.csv", index=False)
    return(saveDF)


#run funtion
eagleNestCamPredictions = batchPredict(modelChkpt = "/Users/zach/ZachTeam Dropbox/Zach Ladin/Projects/Eagle_Nest_Cam/Kamran_Final/Model_checkpoint/final_model.ckpt", imagesPath = "/Volumes/Seagate Bac/Nest_Cam_Imagery/", saveImages = "/Volumes/Seagate Bac/Nest_Cam_Imagery_labelled", savePredictions = "/Volumes/Seagate Bac/Data/")
