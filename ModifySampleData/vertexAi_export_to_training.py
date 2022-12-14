from PIL import Image
import json

# jsonl data item format: {'imageGcsUri': 'gs://beam_large_batch/Big Bear_2022_05_19_00_11_06.png', 'boundingBoxAnnotations': [{'displayName': 'Eagle_Adult', 'xMin': 0.7455410225921522, 'xMax': 0.8692033293697978, 'yMin': 0.35443037974683544, 'yMax': 0.6582278481012658, 'annotationResourceLabels': {'aiplatform.googleapis.com/annotation_set_name': '7091511349573844992'}}], 'dataItemResourceLabels': {}}


DATASET_REDUCED_MAP = {
    "Eagle_Adult": 1,
    "Eagle_Chick": 2,
    "Eagle_Egg": -1,
    "Eagle_Juvenile": 3,
    "Eagle_Unknown": -1,
    "Food_Bird": 4,
    "Food_Fish": 4,
    "Food_Mammal": 4,
    "Food_Reptile": 4,
    "Food_Unidentified": 4,
    "Invalid_Bad_Image": -1,
    "Invalid_Empty_Nest": -1,
}

vertex_jsonl = "/home/kamranzolfonoon/dev/eagle-images/beam_large_batch/annotations/export-data-BeamDataset-2022-12-11T14:39:16.532953Z/image_bounding_box/BeamDataset_iod-7091511349573844992/data-00001-of-00001.jsonl"
dest_json = "/home/kamranzolfonoon/dev/eagle-images/beam_large_batch/annotations/annotations.json"
image_dir = "/home/kamranzolfonoon/dev/eagle-images/beam_large_batch/"


def main():
    data_dict = {}

    with open(vertex_jsonl, "r") as f:
        data = [json.loads(line) for line in f]

    for i in range(len(data)):
        # Get relevant information
        filename = data[i]["imageGcsUri"].split("/")[-1]
        annotations = data[i]["boundingBoxAnnotations"]

        if len(annotations) == 0:
            continue

        # Get image size
        width, height = Image.open(f"{image_dir}/{filename}").size

        for annotation in annotations:
            # Get bounding box
            x_min = int(annotation["xMin"] * width)
            x_max = int(annotation["xMax"] * width)
            y_min = int(annotation["yMin"] * height)
            y_max = int(annotation["yMax"] * height)

            # Get label
            label = annotation["displayName"]
            if label not in DATASET_REDUCED_MAP:
                raise ValueError(f"Label {label} not found in DATASET_REDUCED_MAP")
            label = DATASET_REDUCED_MAP[label]
            if label == -1:
                continue

            if filename not in data_dict:
                data_dict[filename] = []
            # Add to dictionary
            data_dict[filename].append(
                {
                    "label": label,
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                }
            )

    with open(dest_json, "w") as f:
        json.dump(data_dict, f)


if __name__ == "__main__":
    main()
