import pandas as pd
import numpy as np
import cv2

from src.bbox_tools.nms_algorithms import old_nms

df_path = "/home/riedlinger/MetaDetect-TestEvaluation/005481.csv"
df = pd.read_csv(df_path).drop("Unnamed: 0", axis=1)
CLASSES = ("Car", "Van", "Truck", "Pedestrian",
           "Person", "Cyclist", "Tram", "Misc")


def draw_boxes(image, prediction):
    for pred in prediction:
        color = (255, 0, 0)
        box = np.array(pred[:4], dtype=int)
        score = pred[4]
        c = CLASSES[int(pred[-1])]

        cv2.rectangle(image, (box[0], box[1]),
                      (box[2], box[3]), color=color, thickness=1)
        string = "{} {:.2}".format(c, score)
        text_size = cv2.getTextSize(string, 0, 0.35, thickness=1)[0]
        cv2.rectangle(image, (box[0], box[1]-text_size[1]),
                      (box[0]+text_size[0], box[1]), color, -1)
        cv2.putText(image, string, (box[0], box[1] - 1), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.35, color=(0, 0, 0), thickness=1)

    return image


def predict_from_csv(dataframe, classes, threshold=0.3):
    sel_by_score = dataframe[dataframe["s"] >= threshold]
    boxes = np.array(
        sel_by_score[["xmin", "ymin", "xmax", "ymax", "s", "category_idx"]])
    nms_res = old_nms(boxes, 0.45)

    s = ""
    for arr in nms_res:
        s = s + "{} {:.4} {} {} {} {}\n".format(classes[int(arr[5])], float(
            arr[4]), int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3]))

    with open("/home/riedlinger/MetaDetect-TestEvaluation/005481.txt", "w") as file:
        file.write(s)

    image = cv2.imread(dataframe.loc[0, "file_path"])
    image = draw_boxes(image, nms_res)
    cv2.imwrite("/home/riedlinger/MetaDetect-TestEvaluation/005481.png", image)


predict_from_csv(df, CLASSES)
