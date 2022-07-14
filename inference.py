import requests
import layoutparser as lp
from numpy import NaN
import pandas as pd
import json


model = lp.Detectron2LayoutModel(
    config_path="./outputs/lastpage/fast_rcnn_R_50_FPN_3x/config.yaml",
    model_path="./outputs/lastpage/fast_rcnn_R_50_FPN_3x/model_final.pth",
    extra_config=[
        "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
        0.8,
    ],
)

with open("./data/annotations/test.json", mode="r") as coco_test_ds:
    coco_test_data = json.load(coco_test_ds)

# reduce list of category dictionaries into key value pairs
categories = coco_test_data["categories"]
category_names = {}
for category in categories:
    category_names[category["id"]] = category["name"]


categories_lookup = pd.DataFrame(
    category_names.items(), columns=["category_id", "name"]
)

r = requests.get(
    "https://cac.annauniv.edu/PhpProject1/aidetails/afug_2017_fu/01.%20B.E.EEE.pdf",
    verify=False,
)
with open("test.pdf", "wb") as fp:
    fp.write(r.content)

pdf_tokens, pdf_images = lp.load_pdf("test.pdf", load_images=True)


page_contents = {}

for page_index in range(len(pdf_images)):  # Pages with Subject-Level Data
    syllabus_items = model.detect(pdf_images[page_index])

    page_syllabus_tokens = []
    for syllabus_item in syllabus_items:
        syllabus_tokens = pdf_tokens[page_index].filter_by(syllabus_item, center=True)
        token_class = syllabus_item.type
        page_syllabus_tokens.append(tuple([token_class, syllabus_tokens]))

    # MAGIC - Sorted based on position coordinate. Check LayoutParser's Layout.filter_by()
    page_syllabus_tokens = sorted(
        page_syllabus_tokens, key=lambda tokens: min([ele.id for ele in tokens[1]])
    )
    page_contents[page_index] = [
        tuple([syllabus_tokens[0], " ".join(syllabus_tokens[1].get_texts())])
        for syllabus_tokens in page_syllabus_tokens
    ]


df = (
    pd.DataFrame(tuple(page_contents.items()), columns=["page_index", "extracted_data"])
    .explode("extracted_data")
    .reset_index(drop=True)
)

# Replace NaN values in extracted data with tuples for data uniformity
df["extracted_data"] = df["extracted_data"].apply(
    lambda x: (NaN, NaN) if x is NaN else x
)

# Split (category_id, data) tuples into separate columns
df = df.join(
    pd.DataFrame(df["extracted_data"].tolist(), columns=["category_id", "data"])
)


df = df.merge(categories_lookup, on="category_id", how="left")

df[["page_index", "category_id", "name", "data"]].to_csv("./test_export.csv")
