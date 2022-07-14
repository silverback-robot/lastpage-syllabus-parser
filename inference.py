# %%
import requests
import layoutparser as lp

# %%
r = requests.get(
    "https://cac.annauniv.edu/PhpProject1/aidetails/afug_2017_fu/01.%20B.E.EEE.pdf",
    verify=False,
)
with open("test.pdf", "wb") as fp:
    fp.write(r.content)

# %%
pdf_tokens, pdf_images = lp.load_pdf("test.pdf", load_images=True)

# %%
model = lp.Detectron2LayoutModel(
    config_path="./outputs/lastpage/fast_rcnn_R_50_FPN_3x/config.yaml",
    model_path="./outputs/lastpage/fast_rcnn_R_50_FPN_3x/model_final.pth",
    extra_config=[
        "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
        0.8,
    ],  # <-- Only output high accuracy preds
)

# %%
# layout = model.detect(pdf_images[24])

# %%
# lp.draw_box(pdf_images[24], layout)

# %%
# layout_data = layout.to_dataframe()
# layout_data

# %% [markdown]
# #### Sort Page Contents Based on Relative Positions

# %%
page_contents = {}

for page_index in range(len(pdf_images)):  # Pages with Subject-Level Data
    syllabus_items = model.detect(pdf_images[page_index])

    page_syllabus_tokens = []
    for syllabus_item in syllabus_items:
        syllabus_tokens = pdf_tokens[page_index].filter_by(syllabus_item, center=True)
        page_syllabus_tokens.append(syllabus_tokens)

    # MAGIC - Sorted based on position coordinate. Check LayoutParser's Layout.filter_by()
    page_syllabus_tokens = sorted(
        page_syllabus_tokens, key=lambda tokens: min([ele.id for ele in tokens])
    )

    page_contents[page_index] = [
        " ".join(syllabus_tokens.get_texts())
        for syllabus_tokens in page_syllabus_tokens
    ]


# %%
# import pandas as pd

# (
#     pd.DataFrame(tuple(page_contents.items()), columns=["page_index", "extracted_data"])
#     .explode("extracted_data")
#     .reset_index(drop=True)
# )[:30]

# %%
# page_contents

# %%
page_contents_mod = {}

for page_index in range(len(pdf_images)):  # Pages with Subject-Level Data
    syllabus_items = model.detect(pdf_images[page_index])

    page_syllabus_tokens = []
    for syllabus_item in syllabus_items:
        syllabus_tokens = pdf_tokens[page_index].filter_by(syllabus_item, center=True)
        token_class = (
            syllabus_item.type
        )  # Include COCO Category ID for downstream data processing
        page_syllabus_tokens.append(tuple([token_class, syllabus_tokens]))

    # MAGIC - Sorted based on position coordinate. Check LayoutParser's Layout.filter_by()
    page_syllabus_tokens = sorted(
        page_syllabus_tokens, key=lambda tokens: min([ele.id for ele in tokens[1]])
    )  # tokens[0] contains category ID and tokens[1] contains the actual tokens

    page_contents_mod[page_index] = [
        tuple([syllabus_tokens[0], " ".join(syllabus_tokens[1].get_texts())])
        for syllabus_tokens in page_syllabus_tokens
    ]


# %% [markdown]
# #### Split tuples of (category_id, data) into separate columns
#
# This way, category_id can be joined with coco dataset to obtain correct category names

# %%
from numpy import NaN
import pandas as pd

# Split each of the page's bbox-contents into a separate record
df = (
    pd.DataFrame(
        tuple(page_contents_mod.items()), columns=["page_index", "extracted_data"]
    )
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

# Data preview
df[20:40]


# %% [markdown]
# Load category_id mappings from coco dataset used for training

# %%
import json

# read test dataset (smaller size)
with open("./data/annotations/test.json", mode="r") as coco_test_ds:
    coco_test_data = json.load(coco_test_ds)

# reduce list of category dictionaries into key value pairs
categories = coco_test_data["categories"]
category_names = {}
for category in categories:
    category_names[category["id"]] = category["name"]

# category_names

# %% [markdown]
# Convert categories dict into Pandas DataFrame

# %%
categories_lookup = pd.DataFrame(
    category_names.items(), columns=["category_id", "name"]
)

# categories_lookup

# %% [markdown]
# Update master dataframe with category names based on category_id

# %%

df = df.merge(categories_lookup, on="category_id", how="left")

# %% [markdown]
# Create subset of master with only the required columns

# %%
df_minimal = df[["page_index", "category_id", "name", "data"]]

# df_minimal[:20]

# %%
df_minimal.to_csv("./test_export.csv")
