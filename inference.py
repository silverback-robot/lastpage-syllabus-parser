import os
from pathlib import Path
import requests
import layoutparser as lp
from numpy import NaN
import pandas as pd
import json
import logging
import warnings


# create and configure logger
log = logging.getLogger("lastpage-syllabus-extractor")
ch = logging.StreamHandler()
log.addHandler(ch)
formatter = logging.Formatter("%(asctime)s : %(name)s [%(levelname)s] : %(message)s")
ch.setFormatter(formatter)
log.setLevel(logging.DEBUG)
ch.setLevel(logging.DEBUG)

# Ignore warnings (for lack of better ideas)
log.warn("Suppressing all warnings!")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load custom layout detection model
model = lp.Detectron2LayoutModel(
    config_path="./outputs/lastpage/fast_rcnn_R_50_FPN_3x/config.yaml",
    model_path="./outputs/lastpage/fast_rcnn_R_50_FPN_3x/model_final.pth",
    extra_config=[
        "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
        0.8,
    ],
)

# prepare working directories
def setup_directories():
    log.info("Checking data directories")
    app_dir = Path.cwd()
    data_dir = app_dir.joinpath("data")
    if data_dir.is_dir():
        log.info("Found existing 'data' directory. Checking subdirectories.")
        if not data_dir.joinpath("pdf").is_dir():
            log.info("'data/pdf' subdirectory not found. Creating...")
            data_dir.joinpath("pdf").mkdir()
        elif any(os.scandir(data_dir.joinpath("pdf"))):
            log.warning("'data/pdf' is NOT empty. Clearing contents of 'data/pdf' now.")
            [f.unlink() for f in data_dir.joinpath("pdf").glob("*") if f.is_file()]
        else:
            log.info("'data/pdf' directory exists and is empty. No actions performed.")

        if not data_dir.joinpath("csv").is_dir():
            log.info("'data/csv' subdirectory not found. Creating...")
            data_dir.joinpath("csv").mkdir()
        elif any(os.scandir(data_dir.joinpath("csv"))):
            log.warning("'data/csv' is NOT empty. Clearing contents of 'data/csv' now.")
            [f.unlink() for f in data_dir.joinpath("csv").glob("*") if f.is_file()]
        else:
            log.info("'data/csv' directory exists and is empty. No actions performed.")
    else:
        log.info("'data' directory not found. Creating required directory tree")
        data_dir.mkdir()
        data_dir.joinpath("pdf").mkdir()
        data_dir.joinpath("csv").mkdir()


def load_categories(coco_json: str) -> dict:
    # Extract category_id and names (static data, required for post-processing)
    # with open("./data/annotations/test.json", mode="r") as coco_test_ds:
    log.info(f"Loading categories from COCO dataset: {coco_json}")
    with open(coco_json, mode="r") as coco_test_ds:
        coco_test_data = json.load(coco_test_ds)

    # reduce list of category dictionaries into key value pairs
    categories = coco_test_data["categories"]
    category_names = {}
    for category in categories:
        category_names[category["id"]] = category["name"]

    # Load category map into a DataFrame
    categories_lookup = pd.DataFrame(
        category_names.items(), columns=["category_id", "name"]
    )

    log.info("COCO dataset categories loaded successfully")
    return categories_lookup


# Download PDF from URL
def download_pdf(pdf_url: str, data_dir: str, ssl_verify=False) -> str:
    requests.packages.urllib3.disable_warnings()  # Suppress SSL warns as this is expected
    filename = pdf_url.split("/")[-1].upper().replace("%20", "")
    target_filename = data_dir.joinpath("pdf").joinpath(filename).as_posix()
    log.info(f"Downloading {pdf_url} to {target_filename}")

    r = requests.get(
        # "https://cac.annauniv.edu/PhpProject1/aidetails/afug_2017_fu/01.%20B.E.EEE.pdf",
        pdf_url,
        verify=ssl_verify,
    )
    log.info(f"Writing to disk: {target_filename}")
    with open(target_filename, "wb") as fp:
        fp.write(r.content)

    log.info(f"Finished downloading: {target_filename}")
    return target_filename


def data_extractor(pdf_file_name: str) -> dict:
    log.info(f"Extracting tokens and images from {pdf_file_name}")
    pdf_tokens, pdf_images = lp.load_pdf(pdf_file_name, load_images=True)
    page_contents = {}

    for page_index in range(len(pdf_images)):  # Pages with Subject-Level Data
        syllabus_items = model.detect(pdf_images[page_index])

        page_syllabus_tokens = []
        for syllabus_item in syllabus_items:
            syllabus_tokens = pdf_tokens[page_index].filter_by(
                syllabus_item, center=True
            )
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

    return page_contents


def post_processing(page_contents: dict, categories_map: dict) -> pd.DataFrame:
    log.info("Performing post-processing actions...")
    df = (
        pd.DataFrame(
            tuple(page_contents.items()), columns=["page_index", "extracted_data"]
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

    df = df.merge(categories_map, on="category_id", how="left")

    log.info("Post-processing completed.")
    return df


def export_data(df: pd.pd.DataFrame, csv_file_name: str, pdf_file_name: str):
    app_dir = Path.cwd()
    csv_dir = app_dir.joinpath("data").joinpath("csv").as_posix
    df[["page_index", "category_id", "name", "data"]].to_csv(
        f"{csv_dir}/{pdf_file_name}.csv"
    )
