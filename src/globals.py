import ast
import os
import sys

import supervisely as sly
from supervisely.io.fs import mkdir
from pathlib import Path

from dotenv import load_dotenv

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

my_app = sly.AppService()
api: sly.Api = my_app.public_api


def str_to_list(data):
    data = ast.literal_eval(data)
    data = [n.strip() for n in data]
    return data

TASK_ID = os.environ["TASK_ID"]
TEAM_ID = int(os.environ["context.teamId"])
WORKSPACE_ID = int(os.environ["context.workspaceId"])

COCO_MODE = os.environ.get("modal.state.cocoDataset")
META = sly.ProjectMeta()

INPUT_DIR = os.environ.get("modal.state.slyFolder")
INPUT_FILE = os.environ.get("modal.state.slyFile")

OUTPUT_PROJECT_NAME = os.environ.get("modal.state.projectName", "")

STORAGE_DIR = os.path.join(my_app.data_dir, "storage_dir")
mkdir(STORAGE_DIR, True)
COCO_BASE_DIR = os.path.join(STORAGE_DIR, "coco_base_dir")
mkdir(COCO_BASE_DIR)
SLY_BASE_DIR = os.path.join(STORAGE_DIR, "supervisely")
mkdir(SLY_BASE_DIR)

img_dir = None
ann_dir = None
src_img_dir = None
dst_img_dir = None

if COCO_MODE == "original":
    is_original = True
    original_ds = str_to_list(os.environ["modal.state.originalDataset"])
else:
    is_original = False
    custom_ds = None
    if INPUT_DIR is None and INPUT_FILE is None:
        FILES = os.environ.get("modal.state.files")
        sly.logger.info(f"Trying to find files in uploaded files: {FILES}")
        files = FILES.get("uploadedFiles")
        if files is None:
            raise RuntimeError("Please upload files or specify directory")
        files = str_to_list(files)
        if len(files) == 1 and sly.fs.get_file_ext(files[0]["path"]) in ["tar", "zip"]:
            INPUT_FILE = files[0]["path"]
        else:
            paths = [n["path"] for n in files]
            INPUT_DIR = Path(os.path.commonpath(paths))

    if INPUT_DIR:
        custom_ds = INPUT_DIR
        sly.logger.info(f"Copying files from directory: {custom_ds}")
    elif INPUT_FILE:
        custom_ds = INPUT_FILE
        sly.logger.info(f"Extracting archive: {custom_ds}")
    

images_links = {
    "train2014": "http://images.cocodataset.org/zips/train2014.zip",
    "val2014": "http://images.cocodataset.org/zips/val2014.zip",
    "test2014": "http://images.cocodataset.org/zips/test2014.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "test2017": "http://images.cocodataset.org/zips/test2017.zip",
}

annotations_links = {
    "trainval2014": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "trainval2017": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}
