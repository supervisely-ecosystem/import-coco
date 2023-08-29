import ast
import os
import shutil

import supervisely as sly
from supervisely.io.fs import mkdir
from os.path import basename, dirname, normpath

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
SLY_SELECTED_CONTEXT = os.environ.get("modal.state.slySelectedContext")
META = sly.ProjectMeta()

INPUT_DIR = os.environ.get("modal.state.slyFolder")
INPUT_FILE = os.environ.get("modal.state.slyFile")

if SLY_SELECTED_CONTEXT != "ecosystem":
    COCO_MODE = "custom"

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
    if SLY_SELECTED_CONTEXT == "ecosystem":
        files = os.environ.get("modal.state.files")
        if files is not None and files != "":
            sly.logger.info(f"Trying to find files in uploaded files: {files}")
            ext = sly.fs.get_file_ext(files.rstrip("/"))
            if ext in [".tar", ".zip"]:
                INPUT_FILE = files
            else:
                INPUT_DIR = files

    if INPUT_DIR:
        listdir = api.file.listdir(TEAM_ID, INPUT_DIR)
        if len(listdir) == 1 and sly.fs.get_file_ext(listdir[0]) in [".zip", ".tar"]:
            sly.logger.warn("Folder mode is selected, but archive file is uploaded.")
            sly.logger.info("Switching to file mode.")
            INPUT_DIR, INPUT_FILE = None, os.path.join(INPUT_DIR, listdir[0])
        elif any(basename(normpath(x)) in ["images", "annotations"] for x in listdir):
            INPUT_DIR = dirname(normpath(INPUT_DIR))
            sly.logger.info(f"INPUT_DIR: {INPUT_DIR}")
        elif basename(normpath(INPUT_DIR)) in ["images", "annotations"]:
            INPUT_DIR = dirname(dirname(normpath(INPUT_DIR)))
            sly.logger.info(f"INPUT_DIR: {INPUT_DIR}")
    elif INPUT_FILE:
        available_archive_formats = list(zip(*shutil.get_archive_formats()))[0]
        file_ext = sly.fs.get_file_ext(INPUT_FILE)
        if file_ext.lstrip(".") in available_archive_formats:
            sly.logger.info(f"Input file is an {file_ext} archive")
        elif file_ext in sly.image.SUPPORTED_IMG_EXTS + [".json"]:
            parent_dir = dirname(normpath(INPUT_FILE))
            listdir = api.file.listdir(TEAM_ID, parent_dir)
            if basename(normpath(parent_dir)) in ["images", "annotations"]:
                parent_dir = dirname(dirname(parent_dir))
            elif ["images", "annotations"] in [basename(normpath(x)) for x in listdir]:
                parent_dir = dirname(parent_dir)
            if not parent_dir.endswith("/"):
                parent_dir += "/"
            INPUT_DIR, INPUT_FILE = parent_dir, None
        else:
            raise ValueError(
                "Incorrect project structure. "
                f"File mode is chosen, but file {INPUT_FILE} is not an archive. "
                "Please, read apps overview and prepare the dataset correctly."
            )

    if INPUT_DIR:
        custom_ds = INPUT_DIR
        sly.logger.info(f"INPUT_DIR: {custom_ds}")
    elif INPUT_FILE:
        custom_ds = INPUT_FILE
        sly.logger.info(f"INPUT_FILE: {custom_ds}")
    

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
