import os
import json
import supervisely_lib as sly
from supervisely_lib.io.fs import mkdir

my_app = sly.AppService()
api: sly.Api = my_app.public_api

task_id = os.environ["TASK_ID"]
team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])

coco_mode = os.environ["modal.state.cocoDataset"]
meta = None

storage_dir = os.path.join(my_app.data_dir, "storage_dir")
mkdir(storage_dir, True)
coco_base_dir = os.path.join(storage_dir, "coco_base_dir")
mkdir(coco_base_dir)
sly_base_dir = os.path.join(storage_dir, "supervisely")
mkdir(sly_base_dir)

original_ds = os.environ['modal.state.originalDataset']
original_ds = json.loads(original_ds)

custom_ds = os.environ['modal.state.customDataset']

images_links = {
         "train2014": "http://images.cocodataset.org/zips/train2014.zip",
         "val2014": "http://images.cocodataset.org/zips/val2014.zip",
         "test2014": "http://images.cocodataset.org/zips/test2014.zip",
         "train2017": "http://images.cocodataset.org/zips/train2017.zip",
         "val2017": "http://images.cocodataset.org/zips/val2017.zip",
         "test2017": "http://images.cocodataset.org/zips/test2017.zip"
}

annotations_links = {
         "trainval2014": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
         "trainval2017": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}
