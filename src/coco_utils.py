import os
import json
import numpy as np
import globals as g
import supervisely_lib as sly
import pycocotools.mask as mask_util


class COCOUtils:
    def __init__(self, ann):
        self.info = ann["info"]
        self.licenses = ann["licenses"]
        self.images = ann["images"]
        self.annotations = ann["annotations"]
        self.categories = ann["categories"]

    def get_description(self):
        info_json = json.dumps(self.info, indent=0, sort_keys=False)
        pretty_info = info_json.replace("{", "").replace("\"", "").replace("}", "").replace(",", "").replace("[", "").replace("]", "")
        return str(pretty_info)

    def get_dataset_info(self):
        ann_info = self.info
        ann_licenses = self.licenses
        img_licenses = []
        for license in ann_licenses:
            img_licenses.append(license)
        ann_info["licenses"] = img_licenses
        return ann_info
