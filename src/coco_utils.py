import json


class COCOUtils:
    def __init__(self, ann):
        self.info = ann["info"]
        self.licenses = ann["licenses"]
        self.images = ann["images"]
        self.annotations = ann["annotations"]
        self.categories = ann["categories"]

    def get_description(self):
        info_json = json.dumps(self.info, indent=0, sort_keys=False)
        pretty_info = (
            info_json.replace("{", "")
            .replace('"', "")
            .replace("}", "")
            .replace(",", "")
            .replace("[", "")
            .replace("]", "")
        )
        return str(pretty_info)

    def get_dataset_info(self):
        ann_info = self.info
        ann_licenses = self.licenses
        img_licenses = list(ann_licenses)
        ann_info["licenses"] = img_licenses
        return ann_info
