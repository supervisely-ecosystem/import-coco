import json
import numpy as np
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

    def convert_to_sly(self, api, project):
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project.id))
        img_info_map = {}
        ann_map = {}
        labels = []
        tag_metas = []
        for img, img_categories in zip(self.images, self.categories): #TODO 3 must be 4
            ann_labels = []
            for img_ann in self.annotations:
                if img["id"] != img_ann["image_id"] and img_categories["id"] != img_ann["category_id"]:
                    continue
                # TAGS
                iscrowd_tag_meta = sly.TagMeta("iscrowd", sly.TagValueType.ANY_NUMBER)
                iscrowd_tag = sly.Tag(iscrowd_tag_meta, img_ann["iscrowd"])
                supercategory_tag_meta = sly.TagMeta("supercategory", sly.TagValueType.ANY_STRING)
                supercategory_tag = sly.Tag(supercategory_tag_meta, img_categories["supercategory"])
                categoryID_tag_meta = sly.TagMeta("category id", sly.TagValueType.ANY_NUMBER)
                categoryID_tag = sly.Tag(categoryID_tag_meta, img_categories["id"])
                if iscrowd_tag_meta.name not in [tag_meta.name for tag_meta in tag_metas]:
                    tag_metas.append(iscrowd_tag_meta)
                if supercategory_tag_meta.name not in [tag_meta.name for tag_meta in tag_metas]:
                    tag_metas.append(supercategory_tag_meta)
                if categoryID_tag_meta.name not in [tag_meta.name for tag_meta in tag_metas]:
                    tag_metas.append(categoryID_tag_meta)
                label_tags = sly.TagCollection([iscrowd_tag, supercategory_tag, categoryID_tag])
                # TAGS

                is_RLE = False
                if type(img_ann["segmentation"]) is dict:
                    is_RLE = True

                if is_RLE is False:
                    for polygons in img_ann["segmentation"]:
                        exterior = polygons
                        exterior = [exterior[i * 2:(i + 1) * 2] for i in range((len(exterior) + 2 - 1) // 2)]
                        exterior = [sly.PointLocation(height, width) for width, height in exterior]
                        figure = sly.Polygon(exterior, [], img_ann["id"])
                        obj_class = sly.ObjClass(img_categories["name"], sly.Polygon)

                        img_label = sly.Label(figure, obj_class, label_tags)
                        ann_labels.append(img_label)
                        if img_label.obj_class.name not in [label.obj_class.name for label in labels]:
                            labels.append(img_label)
                else:
                    continue
                    rleObj = mask_util.frPyObjects(img_ann["segmentation"], img_ann["segmentation"]["size"][0], img_ann["segmentation"]["size"][1])
                    mask = mask_util.decode(rleObj)
                    mask = np.array(mask, dtype=bool)
                    figure = sly.Bitmap(mask)

                    img_class = sly.ObjClass(img_categories["name"], sly.Bitmap)
                    img_label = sly.Label(figure, img_class, label_tags)
                    ann_labels.append(img_label)
                    if img_label.obj_class.name not in [label.obj_class.name for label in labels]:
                        labels.append(img_label)

                img_info_map[img["file_name"]] = img
                ann_map[img["file_name"]] = sly.Annotation((img["height"], img["width"]), ann_labels)

        project_meta = project_meta.add_obj_classes([label.obj_class for label in labels])
        project_meta = project_meta.add_tag_metas(tag_metas)
        api.project.update_meta(project.id, project_meta.to_json())
        return img_info_map, ann_map
