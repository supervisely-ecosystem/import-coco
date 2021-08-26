import os
import json
import shutil
import numpy as np
import globals as g
import supervisely_lib as sly
import pycocotools.mask as mask_util
from coco_utils import COCOUtils
from supervisely_lib.io.fs import mkdir

# create meta from categories
# convert coco anns to sly anns


def create_sly_meta_from_coco_categories(coco_categories):
    meta = sly.ProjectMeta()
    colors = []
    for category in coco_categories:
        new_color = sly.color.generate_rgb(colors)
        colors.append(new_color)
        obj_class = sly.ObjClass(category["name"], sly.Polygon, new_color)
        meta = meta.add_obj_class(obj_class)
    path_to_meta = os.path.join(g.sly_base_dir, "meta.json")
    meta_json = meta.to_json()
    sly.json.dump_json_file(meta_json, path_to_meta)
    return meta


def get_coco_annotations_for_current_image(coco_image, coco_anns):
    image_id = coco_image["id"]
    image_annotations = []
    for coco_ann in coco_anns:
        if image_id == coco_ann["image_id"]:
            image_annotations.append(coco_ann)
    return image_annotations


def coco_category_to_class_name(coco_categories):
    name_cat_id_map = {}
    for category in coco_categories:
        name_cat_id_map[category["id"]] = category["name"]
    return name_cat_id_map


def convert_polygon_vertices(coco_ann):
    for polygons in coco_ann["segmentation"]:
        exterior = polygons
        exterior = [exterior[i * 2:(i + 1) * 2] for i in range((len(exterior) + 2 - 1) // 2)]
        exterior = [sly.PointLocation(height, width) for width, height in exterior]
        figure = sly.Polygon(exterior, [])
        return figure


def convert_rle_mask_to_polygon(coco_ann):
    rle_obj = mask_util.frPyObjects(coco_ann["segmentation"], coco_ann["segmentation"]["size"][0],
                                    coco_ann["segmentation"]["size"][1])
    mask = mask_util.decode(rle_obj)
    mask = np.array(mask, dtype=bool)
    polygons = sly.Bitmap(mask).to_contours()
    return polygons


def create_sly_ann_from_coco_annotation(meta, coco_categories, coco_annotations, image_size):
    name_cat_id_map = coco_category_to_class_name(coco_categories)
    labels = []
    for coco_ann in coco_annotations: #16
        obj_class = meta.get_obj_class(name_cat_id_map[coco_ann["category_id"]])
        if type(coco_ann["segmentation"]) is dict:
            polygons = convert_rle_mask_to_polygon(coco_ann)
            for polygon in polygons:
                figure = polygon
                label = sly.Label(figure, obj_class)
                labels.append(label)
        else:
            figure = convert_polygon_vertices(coco_ann)
            label = sly.Label(figure, obj_class)
            labels.append(label)
    ann = sly.Annotation(image_size, labels)
    return ann


def create_sly_dataset_dir(dataset_name):
    dataset_dir = os.path.join(g.sly_base_dir, dataset_name)
    img_dir = os.path.join(dataset_dir, "img")
    ann_dir = os.path.join(dataset_dir, "ann")
    mkdir(dataset_dir)
    mkdir(img_dir)
    mkdir(ann_dir)
    return dataset_dir


def move_to_sly_dataset(dataset, sly_dataset_dir, coco_image, ann):
    img_dir = os.path.join(sly_dataset_dir, "img")
    ann_dir = os.path.join(sly_dataset_dir, "ann")
    image_name = coco_image['file_name']
    ann_json = ann.to_json()
    sly.json.dump_json_file(ann_json, os.path.join(ann_dir, f"{image_name}.json"))
    coco_img_path = os.path.join(g.coco_base_dir, dataset, "images", image_name)
    sly_img_path = os.path.join(img_dir, image_name)
    shutil.move(coco_img_path, sly_img_path)


def read_coco_dataset(dataset_name):
    coco_ann_dir = os.path.join(g.coco_base_dir, dataset_name, "annotations")
    path_to_coco_ann = os.path.join(coco_ann_dir, os.listdir(coco_ann_dir)[0])
    coco_ann_json = open(path_to_coco_ann)
    coco_ann = json.load(coco_ann_json)
    coco_ann = COCOUtils(coco_ann)
    return coco_ann
