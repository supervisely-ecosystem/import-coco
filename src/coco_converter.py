import os
import json
import shutil
import numpy as np
import globals as g
from PIL import Image
import supervisely_lib as sly
from coco_utils import COCOUtils
import pycocotools.mask as mask_util
from supervisely_lib.io.fs import mkdir, file_exists


def create_sly_meta_from_coco_categories(coco_categories):
    meta = sly.ProjectMeta()
    colors = []
    for category in coco_categories:
        new_color = sly.color.generate_rgb(colors)
        colors.append(new_color)
        obj_class = sly.ObjClass(category["name"], sly.Polygon, new_color)
        meta = meta.add_obj_class(obj_class)
    return meta


def get_sly_meta_from_coco(coco_dataset, dataset_name):
    path_to_meta = os.path.join(g.sly_base_dir, "meta.json")
    if not os.path.exists(os.path.join(g.sly_base_dir, "meta.json")):
        g.meta = create_sly_meta_from_coco_categories(coco_dataset.categories)
        meta_json = g.meta.to_json()
        sly.json.dump_json_file(meta_json, path_to_meta)
        return g.meta
    else:
        if dataset_name in ["train2014", "val2014", "train2017", "val2017"]:
            return g.meta
        else:
            meta = create_sly_meta_from_coco_categories(coco_dataset.categories)
            g.meta = g.meta.merge(meta)
            meta_json = g.meta.to_json()
            sly.json.dump_json_file(meta_json, path_to_meta)
            return g.meta


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
    for coco_ann in coco_annotations:
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
    mkdir(dataset_dir)
    img_dir = os.path.join(dataset_dir, "img")
    mkdir(img_dir)
    ann_dir = os.path.join(dataset_dir, "ann")
    mkdir(ann_dir)
    return dataset_dir


def move_trainvalds_to_sly_dataset(dataset, sly_dataset_dir, coco_image, ann):
    img_dir = os.path.join(sly_dataset_dir, "img")
    ann_dir = os.path.join(sly_dataset_dir, "ann")
    image_name = coco_image['file_name']
    ann_json = ann.to_json()
    sly.json.dump_json_file(ann_json, os.path.join(ann_dir, f"{image_name}.json"))
    coco_img_path = os.path.join(g.coco_base_dir, dataset, "images", image_name)
    sly_img_path = os.path.join(img_dir, image_name)
    if file_exists(os.path.join(coco_img_path)):
        shutil.move(coco_img_path, sly_img_path)


def move_testds_to_sly_dataset(dataset, coco_base_dir, sly_dataset_dir):
    src_img_dir = os.path.join(coco_base_dir, dataset, "images")
    dst_img_dir = os.path.join(sly_dataset_dir, "img")
    ann_dir = os.path.join(sly_dataset_dir, "ann")

    ds_progress = sly.Progress(f"Converting dataset: {dataset}", len(os.listdir(src_img_dir)), min_report_percent=1)
    for image in os.listdir(src_img_dir):
        src_image_path = os.path.join(src_img_dir, image)
        dst_image_path = os.path.join(dst_img_dir, image)
        shutil.move(src_image_path, dst_image_path)
        im = Image.open(dst_image_path)
        width, height = im.size
        img_size = [height, width]
        ann = sly.Annotation(img_size)
        ann_json = ann.to_json()
        sly.json.dump_json_file(ann_json, os.path.join(ann_dir, f"{image}.json"))
        ds_progress.iter_done_report()


def check_dataset_for_annotation(dataset):
    ann_dir = os.path.join(g.coco_base_dir, dataset, "annotations")
    if os.path.exists(ann_dir) and len(os.listdir(ann_dir)) != 0:
        return True
    else:
        return False


def read_coco_dataset(dataset_name):
    coco_ann_dir = os.path.join(g.coco_base_dir, dataset_name, "annotations")
    path_to_coco_ann = os.path.join(coco_ann_dir, os.listdir(coco_ann_dir)[0])
    coco_ann_json = open(path_to_coco_ann)
    coco_ann = json.load(coco_ann_json)
    if "annotations" not in coco_ann.keys():
        return None
    coco_ann = COCOUtils(coco_ann)
    return coco_ann