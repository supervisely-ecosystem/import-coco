import glob
import os
import shutil

import numpy as np
import pycocotools.mask as mask_util
import supervisely as sly
from pycocotools.coco import COCO
from PIL import Image
from supervisely.io.fs import file_exists, mkdir
from typing import List

import globals as g

def add_tail(body:str, tail:str):
    if ' ' in body:
        return f"{body} {tail}"
    return f"{body}_{tail}"
    

def get_ann_types(coco: COCO) -> List[str]:
    ann_types = []

    sly.logger.info('Getting info about annotation types..')

    annotation_ids = coco.getAnnIds()
    if any('bbox' in coco.anns[ann_id] for ann_id in annotation_ids):
        ann_types.append('bbox')
    if any('segmentation' in coco.anns[ann_id] for ann_id in annotation_ids):
        ann_types.append('segmentation')

    return ann_types

def create_sly_meta_from_coco_categories(coco_categories, ann_types=None):
    colors = []
    for category in coco_categories:
        if category["name"] in [obj_class.name for obj_class in g.META.obj_classes]:
            continue
        new_color = sly.color.generate_rgb(colors)
        colors.append(new_color)

        obj_classes = []

        if ann_types is not None:
            if 'segmentation' in ann_types:
                obj_classes.append(
                    sly.ObjClass(category['name'], sly.Polygon, new_color)
                )
            if 'bbox' in ann_types:
                obj_classes.append(
                    sly.ObjClass(add_tail(category['name'], "bbox"), sly.Rectangle, new_color)
                )

        g.META = g.META.add_obj_classes(obj_classes)
    return g.META




def get_sly_meta_from_coco(coco_categories, dataset_name, ann_types=None):
    path_to_meta = os.path.join(g.SLY_BASE_DIR, "meta.json")
    if not os.path.exists(path_to_meta):
        g.META = dump_meta(coco_categories, path_to_meta, ann_types)
    elif dataset_name not in ["train2014", "val2014", "train2017", "val2017"]:
        g.META = dump_meta(coco_categories, path_to_meta, ann_types)
    return g.META


def dump_meta(coco_categories, path_to_meta, ann_types=None):
    g.META = create_sly_meta_from_coco_categories(coco_categories, ann_types)
    meta_json = g.META.to_json()
    sly.json.dump_json_file(meta_json, path_to_meta)
    return g.META


def get_coco_annotations_for_current_image(coco_image, coco_anns):
    image_id = coco_image["id"]
    return [coco_ann for coco_ann in coco_anns if image_id == coco_ann["image_id"]]


def coco_category_to_class_name(coco_categories):
    return {category["id"]: category["name"] for category in coco_categories}


def convert_polygon_vertices(coco_ann):
    polygons = coco_ann["segmentation"]
    if all(type(coord) is float for coord in polygons):
        polygons = [polygons]

    for polygon in polygons:
        exterior = polygon
        exterior = [exterior[i * 2 : (i + 1) * 2] for i in range((len(exterior) + 2 - 1) // 2)]
        exterior = [sly.PointLocation(height, width) for width, height in exterior]
        return sly.Polygon(exterior, [])


def convert_rle_mask_to_polygon(coco_ann):
    if type(coco_ann["segmentation"]["counts"]) is str:
        coco_ann["segmentation"]["counts"] = bytes(
            coco_ann["segmentation"]["counts"], encoding="utf-8"
        )
        mask = mask_util.decode(coco_ann["segmentation"])
    else:
        rle_obj = mask_util.frPyObjects(
            coco_ann["segmentation"],
            coco_ann["segmentation"]["size"][0],
            coco_ann["segmentation"]["size"][1],
        )
        mask = mask_util.decode(rle_obj)
    mask = np.array(mask, dtype=bool)
    return sly.Bitmap(mask).to_contours()


def create_sly_ann_from_coco_annotation(meta, coco_categories, coco_ann, image_size):
    labels = []
    for object in coco_ann:
        name_cat_id_map = coco_category_to_class_name(coco_categories)

        segm = object.get("segmentation")
        if segm is not None:
            obj_class_name_polygon = name_cat_id_map[object["category_id"]]
            obj_class_polygon = meta.get_obj_class(obj_class_name_polygon)

            if type(segm) is dict:
                polygons = convert_rle_mask_to_polygon(object)
                for polygon in polygons:
                    figure = polygon
                    label = sly.Label(figure, obj_class_polygon)
                    labels.append(label)
            elif type(segm) is list and object["segmentation"]:
                figure = convert_polygon_vertices(object)
                label = sly.Label(figure, obj_class_polygon)
                labels.append(label)

        bbox = object.get("bbox")
        if bbox is not None and len(bbox) >= 4:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = xmin + bbox[2]
            ymax = ymin + bbox[3]

            obj_class_name_rectangle = add_tail( name_cat_id_map[object["category_id"]], "bbox")
            obj_class_rectangle = meta.get_obj_class(obj_class_name_rectangle)
            rectangle = sly.Label(
                sly.Rectangle(top=ymin, left=xmin, bottom=ymax, right=xmax), obj_class_rectangle
            )
            labels.append(rectangle)
    return sly.Annotation(image_size, labels)


def create_sly_dataset_dir(dataset_name):
    dataset_dir = os.path.join(g.SLY_BASE_DIR, dataset_name)
    mkdir(dataset_dir)
    img_dir = os.path.join(dataset_dir, "img")
    mkdir(img_dir)
    ann_dir = os.path.join(dataset_dir, "ann")
    mkdir(ann_dir)
    return dataset_dir


def move_trainvalds_to_sly_dataset(dataset, coco_image, ann):
    image_name = coco_image["file_name"]
    ann_json = ann.to_json()
    sly.json.dump_json_file(ann_json, os.path.join(g.ann_dir, f"{image_name}.json"))
    coco_img_path = os.path.join(g.COCO_BASE_DIR, dataset, "images", image_name)
    sly_img_path = os.path.join(g.img_dir, image_name)
    if file_exists(os.path.join(coco_img_path)):
        shutil.move(coco_img_path, sly_img_path)


def move_testds_to_sly_dataset(dataset):
    ds_progress = sly.Progress(
        f"Converting dataset: {dataset}",
        len(os.listdir(g.src_img_dir)),
        min_report_percent=1,
    )
    for image in os.listdir(g.src_img_dir):
        src_image_path = os.path.join(g.src_img_dir, image)
        dst_image_path = os.path.join(g.dst_img_dir, image)
        shutil.move(src_image_path, dst_image_path)
        im = Image.open(dst_image_path)
        width, height = im.size
        img_size = (height, width)
        ann = sly.Annotation(img_size)
        ann_json = ann.to_json()
        sly.json.dump_json_file(ann_json, os.path.join(g.ann_dir, f"{image}.json"))
        ds_progress.iter_done_report()


def check_dataset_for_annotation(dataset_name, ann_dir, is_original):
    if is_original:
        ann_path = os.path.join(ann_dir, f"instances_{dataset_name}.json")
        return bool(os.path.exists(ann_path) and os.path.isfile(ann_path))
    else:
        ann_files = glob.glob(os.path.join(ann_dir, "*.json"))
        if len(ann_files) == 1:
            return True
        elif len(ann_files) > 1:
            sly.logger.warn(f"Found more than one .json annotation file in the {ann_dir} directory. Please, read apps overview and prepare the dataset correctly.")
        elif len(ann_files) == 0:
            sly.logger.info(f"Annotation file not found in {ann_dir}. Please, read apps overview and prepare the dataset correctly.")
        return False
        

def get_ann_path(ann_dir, dataset_name, is_original):
    if is_original:
        return os.path.join(ann_dir, f"instances_{dataset_name}.json")
    else:
        ann_files = glob.glob(os.path.join(ann_dir, "*.json"))
        return ann_files[0]
