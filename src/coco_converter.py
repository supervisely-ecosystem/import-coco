import glob
import os
import json
import shutil
import cv2
from copy import deepcopy

import numpy as np
import pycocotools.mask as mask_util
import supervisely as sly
from pycocotools.coco import COCO
from PIL import Image
from supervisely.io.fs import file_exists, mkdir
from typing import List

import globals as g


def check_high_level_coco_ann_structure(ann_path):
    with open(ann_path, 'r') as f:
        dataset = json.load(f)
        for key in ['annotations', 'images', 'categories']:
            if key not in dataset:
                raise Exception(f"[{key}] field is missing")
            if not isinstance(dataset[key], list):
                raise Exception(f"[{key}] field value must be a list of dicts")


def add_tail(body: str, tail: str):
    if " " in body:
        return f"{body} {tail}"
    return f"{body}_{tail}"


def get_ann_types(coco: COCO) -> List[str]:
    ann_types = []

    sly.logger.info("Getting info about annotation types..")

    annotation_ids = coco.getAnnIds()
    if any("bbox" in coco.anns[ann_id] for ann_id in annotation_ids):
        ann_types.append("bbox")
    if any("segmentation" in coco.anns[ann_id] for ann_id in annotation_ids):
        ann_types.append("segmentation")
    if any("caption" in coco.anns[ann_id] for ann_id in annotation_ids):
        ann_types.append("caption")

    return ann_types


def create_sly_meta_from_coco_categories(coco_categories, ann_types=None):
    colors = []
    tag_metas = []
    for category in coco_categories:
        if category["name"] in [obj_class.name for obj_class in g.META.obj_classes]:
            continue
        new_color = sly.color.generate_rgb(colors)
        colors.append(new_color)

        obj_classes = []

        if ann_types is not None:
            if "segmentation" in ann_types:
                obj_classes.append(sly.ObjClass(category["name"], sly.Polygon, new_color))
            if "bbox" in ann_types:
                obj_classes.append(
                    sly.ObjClass(add_tail(category["name"], "bbox"), sly.Rectangle, new_color)
                )

        g.META = g.META.add_obj_classes(obj_classes)
    if ann_types is not None and "caption" in ann_types:
        tag_metas.append(sly.TagMeta("caption", sly.TagValueType.ANY_STRING))
    g.META = g.META.add_tag_metas(tag_metas)
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


def convert_polygon_vertices(coco_ann, image_size):
    polygons = coco_ann["segmentation"]
    if all(type(coord) is float for coord in polygons):
        polygons = [polygons]
    if any(type(coord) is str for polygon in polygons for coord in polygon):
        return []

    exteriors = []
    for polygon in polygons:
        polygon = [polygon[i * 2 : (i + 1) * 2] for i in range((len(polygon) + 2 - 1) // 2)]
        exterior_points = [(width, height) for width, height in polygon]
        if len(exterior_points) == 0:
            continue
        exteriors.append(exterior_points)

    interiors = {idx: [] for idx in range(len(exteriors))}
    id2del = []
    for idx, exterior in enumerate(exteriors):
        temp_img = np.zeros(image_size + (3,), dtype=np.uint8)
        geom = sly.Polygon([sly.PointLocation(y, x) for x, y in exterior])
        geom.draw_contour(temp_img, color=[255, 255, 255])
        im = cv2.cvtColor(temp_img, cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        for idy, exterior2 in enumerate(exteriors):
            if idx == idy or idy in id2del:
                continue
            points_inside = [
                cv2.pointPolygonTest(contours[0], (x, y), False) > 0 for x, y in exterior2
            ]
            # if all list elements are True, then all points are inside or on contour
            if all(points_inside):
                interiors[idx].append(deepcopy(exteriors[idy]))
                id2del.append(idy)

    # remove contours from exteriors that are inside other contours
    for j in sorted(id2del, reverse=True):
        del exteriors[j]

    figures = []
    for exterior, interior in zip(exteriors, interiors.values()):
        exterior = [sly.PointLocation(y, x) for x, y in exterior]
        interior = [[sly.PointLocation(y, x) for x, y in points] for points in interior]
        figures.append(sly.Polygon(exterior, interior))

    return figures


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
    if not np.any(mask):
        return []
    return sly.Bitmap(mask).to_contours()


def create_sly_ann_from_coco_annotation(meta, coco_categories, coco_ann, image_size):
    labels = []
    imag_tags = []
    name_cat_id_map = coco_category_to_class_name(coco_categories)
    for object in coco_ann:
        category_id = object.get("category_id")
        if category_id is None:
            continue
        obj_class_name = name_cat_id_map.get(category_id)
        if obj_class_name is None:
            sly.logger.warn(f"Category with id {category_id} not found in categories list")
            continue

        segm = object.get("segmentation")
        curr_labels = []
        if segm is not None and len(segm) > 0:
            obj_class_polygon = meta.get_obj_class(obj_class_name)

            if type(segm) is dict:
                polygons = convert_rle_mask_to_polygon(object)
                for polygon in polygons:
                    figure = polygon
                    label = sly.Label(figure, obj_class_polygon)
                    curr_labels.append(label)
            elif type(segm) is list and object["segmentation"]:
                figures = convert_polygon_vertices(object, image_size)
                curr_labels.extend([sly.Label(figure, obj_class_polygon) for figure in figures])

        labels.extend(curr_labels)
        bbox = object.get("bbox")
        if bbox is not None and len(bbox) == 4:
            if not obj_class_name.endswith("bbox"):
                obj_class_name = add_tail(obj_class_name, "bbox")
            obj_class_rectangle = meta.get_obj_class(obj_class_name)
            if len(curr_labels) > 1:
                for label in curr_labels:
                    bbox = label.geometry.to_bbox()
                    labels.append(sly.Label(bbox, obj_class_rectangle))
            else:
                x, y, w, h = bbox
                rectangle = sly.Label(sly.Rectangle(y, x, y + h, x + w), obj_class_rectangle)
                labels.append(rectangle)

        caption = object.get("caption")
        if caption is not None:
            imag_tags.append(sly.Tag(meta.get_tag_meta("caption"), caption))

    return sly.Annotation(image_size, labels=labels, img_tags=imag_tags)


def create_sly_dataset_dir(dataset_name):
    dataset_dir = os.path.join(g.SLY_BASE_DIR, dataset_name)
    mkdir(dataset_dir)
    img_dir = os.path.join(dataset_dir, "img")
    mkdir(img_dir)
    ann_dir = os.path.join(dataset_dir, "ann")
    mkdir(ann_dir)
    return dataset_dir


def remove_empty_sly_dataset_dir(dataset_name):
    dataset_dir = os.path.join(g.SLY_BASE_DIR, dataset_name)
    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)


def move_trainvalds_to_sly_dataset(dataset, coco_image, ann):
    image_name = coco_image["file_name"]
    if "/" in image_name:
        image_name = os.path.basename(image_name)
    ann_json = ann.to_json()
    coco_img_path = os.path.join(g.src_img_dir, image_name)
    sly_img_path = os.path.join(g.img_dir, image_name)
    if file_exists(os.path.join(coco_img_path)):
        sly.json.dump_json_file(ann_json, os.path.join(g.ann_dir, f"{image_name}.json"))
        shutil.move(coco_img_path, sly_img_path)


def move_testds_to_sly_dataset(dataset, image_cnt):
    ds_progress = sly.Progress(
        f"Converting dataset: {dataset}",
        len(os.listdir(g.src_img_dir)),
        min_report_percent=1,
    )
    for image in os.listdir(g.src_img_dir):
        if not sly.image.has_valid_ext(image):
            continue
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
        image_cnt += 1
    return image_cnt


def get_ann_path(ann_dir, dataset_name, is_original):
    instances_ann, captions_ann = None, None
    if is_original:
        instances_ann = os.path.join(ann_dir, f"instances_{dataset_name}.json")
        if not (os.path.exists(instances_ann) and os.path.isfile(instances_ann)):
            instances_ann = None
        if g.INCLUDE_CAPTIONS:
            captions_ann = os.path.join(ann_dir, f"captions_{dataset_name}.json")
            if not (os.path.exists(captions_ann) and os.path.isfile(captions_ann)):
                captions_ann = None
    else:
        ann_files = glob.glob(os.path.join(ann_dir, "*.json"))
        if len(ann_files) == 1:
            instances_ann, captions_ann = ann_files[0], None
            if g.INCLUDE_CAPTIONS:
                sly.logger.warn(
                    "Import captions is enabled, but only one .json annotation file found. "
                    "It will be used for instances. "
                    "If you want to import captions, please, add captions annotation file."
                )

        elif len(ann_files) > 1:
            if g.INCLUDE_CAPTIONS:
                instances_anns = [ann_file for ann_file in ann_files if "instance" in ann_file]
                captions_anns = [ann_file for ann_file in ann_files if "caption" in ann_file]
                if len(instances_anns) == 1:
                    instances_ann = instances_anns[0]
                if len(captions_anns) == 1:
                    captions_ann = captions_anns[0]
                if (
                    instances_ann == captions_anns
                    or len(captions_anns) == 0
                    or len(instances_anns) == 0
                ):
                    instances_ann = ann_files[0]
                    captions_ann = None
                    sly.logger.warn(
                        "Found more than one .json annotation file. "
                        "Import captions option is enabled, but more than one .json annotation file found. "
                        "It will be used for instances. "
                        "If you want to import captions, please, specify captions annotation file name."
                    )
            else:
                instances_anns = [ann_file for ann_file in ann_files if "instance" in ann_file]
                sly.logger.warn(
                    "Import captions is disabled, but more than one .json annotation file found."
                )
                if len(instances_anns) == 1:
                    instances_ann = instances_anns[0]
                    sly.logger.info(f"Instances annotation file found: {instances_ann}")
                else:
                    sly.logger.warn(
                        "Cannot find instances annotation file. "
                        "Please, specify instances and captions annotation file names (read app README).)"
                    )
    sly.logger.info(f"instances_ann: {instances_ann}")
    if g.INCLUDE_CAPTIONS:
        sly.logger.info(f"captions_ann: {captions_ann}")
    return instances_ann, captions_ann


def get_image_size_from_coco_annotation(image_info, img_id):
    for key in ["height", "width"]:
        if key not in image_info:
            raise KeyError(
                "Incorrect COCO annotation file: "
                f"image info (ID:{img_id}) does not contain '{key}' key"
            )
    return image_info["height"], image_info["width"]

