import glob
import os
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
    for category in coco_categories:
        if category["name"] in [obj_class.name for obj_class in g.META.obj_classes]:
            continue
        new_color = sly.color.generate_rgb(colors)
        colors.append(new_color)

        obj_classes = []
        tag_metas = []

        if ann_types is not None:
            if "segmentation" in ann_types:
                obj_classes.append(sly.ObjClass(category["name"], sly.Polygon, new_color))
            if "bbox" in ann_types:
                obj_classes.append(
                    sly.ObjClass(add_tail(category["name"], "bbox"), sly.Rectangle, new_color)
                )
            if "caption" in ann_types:
                tag_metas.append(sly.TagMeta("caption", sly.TagValueType.ANY_STRING))

        g.META = g.META.add_obj_classes(obj_classes)
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

    exteriors = []
    for polygon in polygons:
        polygon = [polygon[i * 2 : (i + 1) * 2] for i in range((len(polygon) + 2 - 1) // 2)]
        exteriors.append([(width, height) for width, height in polygon])

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
            results = [cv2.pointPolygonTest(contours[0], (x, y), False) > 0 for x, y in exterior2]
            # if results of True, then all points are inside or on contour
            if all(results):
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
    return sly.Bitmap(mask).to_contours()


def create_sly_ann_from_coco_annotation(meta, coco_categories, coco_ann, image_size, captions):
    labels = []
    name_cat_id_map = coco_category_to_class_name(coco_categories)
    for object in coco_ann:
        segm = object.get("segmentation")
        curr_labels = []
        if segm is not None and len(segm) > 0:
            obj_class_name_polygon = name_cat_id_map[object["category_id"]]
            obj_class_polygon = meta.get_obj_class(obj_class_name_polygon)

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
            obj_class_name_rectangle = name_cat_id_map[object["category_id"]]
            if not obj_class_name_rectangle.endswith("bbox"):
                obj_class_name_rectangle = add_tail(obj_class_name_rectangle, "bbox")
            obj_class_rectangle = meta.get_obj_class(obj_class_name_rectangle)
            if len(curr_labels) > 1:
                for label in curr_labels:
                    bbox = label.geometry.to_bbox()
                    labels.append(sly.Label(bbox, obj_class_rectangle))
            else:
                x, y, w, h = bbox
                rectangle = sly.Label(sly.Rectangle(y, x, y + h, x + w), obj_class_rectangle)
                labels.append(rectangle)
    imag_tags = []
    if captions is not None:
        for caption in captions:
            imag_tags.append(sly.Tag(meta.get_tag_meta("caption"), caption["caption"]))
    return sly.Annotation(image_size, labels=labels, img_tags=imag_tags)


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
    if "/" in image_name:
        image_name = os.path.basename(image_name)
    ann_json = ann.to_json()
    coco_img_path = os.path.join(g.COCO_BASE_DIR, dataset, "images", image_name)
    sly_img_path = os.path.join(g.img_dir, image_name)
    if file_exists(os.path.join(coco_img_path)):
        sly.json.dump_json_file(ann_json, os.path.join(g.ann_dir, f"{image_name}.json"))
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
            instances_ann_exists = any("instances" in ann_file for ann_file in ann_files)
            captions_ann_exists = any("captions" in ann_file for ann_file in ann_files)
            if instances_ann_exists and captions_ann_exists:
                # instances_ann = [ann_file for ann_file in ann_files if "instances" in ann_file][0]
                # captions_ann = [ann_file for ann_file in ann_files if "captions" in ann_file][0]
                # if instances_ann != captions_ann:
                sly.logger.warn(f"Found instances and captions annotation files.")
                return True
            else:
                sly.logger.warn(
                    f"Found more than one .json annotation file. Please, specify names which one is for instances and which one is for captions."
                )
        elif len(ann_files) == 0:
            sly.logger.info(
                f"Annotation file not found in {ann_dir}. Please, read apps overview and prepare the dataset correctly."
            )
        return False


def get_ann_path(ann_dir, dataset_name, is_original):
    instances_ann, captions_ann = None, None
    if is_original:
        instances_ann = os.path.join(ann_dir, f"instances_{dataset_name}.json")
        captions_ann = os.path.join(ann_dir, f"captions_{dataset_name}.json")
    else:
        ann_files = glob.glob(os.path.join(ann_dir, "*.json"))
        if len(ann_files) == 1:
            instances_ann, captions_ann = ann_files[0], None
        elif len(ann_files) > 1:
            instances_ann_exists = any("instances" in ann_file for ann_file in ann_files)
            captions_ann_exists = any("captions" in ann_file for ann_file in ann_files)
            if instances_ann_exists and captions_ann_exists:
                instances_ann = [ann_file for ann_file in ann_files if "instances" in ann_file][0]
                captions_ann = [ann_file for ann_file in ann_files if "captions" in ann_file][0]
                if instances_ann == captions_ann:
                    instances_ann = captions_ann = None
                    sly.logger.warn(
                        "Found same names for instances and captions annotation files. "
                        f"Please, specify names which one is for instances and which one is for captions."
                    )
                sly.logger.info(
                    f"Found instances and captions annotation files: {instances_ann} {captions_ann}"
                )
            else:
                sly.logger.warn(
                    f"Found more than one .json annotation file. "
                    "Please, specify names which one is for instances and which one is for captions."
                )
    return instances_ann, captions_ann
