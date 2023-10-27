import os

import supervisely as sly
from pycocotools.coco import COCO
from supervisely.io.fs import dir_exists

import coco_converter
import coco_downloader
import globals as g


@g.my_app.callback("import_coco")
@sly.timeit
def import_coco(api: sly.Api, task_id, context, state, app_logger):
    project_name, coco_datasets = coco_downloader.start(app_logger)
    total_images = 0
    for dataset in coco_datasets:
        current_dataset_images_cnt = 0
        sly.logger.info(f"Start processing {dataset} dataset...")
        coco_dataset_dir = os.path.join(g.COCO_BASE_DIR, dataset)
        g.src_img_dir = os.path.join(coco_dataset_dir, "images")
        if not dir_exists(coco_dataset_dir):
            app_logger.info(f"File {coco_dataset_dir} has been skipped.")
            continue
        coco_ann_dir = os.path.join(coco_dataset_dir, "annotations")
        if not dir_exists(g.src_img_dir):
            app_logger.warn(
                "Incorrect input data. Folder with images must be named 'images'. See 'README' for more information."
            )
            continue

        images = sly.fs.list_files_recursively(g.src_img_dir, sly.image.SUPPORTED_IMG_EXTS)
        if len(images) == 0:
            app_logger.warn(
                f"Folder {g.src_img_dir} has no images on this level. Read application overview."
            )
            continue

        coco_instances_ann_path, coco_captions_ann_path = coco_converter.get_ann_path(
            ann_dir=coco_ann_dir, dataset_name=dataset, is_original=g.is_original
        )
        if coco_instances_ann_path is not None:
            try:
                coco_instances = COCO(annotation_file=coco_instances_ann_path)
            except Exception as e:
                raise Exception(
                    f"Incorrect instances annotation file: {coco_instances_ann_path}: {e}"
                )

            categories = coco_instances.loadCats(ids=coco_instances.getCatIds())
            coco_images = coco_instances.imgs
            coco_anns = coco_instances.imgToAnns

            types = coco_converter.get_ann_types(coco=coco_instances)

            if coco_captions_ann_path is not None and sly.fs.file_exists(coco_captions_ann_path):
                try:
                    coco_captions = COCO(annotation_file=coco_captions_ann_path)
                    types += coco_converter.get_ann_types(coco=coco_captions)
                    for img_id, ann in coco_instances.imgToAnns.items():
                        ann.extend(coco_captions.imgToAnns[img_id])
                except:
                    coco_captions = None

            sly_dataset_dir = coco_converter.create_sly_dataset_dir(dataset_name=dataset)
            g.img_dir = os.path.join(sly_dataset_dir, "img")
            g.ann_dir = os.path.join(sly_dataset_dir, "ann")

            meta = coco_converter.get_sly_meta_from_coco(
                coco_categories=categories, dataset_name=dataset, ann_types=types
            )

            ds_progress = sly.Progress(
                message=f"Converting dataset: {dataset}",
                total_cnt=len(coco_images),
                min_report_percent=1,
            )

            for img_id, img_info in coco_images.items():
                image_name = img_info["file_name"]
                if "/" in image_name:
                    image_name = os.path.basename(image_name)
                if sly.fs.file_exists(os.path.join(g.src_img_dir, image_name)):
                    img_ann = coco_anns[img_id]
                    img_size = (img_info["height"], img_info["width"])
                    ann = coco_converter.create_sly_ann_from_coco_annotation(
                        meta=meta,
                        coco_categories=categories,
                        coco_ann=img_ann,
                        image_size=img_size,
                    )
                    coco_converter.move_trainvalds_to_sly_dataset(
                        dataset=dataset, coco_image=img_info, ann=ann
                    )
                    current_dataset_images_cnt += 1

                ds_progress.iter_done_report()
        else:
            coco_converter.get_sly_meta_from_coco(coco_categories=[], dataset_name=dataset)
            sly_dataset_dir = coco_converter.create_sly_dataset_dir(dataset_name=dataset)
            g.dst_img_dir = os.path.join(sly_dataset_dir, "img")
            g.ann_dir = os.path.join(sly_dataset_dir, "ann")
            current_dataset_images_cnt = coco_converter.move_testds_to_sly_dataset(
                dataset=dataset, image_cnt=current_dataset_images_cnt
            )
        if current_dataset_images_cnt == 0:
            coco_converter.remove_empty_sly_dataset_dir(dataset_name=dataset)
        else:
            sly.logger.info(f"Dataset {dataset} has been successfully converted.")
            total_images += current_dataset_images_cnt

    if len(coco_datasets) == 0 or total_images == 0:
        sly.logger.warn("Not found COCO format datasets in the input directory")
    else:
        sly.upload_project(
            dir=g.SLY_BASE_DIR,
            api=api,
            workspace_id=g.WORKSPACE_ID,
            project_name=project_name,
            log_progress=True,
        )
    g.my_app.stop()


def main():
    sly.logger.info(
        "Script arguments", extra={"TEAM_ID": g.TEAM_ID, "WORKSPACE_ID": g.WORKSPACE_ID}
    )
    g.my_app.run(initial_events=[{"command": "import_coco"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)
