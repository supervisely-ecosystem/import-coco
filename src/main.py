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
    for dataset in coco_datasets:
        sly.logger.info(f"Start processing {dataset} dataset...")
        coco_dataset_dir = os.path.join(g.COCO_BASE_DIR, dataset)
        if not dir_exists(coco_dataset_dir):
            app_logger.info(
                f"File {coco_dataset_dir} has been skipped."
            )
            continue
        coco_ann_dir = os.path.join(coco_dataset_dir, "annotations")
        if not dir_exists(os.path.join(coco_dataset_dir, "images")):
            app_logger.warn(
                "Incorrect input data. Folder with images must be named 'images'. See 'README' for more information."
            )
            continue

        if coco_converter.check_dataset_for_annotation(
            dataset_name=dataset, ann_dir=coco_ann_dir, is_original=g.is_original
        ):
            coco_ann_path = coco_converter.get_ann_path(
                ann_dir=coco_ann_dir, dataset_name=dataset, is_original=g.is_original
            )

            try:
                coco = COCO(annotation_file=coco_ann_path)
            except Exception as e:
                raise Exception(f"Incorrect annotation file: {coco_ann_path}: {e}")
            categories = coco.loadCats(ids=coco.getCatIds())
            coco_images = coco.imgs
            coco_anns = coco.imgToAnns

            sly_dataset_dir = coco_converter.create_sly_dataset_dir(
                dataset_name=dataset
            )
            g.img_dir = os.path.join(sly_dataset_dir, "img")
            g.ann_dir = os.path.join(sly_dataset_dir, "ann")

            types = coco_converter.get_ann_types(coco)

            meta = coco_converter.get_sly_meta_from_coco(
                coco_categories=categories, dataset_name=dataset, ann_types=types
            )

            ds_progress = sly.Progress(
                message=f"Converting dataset: {dataset}",
                total_cnt=len(coco_images),
                min_report_percent=1,
            )

            for img_id, img_info in coco_images.items():
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
                ds_progress.iter_done_report()
        else:
            coco_converter.get_sly_meta_from_coco(coco_categories=[], dataset_name=dataset)
            sly_dataset_dir = coco_converter.create_sly_dataset_dir(dataset_name=dataset)
            g.src_img_dir = os.path.join(g.COCO_BASE_DIR, dataset, "images")
            g.dst_img_dir = os.path.join(sly_dataset_dir, "img")
            g.ann_dir = os.path.join(sly_dataset_dir, "ann")
            coco_converter.move_testds_to_sly_dataset(dataset=dataset)
        sly.logger.info(f"Dataset {dataset} has been successfully converted.")

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
