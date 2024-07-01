import os
import sys

import supervisely as sly
from pycocotools.coco import COCO
from supervisely.io.fs import dir_exists

import coco_converter
import coco_downloader
import globals as g


class HiddenCocoPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


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
        if not dir_exists(coco_ann_dir):
            if dir_exists(os.path.join(coco_dataset_dir, "annotation")):
                coco_ann_dir = os.path.join(coco_dataset_dir, "annotation")
            else:
                app_logger.warn(f"Not found 'annotations' folder")
        if not dir_exists(g.src_img_dir):
            app_logger.warn("Not found 'images' folder.")
            imgs_list = sly.fs.list_files_recursively(
                coco_dataset_dir, sly.image.SUPPORTED_IMG_EXTS
            )
            if len(imgs_list) > 0:
                imgs_dirs = [os.path.dirname(img_path) for img_path in imgs_list]
                imgs_dirs = list(set(imgs_dirs))
                if len(imgs_dirs) == 1:
                    g.src_img_dir = imgs_dirs[0]
                    app_logger.warn(f"Found images in '{g.src_img_dir}' folder.")
                else:
                    continue
            else:
                continue

        images = sly.fs.list_files_recursively(g.src_img_dir, sly.image.SUPPORTED_IMG_EXTS)
        if len(images) == 0:
            app_logger.warn(
                f"Folder '{g.src_img_dir}' has no images at this level. Read the application overview."
            )
            continue

        coco_instances_ann_path, coco_captions_ann_path = coco_converter.get_ann_path(
            ann_dir=coco_ann_dir, dataset_name=dataset, is_original=g.is_original
        )
        if coco_instances_ann_path is not None:
            try:
                coco_instances_file_name = os.path.basename(coco_instances_ann_path)
                coco_converter.check_high_level_coco_ann_structure(coco_instances_ann_path)
                with HiddenCocoPrints():
                    coco_instances = COCO(annotation_file=coco_instances_ann_path)
            except Exception as e:
                raise Exception(
                    f"Incorrect instances annotation file {coco_instances_file_name}: {repr(e)}"
                ) from e

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
            incorrect_anns = 0
            skipped_images = 0

            for img_id, img_info in coco_images.items():
                image_name = img_info.get("file_name")
                if image_name is None:
                    incorrect_anns += 1
                    ds_progress.iter_done_report()
                    continue
                if "/" in image_name:
                    image_name = os.path.basename(image_name)
                if not sly.fs.file_exists(os.path.join(g.src_img_dir, image_name)):
                    skipped_images += 1
                    ds_progress.iter_done_report()
                    continue
                img_ann = coco_anns[img_id]
                img_size = coco_converter.get_image_size_from_coco_annotation(img_info, img_id)
                ann, meta = coco_converter.create_sly_ann_from_coco_annotation(
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
            if incorrect_anns > 0:
                app_logger.warn(f"{skipped_images} images skipped because of incorrect annotation.")
            if skipped_images > 0:
                app_logger.warn(f"{skipped_images} images skipped because of missing files.")
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
        msg = "Not found COCO format datasets in the input directory"
        description = "Please, read the application overview."
        sly.logger.error(msg)
        api.task.set_output_error(task_id, msg, description)
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
    sly.main_wrapper("main", main, log_for_agent=False)
