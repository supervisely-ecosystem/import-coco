import globals as g
import coco_downloader
import coco_converter
import upload_images_project
import supervisely_lib as sly
from supervisely_lib.io.fs import dir_exists
import os


@g.my_app.callback("import_coco")
@sly.timeit
def import_coco(api: sly.Api, task_id, context, state, app_logger):
    if g.is_original:
        coco_datasets = coco_downloader.download_original_coco_dataset(g.original_ds, app_logger)
        project_name = "Original COCO"
    else:
        coco_datasets = coco_downloader.download_custom_coco_dataset(g.custom_ds, app_logger)
        project_name = "Custom COCO"

    for dataset in coco_datasets:
        if not dir_exists(os.path.join(g.coco_base_dir, dataset, "images")):
            app_logger.warn(
                "Incorrect input data. Folder with images must have name 'images'. See 'READMY' for more information.")
            g.my_app.stop()

        has_ann = coco_converter.check_dataset_for_annotation(dataset)
        if has_ann:
            coco_dataset = coco_converter.read_coco_dataset(dataset)
            sly_dataset_dir = coco_converter.create_sly_dataset_dir(dataset)
            meta = coco_converter.get_sly_meta_from_coco(coco_dataset, dataset)
            ds_progress = sly.Progress(f"Converting dataset: {dataset}", len(coco_dataset.images), min_report_percent=1)
            for batch in sly.batched(coco_dataset.images, batch_size=10):
                for coco_image in batch:
                    h, w = (coco_image["height"], coco_image["width"])
                    img_size = (h, w)
                    coco_annotations = coco_converter.get_coco_annotations_for_current_image(coco_image, coco_dataset.annotations)
                    ann = coco_converter.create_sly_ann_from_coco_annotation(meta, coco_dataset.categories, coco_annotations, img_size)
                    coco_converter.move_trainvalds_to_sly_dataset(dataset, sly_dataset_dir, coco_image, ann)
                    ds_progress.iter_done_report()
        else:
            sly_dataset_dir = coco_converter.create_sly_dataset_dir(dataset)
            coco_converter.move_testds_to_sly_dataset(dataset, g.coco_base_dir, sly_dataset_dir)

    upload_images_project.start(api, g.sly_base_dir, g.workspace_id, project_name)
    g.my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "TEAM_ID": g.team_id,
        "WORKSPACE_ID": g.workspace_id
    })
    g.my_app.run(initial_events=[{"command": "import_coco"}])


if __name__ == '__main__':
    sly.main_wrapper("main", main)
