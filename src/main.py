import os
import globals as g
import coco_downloader
import supervisely_lib as sly
import coco_converter

import upload_images_project

def upload_project():
    pass


@g.my_app.callback("import_coco")
@sly.timeit
def import_coco(api: sly.Api, task_id, context, state, app_logger):
    #coco_datasets = coco_downloader.download_custom_coco_dataset(g.custom_ds, app_logger)
    coco_datasets = coco_downloader.download_original_coco_dataset(g.original_ds, app_logger)

    for dataset in coco_datasets:
        coco_dataset = coco_converter.read_coco_dataset(dataset)
        sly_dataset_dir = coco_converter.create_sly_dataset_dir(dataset)
        if not os.path.exists(os.path.join(g.sly_base_dir, "meta.json")):
            meta = coco_converter.create_sly_meta_from_coco_categories(coco_dataset.categories)  # @TODO: MERGE META WITH OTHER DATASETS
        ds_progress = sly.Progress(f"Converting dataset: {dataset}", len(coco_dataset.images))
        for coco_image in coco_dataset.images:
            h, w = (coco_image["height"], coco_image["width"])
            img_size = (h, w)
            coco_annotations = coco_converter.get_coco_annotations_for_current_image(coco_image, coco_dataset.annotations)
            ann = coco_converter.create_sly_ann_from_coco_annotation(meta, coco_dataset.categories, coco_annotations, img_size)
            coco_converter.move_to_sly_dataset(dataset, sly_dataset_dir, coco_image, ann)
            ds_progress.iter_done_report()

    upload_images_project.start(api, g.sly_base_dir, g.workspace_id, "coco_val_2017")

    g.my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "TEAM_ID": g.team_id,
        "WORKSPACE_ID": g.workspace_id
    })
    g.my_app.run(initial_events=[{"command": "import_coco"}])


if __name__ == '__main__':
    sly.main_wrapper("main", main)
