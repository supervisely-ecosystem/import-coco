import os
import globals as g
import supervisely_lib as sly
from supervisely_lib.project.project import read_single_project


def start(api, project_dir, workspace_id, project_name):
    if "meta.json" not in os.listdir(project_dir):
        meta = sly.ProjectMeta()
        path_to_meta = os.path.join(project_dir, "meta.json")
        meta_json = meta.to_json()
        sly.json.dump_json_file(meta_json, path_to_meta)

    project = g.api.project.create(workspace_id,
                                   project_name,
                                   type=sly.ProjectType.IMAGES,
                                   change_name_if_conflict=True)

    project_fs = read_single_project(project_dir)
    g.api.project.update_meta(project.id, project_fs.meta.to_json())
    sly.logger.info("Project {!r} [id={!r}] has been created".format(project.name, project.id))
    for dataset_fs in project_fs:
        dataset = api.dataset.create(project.id, dataset_fs.name)
        names, img_paths, ann_paths = [], [], []
        ds_progress = sly.Progress(f"Upload dataset: {dataset.name}", len(dataset_fs), min_report_percent=1)
        for item_name in dataset_fs:
            img_path, ann_path = dataset_fs.get_item_paths(item_name)
            names.append(item_name)
            img_paths.append(img_path)
            ann_paths.append(ann_path)

        for name, img_path, ann_path in zip(names, img_paths, ann_paths):
            img_info = api.image.upload_path(dataset.id, name, img_path)
            image_id = img_info.id
            api.annotation.upload_path(image_id, ann_path)
            ds_progress.iter_done_report()

        # img_infos = api.image.upload_paths(dataset.id, names, img_paths)
        # image_ids = [img_info.id for img_info in img_infos]
        # api.annotation.upload_paths(image_ids, ann_paths)
        # ds_progress.iter_done_report()
