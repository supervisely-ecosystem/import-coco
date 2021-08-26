import globals as g
import supervisely_lib as sly

from supervisely_lib.project.project import read_single_project


def start(api, project_dir, workspace_id, project_name):
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
        for item_name in dataset_fs:
            img_path, ann_path = dataset_fs.get_item_paths(item_name)
            names.append(item_name)
            img_paths.append(img_path)
            ann_paths.append(ann_path)

        img_infos = api.image.upload_paths(dataset.id, names, img_paths)
        image_ids = [img_info.id for img_info in img_infos]
        api.annotation.upload_paths(image_ids, ann_paths)
