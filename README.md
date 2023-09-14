<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/48913536/183913663-f48c6c5e-65af-4f7d-b442-8699f2b48309.png"/>


# Import COCO

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#Results">Results</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/import-coco)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/import-coco)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/import-coco&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/import-coco&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/import-coco&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

App converts [COCO format](https://cocodataset.org/#home) datasets to [Supervisely format](https://docs.supervise.ly/data-organization/00_ann_format_navi)

Application key points:  
- Import full original COCO 2017 & COCO 2014 datasets
- Supports custom coco datasets
- Supports only instance segmentation(polygons), object detection(bounding boxes) and captions (tags) from COCO format
- All information about dataset, licenses and images from COCO annotation file **will be lost**
- Backward compatible with [Export to COCO](https://ecosystem.supervisely.com/apps/export-to-coco?_ga=2.203216728.833506216.1692536477-1574751671.1670221597)
- Support holes in polygons

Custom project structure:

Here is an example of a valid project structure to import custom COCO dataset - [Lemons.zip](https://github.com/supervisely-ecosystem/import-coco/files/12407330/Lemons.zip).

To import COCO Keypoints use [Import COCO Keypoints](https://ecosystem.supervisely.com/apps/import-coco-keypoints) app.

```
.
COCO_BASE_DIRECTORY
├── coco_dataset_1        # With annotations
│   ├── annotations
│   │   ├── instances.json
│   │   └── captions.json # optional (if you want to import captions)
│   └── images
│       ├── IMG_3861.jpeg
│       ├── IMG_4451.jpeg
│       └── IMG_8144.jpeg
├── coco_dataset_2        # Dataset with empty annotations dir will be treated like dataset without annotations
│   ├── annotations
│   └── images
│       ├── IMG_0748.jpeg
│       ├── IMG_1836.jpeg
│       └── IMG_2084.jpeg
└── coco_dataset_3        # Without annotations
    └── images
        ├── IMG_0428.jpeg
        ├── IMG_1885.jpeg
        └── IMG_2994.jpeg
```

# How to Use
**Step 1.** Add [Import COCO](https://ecosystem.supervise.ly/apps/import-coco) to your team from Ecosystem.

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/import-coco" src="https://i.imgur.com/d6ilGDr.png" width="350px" style='padding-bottom: 20px'/>

**Step 2.** Run app from the `Plugins & Apps` chapter:

<div align="center" markdown>
  <img src="https://i.imgur.com/2luJyn4.png"/>
</div>

**Step 3.** Select import mode:

- Your can download selected datasets from [COCO](https://cocodataset.org/#download).  
- Use your custom dataset in COCO format by path to your archive in `Team Files`.

<div align="center" markdown>
  <img src="https://user-images.githubusercontent.com/48913536/183898478-05fc7314-3d28-408e-bbe5-90e6522f0102.png" width="700px"/>
</div>

**Step 4.** After pressing the `Run` button you will be redirected to the `Tasks` page.

# Results

Result project will be saved in your current `Workspace` with name `Original COCO` for original datasets and `Custom COCO` for custom datasets

<div align="center" markdown>
<img src="https://i.imgur.com/BJuGxtL.png"/>
</div>
