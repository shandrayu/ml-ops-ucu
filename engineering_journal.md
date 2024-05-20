# Engineering journal for MlOps tasks

## TODO

- Choose object detection model
  - Small. For example ,Nvidia Drive PX 2 has 16 GB LPDDR4.
- Can it be pretrained?
- Training code (preferebly from pretrained)
- Dataset filtering code (create pandas dataset from zenseact?)
  - convert to ... format (what dataset conversion is available? does the script preserve all of the interesting categories?)
  - Investigate what categories are present in the dataset
  - Filter data by
    - country (priority)
    - weather conditions
    - city/rural
    - ...
- create visualization in fiftyone
  - include lidar frames, 3D
- write about adding data by country in requirements
  - data management: retrain the model with adding new data from new country (combine all the data). If the country has a lot of data, add in chunks not more than X
  - (optional) continual learning  - continue training with new data, task TBD

## Done

- `zod generate coco --dataset-root=/Users/shandra/pro/ml-ops-ucu/data --output-dir=/Users/shandra/pro/ml-ops-ucu/data/coco --version=mini --anonymization=dnat`
- `pip uninstall zod -y && pip install zod`
- `zod download -y --url="https://www.dropbox.com/sh/04dfm3npbwg5vpj/AAAVKmFIO0VClMFVy7qiRdQQa\?dl\=0" --output-dir="/Users/shandra/pro/ml-ops-ucu/data" --subset=frames --version=mini --dnat --no-oxts --no-vehicle-data --no-blur --rm`
- Install zod[all] `pip install "zod[all]"`
