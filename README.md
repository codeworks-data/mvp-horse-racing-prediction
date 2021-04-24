# Hong Kong Horse Racing Prediction

The aim of this project is to predict the outcome of horse racing using machine learning algorithms.

## Dataset
The dataset comes from [Kaggle](https://www.kaggle.com/gdaley/hkracing) and covers races in HK from **1997 to 2005**. <br>
The data consists of **6349** races with 4,405 runners. <br>
The 5878 races run before January 2005 are used to develop the forecasting models whereas the remaining 471 races run after January 2005 are preserved to conduct out-of-sample testing.

## GCP Training Part

## 1. Create a new GCP Project

* Get the billing accounts list

```bash
gcloud alpha billing accounts list
```

* Get the GCP Folder ID

```bash
GCP_FOLDER_ID=$( gcloud alpha resource-manager folders list --folder=244298749746 --format=json | jq -c '.[] | select( .displayName | contains("DATA"))' | jq '.name' | cut -f 2 -d '/' | sed 's/"//g')
```

* Name the project

```bash
PROJECT_ID=<my-project>
```

* Create new project

```bash
gcloud projects create ${PROJECT_ID} --folder=${GCP_FOLDER_ID}
```

* Get the project number

```bash
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
```

- Link the project to the billing account

```bash
gcloud alpha billing projects link ${PROJECT_NUMBER} --billing-account=${REPLACE_WITH_AN_ENABLED_ACCOUNT_ID}
```

## 2. Copy data to GCS
===

* Use the gsutil command to create a bucket

```bash
BUCKET_NAME=<my-project-bucket>

gsutil mb gs://${BUCKET_NAME}
```

* List all the buckets in the main project

```bash
gsutil ls
```

* Upload data to GCS : copy train and test data

```bash
gsutil cp data/*.h5 gs://${BUCKET_NAME}/data
```





