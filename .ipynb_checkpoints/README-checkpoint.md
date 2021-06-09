# Hong Kong Horse Racing Prediction

The aim of this project is to predict the outcome of horse racing using machine learning algorithms.

## Dataset
The dataset comes from [Kaggle](https://www.kaggle.com/gdaley/hkracing) and covers races in HK from **1997 to 2005**. <br>
The data consists of **6349** races with 4,405 runners. <br>
The 5878 races run before January 2005 are used to develop the forecasting models whereas the remaining 471 races run after January 2005 are preserved to conduct out-of-sample testing.

## GCP

The aim of this folder is to show all notebooks and algorithms we used to predict either winner and placed horses.

We have an article explainning our journey through this process. You can find a link below:
* Part 1: https://medium.com/codeworksparis/horse-racing-prediction-a-machine-learning-approach-part-1-44ed7fca869e
* Part 2: .......

### List of all folder and files

* requirement : it's a list of requirements needed to run this project
* create_dataset and config are both used to split our inital data into train and test sets respecting the date of runs
* data is a folder containing all data once they are split by the create_dataset script
* winner is a folder containing all notebook and ml models to bet on winner horses
* placed is a folder containing all notebook and ml models to bet on placed horses

### winner folder

Let's have a look about the winner files:

* winner_01_lgbm_optim : notebook to run the hyperoptimization for lgbm
* winner_02_train : notebook which run all trainning process either for lgbm and deep learning and saved results
* winner_03_show_result : notebook which help us to verify our information and go deeper about our prediction for a specific month
* winner_04_all_results : notebook which consolidated all month with and ensemble model and show final results
* winner_functions.py : this python file have all needed function to run those 4 previous notebooks
* model folder contains all saved models the winner_02_train.
* result_hyperopt.csv : we saved here our optimizations steps


### placed folder

Let's have a look about the placed files:


* placed_01_train : notebook which run all trainning process for deep learning and saved results
* placed_02_show_result : notebook which help us to verify our information and go deeper about our prediction for a specific month
* placed_03_consolidated : notebook which consolidated all month with and ensemble model and show final results
* placed_functions.py : this python file have all needed function to run those 4 previous notebooks
* model folder contains all saved models the placed_01_train and lgbm models from the winner_folder (we kept the same)


