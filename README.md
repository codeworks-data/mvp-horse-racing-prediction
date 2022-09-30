# Hong Kong Horse Racing Prediction

The aim of this project is to predict the outcome of horse racing using machine learning algorithms.

![horse_racing](https://user-images.githubusercontent.com/71772293/123072394-ca65c100-d415-11eb-9660-24d726f7541d.jpeg)

From RaceBets



## Dataset
The dataset comes from [Kaggle](https://www.kaggle.com/gdaley/hkracing) and covers races in HK from **1997 to 2005**. <br>
The data consists of **6,349** races with 4,405 runners. <br>
The 5,878 races ran before January 2005 are used to develop the forecasting models whereas the remaining 471 races ran after January 2005 are preserved to conduct out-of-sample testing.

We have an article explaining our journey through this process. You can find a link below:
* **[Horse Racing Prediction Part 1](https://medium.com/codeworksparis/horse-racing-prediction-a-machine-learning-approach-part-1-44ed7fca869e)**
* **[Horse Racing Prediction Part 2](https://medium.com/codeworksparis/horse-racing-prediction-a-machine-learning-approach-part-2-e9f5eb9a92e9)** 

### List of all folder and files

* **requirements.txt** is a list of requirements needed to run this project
* **baseline_models.ipynb** is a notebook containing informations for part 1 on baseline models
* **quick_eda_horse_racing.ipynb** is a notebook with a quick EDA on our dataset
* **create_dataset.py** and **config** are both used to split our inital data into train and test sets depending on the date of races
* **extract_features.py** is used to perform feature engineering
* **winner/** is a folder containing all notebooks and ML models to bet on winner horses
* **placed/** is a folder containing all notebooks and ML models to bet on placed horses

### winner folder

Let's have a look about the winner files:

* **winner_01_lgbm_optim** is a notebook which runs the hyperoptimization for LGBM
* **winner_02_train** is a notebook which runs all training processes either for LGBM and deep learning then saves results
* **winner_03_show_result** is a notebook which helps us to verify our informations and go deeper about our predictions for a specific month
* **winner_04_all_results** is a notebook which consolidates all months with an ensemble model and shows final results
* **winner_functions.py** is a python file which has all the required functions to run those 4 previous notebooks
* **model** is a folder that contains all saved models from winner_02_train
* **result_hyperopt.csv** is a csv file with all our optimizations steps


### placed folder

Let's have a look about the placed files:

* **placed_01_train** is a notebook which runs all training processes for deep learning then saves results
* **placed_02_show_result** is a notebook which helps us to verify our informations and go deeper about our predictions for a specific month
* **placed_03_consolidated** is a notebook which consolidates all months with an ensemble model and shows final results
* **placed_functions.py** is a python file which has all the required functions to run those 4 previous notebooks
* **model** is a folder that contains all saved models from placed_01_train and LGBM models from winner_folder
