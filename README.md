# Trabajo_CV_2
## Modelo de intelignecia 


## How to run the code:
In order to run the code you need to have installed:
-python
-poetry
And download the code in the repository.

## How to train a model new model
### to make a new checkpoint file 

to train a new model run the following command in the folder of the proyect
"./easy_run.sh train"

or

"poetry install --no-root"

"poetry run python Train_Model.py"

## How to deploy the REST API with the network
first you need to have a checkpoint file

once you have a chekpoint file run the following command in the folder of the proyect
"./easy_run.sh deploy"

or

"poetry install --no-root"

"poetry run python Flask_Deployment.py"


### Options available
options such as the number of workers, GPU acceleration, ip of the API, 
port os the API are available using the "poetry run python Train_Model.py" or 
"poetry run python Flask_Deployment.py"
. Use --help at the end of the command to learn more.