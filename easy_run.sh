#!/bin/bash
poetry install --no-root
case $1 in
train)
echo Starting Training
poetry run python Train_Model.py
;;
#test)
#echo Starting Test
#;;
deploy)
echo Starting API
poetry run python Flask_Deployment.py
;;
*)
echo Invalid Option
echo "./easy_run.sh train => train neural network model"
#echo "./easy_run.sh test => test the train model"
echo "./easy_run.sh deploy => start the API with the traind model"
;;
esac
