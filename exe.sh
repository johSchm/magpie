#!/bin/bash

# -------------------------------------------
# author:     Johann Schmidt
# date:       October 2019
# -------------------------------------------

if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi

TEXT_RESET='\e[0m'
TEXT_ERROR='\e[31m'
TEXT_HEADER='\e[1m'
TEMP=`getopt -o m:tdh --long mode:,tensorboard,dvc,help \
             -n 'javawrap' -- "$@"`
MODE=False
HELP=False
DVC=False
TENSORBOARD=False

MODE_NAME_EVAL="eval"
MODE_NAME_CONV="conv"
MODE_NAME_TRAIN="train"
MODE_NAME_PRED="pred"

# Starts tensorboard.
function tensorboard()
{
  echo "Starting Tensor Board in Browser ..."
  venv/bin/python3.6 venv/lib/python3.6/site-packages/tensorboard/main.py --logdir=res/models/I3D2S_002/log
  sleep 2s
  chromium http://localhost:6006/ -incognito
}

# Progesses modes.
function modeHandler()
{
  if [ $MODE = $MODE_NAME_CONV ] ; then
    convertData
  elif [ $MODE = $MODE_NAME_TRAIN ] ; then
    trainModel
  elif [ $MODE = $MODE_NAME_EVAL ] ; then
    evaluateModel
  elif [ $MODE = $MODE_NAME_PRED ] ; then
    predictLabels
  else
    echo -e "${TEXT_ERROR}ERROR: Invalid Mode!${TEXT_RESET}"
  fi
}

# Converts the data to appropriate formate.
function convertData()
{
  if [ $DVC == True ] ; then
    dvc run -o ../res/data \
            -f ../convert.dvc \
            python prepare.py
  else
    python prepare.py
  fi
}

# Trains the learn.
function trainModel()
{
  #../venv/bin/python train.py --sidx=0
  #../venv/bin/python train.py --sidx=1
  #../venv/bin/python train.py --sidx=2
  ../venv/bin/python train.py --sidx=3
  ../venv/bin/python train.py --sidx=4
  ../venv/bin/python train.py --sidx=5
  #mpirun -np 2 \
  #       -bind-to none -map-by slot \
  #       -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
  #       -mca pml ob1 -mca btl ^openib \
   #      python train.py

  #if [ $DVC == True ] ; then
  #  dvc run -d ../res/data \
  #          -d ../src/train.py \
  #          -o ../res/model/model.model \
  #          -f ../train.dvc \
  #          python train.py
  #else
  #  python train.py
  #fi
}

# Evaluates the learn.
function evaluateModel()
{
  ../venv/bin/python evaluate.py
}

# Predicts data labels.
function predictLabels()
{
  python predict.py
}

# display help text
function displayHelp()
{
  echo -e "\n\t${TEXT_HEADER}Human Action Recognition${TEXT_RESET}\n"
  echo -e "\t-m, --mode\t\tEnter desired mode
                       \t\t[conv]:  Converts data to appropriate formate.
                       \t\t[train]: Trains the model.
                       \t\t[eval]:  Evaluates the model.
                       \t\t[pred]:  Predicts labels."
  echo -e "\t-t, --tensorboard\tCalls tensorboard in the end."
  echo -e "\t-d, --dvc\t\tExecutes the mode using DVC."
  echo -e "\t-h, --help\t\tDisplay help information."
  echo -e "\n"
}

# Sets the python reference.
function setPythonRef()
{
  pwd
  source venv/bin/activate
  command -v python
}

while true; do
  case "$1" in
    -m | --mode ) MODE="$2"; shift 2 ;;
    -t | --tensorboard ) TENSORBOARD=True; shift ;;
    -d | --dvc ) DVC=True; shift ;;
    -h | --help ) HELP=True; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if [ $HELP == True ] ; then
  displayHelp
else
  setPythonRef
  if [ -n "$MODE" ] ; then
    modeHandler
  else
    echo -e "[${TEXT_ERROR}ERROR${TEXT_RESET}]: Nothing to do!"
  fi
fi
if [ $TENSORBOARD == True ] ; then
  tensorboard
fi
