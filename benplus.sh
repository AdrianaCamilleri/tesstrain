
mkdir -p data
cd data
wget -O radical-stroke.txt https://github.com/tesseract-ocr/langdata_lstm/raw/master/radical-stroke.txt
wget -O Latin.unicharset https://github.com/tesseract-ocr/langdata_lstm/raw/master/Latin.unicharset
wget -O Bengali.unicharset https://github.com/tesseract-ocr/langdata_lstm/raw/master/Bengali.unicharset

cd ben-ground-truth
ls
rm *.lstmf
rm *.box

python ../../normalize.py *.gt.txt

cd ../..

nohup make MODEL_NAME=ben START_MODEL=ben LANG_TYPE=Indic  GROUND_TRUTH_DIR=data/ben-ground-truth TESSDATA=$HOME/tessdata_best DEBUG_INTERVAL=-1 lists > data/ben-lists.log &

nohup make MODEL_NAME=ben START_MODEL=ben LANG_TYPE=Indic  GROUND_TRUTH_DIR=data/ben-ground-truth TESSDATA=$HOME/tessdata_best DEBUG_INTERVAL=-1 unicharset > data/ben.log &

nohup make MODEL_NAME=ben START_MODEL=ben LANG_TYPE=Indic  GROUND_TRUTH_DIR=data/ben-ground-truth TESSDATA=$HOME/tessdata_best DEBUG_INTERVAL=-1 training MAX_ITERATIONS=10000 >> data/ben.log &
