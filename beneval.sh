MODEL_NAME=ben
REPORTS=data/$MODEL_NAME/reports
START_MODEL=ben
SCRIPT_MODEL=Bengali
VALIDATE_LIST=test
tesseract -v

rm -rf $REPORTS
mkdir -p $REPORTS

sed -e 's/lstmf/tif/' data/$MODEL_NAME/list.$VALIDATE_LIST > data/$MODEL_NAME/list.tif.$VALIDATE_LIST
sed -e 's/tif/gt.txt/' data/$MODEL_NAME/list.tif.$VALIDATE_LIST > data/$MODEL_NAME/list.txt.$VALIDATE_LIST
{ xargs -I{} sh -c "cat {}; echo ''" < data/${MODEL_NAME}/list.txt.$VALIDATE_LIST ; } > $REPORTS/gt.txt

### Evaluate SCRIPT_MODEL

OMP_THREAD_LIMIT=1 tesseract data/$MODEL_NAME/list.tif.$VALIDATE_LIST $REPORTS/$SCRIPT_MODEL.fast.OCR -l $SCRIPT_MODEL --dpi 300 --psm 13 --tessdata-dir ~/tessdata_fast/script -c page_separator=''
accuracy $REPORTS/gt.txt $REPORTS/$SCRIPT_MODEL.fast.OCR.txt  $REPORTS/$SCRIPT_MODEL.fast.acc.report.txt
wordacc $REPORTS/gt.txt $REPORTS/$SCRIPT_MODEL.fast.OCR.txt  $REPORTS/$SCRIPT_MODEL.fast.wordacc.report.txt

java -cp ~/ocreval.jar eu.digitisation.Main \
	-gt $REPORTS/gt.txt -e UTF-8  \
	-ocr $REPORTS/$SCRIPT_MODEL.fast.OCR.txt -e UTF-8  \
	-o $REPORTS/$SCRIPT_MODEL.fast.ocrevaluation.html

### Evaluate START_MODEL

OMP_THREAD_LIMIT=1 tesseract data/$MODEL_NAME/list.tif.$VALIDATE_LIST $REPORTS/$START_MODEL.fast.OCR -l $START_MODEL --dpi 300 --psm 13 --tessdata-dir ~/tessdata_fast -c page_separator=''
accuracy $REPORTS/gt.txt $REPORTS/$START_MODEL.fast.OCR.txt  $REPORTS/$START_MODEL.fast.acc.report.txt
wordacc $REPORTS/gt.txt $REPORTS/$START_MODEL.fast.OCR.txt  $REPORTS/$START_MODEL.fast.wordacc.report.txt

java -cp ~/ocreval.jar eu.digitisation.Main \
	-gt $REPORTS/gt.txt -e UTF-8  \
	-ocr $REPORTS/$START_MODEL.fast.OCR.txt -e UTF-8  \
	-o $REPORTS/$START_MODEL.fast.ocrevaluation.html

### Evaluate Model with minimum VALIDATION CER

for LANG in ben_2.494_3422_5200 ben_1.439_5105_9500 ; 
do
	OMP_THREAD_LIMIT=1 tesseract data/$MODEL_NAME/list.tif.$VALIDATE_LIST $REPORTS/$LANG.OCR -l $LANG --dpi 300 --psm 13 --tessdata-dir data/$MODEL_NAME/tessdata_fast -c page_separator=''
	accuracy $REPORTS/gt.txt $REPORTS/$LANG.OCR.txt  $REPORTS/$LANG.acc.report.txt
	wordacc $REPORTS/gt.txt $REPORTS/$LANG.OCR.txt  $REPORTS/$LANG.wordacc.report.txt
	
	OMP_THREAD_LIMIT=1 lstmeval  \
		--verbosity=1 \
		--model data/$MODEL_NAME/tessdata_fast/$LANG.traineddata \
		--eval_listfile data/$MODEL_NAME/list.$VALIDATE_LIST 2>&1 | grep -v ^Loaded | grep -v ^Warning > $REPORTS/$LANG.lstmeval.log
		grep ^OCR $REPORTS/$LANG.lstmeval.log | sed -e 's/^OCR  ://' > $REPORTS/$LANG.lstmeval.OCR.txt
		grep ^Truth $REPORTS/$LANG.lstmeval.log | sed -e 's/^Truth://' > $REPORTS/$LANG.lstmeval.Truth.txt
		wdiff -3 -s  $REPORTS/$LANG.lstmeval.Truth.txt $REPORTS/$LANG.lstmeval.OCR.txt > $REPORTS/$LANG.lstmeval.wdiff.report.txt
		diff -a --suppress-common-lines -y $REPORTS/$LANG.lstmeval.Truth.txt $REPORTS/$LANG.lstmeval.OCR.txt > $REPORTS/$LANG.lstmeval.diff.report.txt
		
	java -cp ~/ocreval.jar eu.digitisation.Main \
		-gt $REPORTS/gt.txt -e UTF-8  \
		-ocr $REPORTS/$LANG.OCR.txt -e UTF-8  \
		-o $REPORTS/$LANG.ocrevaluation.html
done
