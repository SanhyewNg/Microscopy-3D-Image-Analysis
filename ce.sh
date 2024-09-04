#!/usr/bin/env bash
set -e
set -x
CONTAINER_ID="$1"
RUN_TEST=$2

# CE for segmentation
docker exec $CONTAINER_ID python -u -m clb.ce_segmentation --data data/instance/training/ --datasets "T8/train+T8/val+T8/test+T3/train+T3/val+T5/train+T5/val+T6/train+T6/val+NC1/train+NC1/val+NC2/train+NC2/val" --output data/evaluator/

# Copy results of evaluation into current directory.
docker cp $CONTAINER_ID:/code/data/evaluator/results .

# CE for classification
docker exec $CONTAINER_ID python -u -m clb.ce_classification --markers epith_Ki67 --data data/classification/sample/epith_ki67 --datasets "T3+T5+T6+T8" --output data/evaluator_classes/ --training_data_root data/classification/training --cross_validate GroupKFold
docker exec $CONTAINER_ID python -u -m clb.ce_classification --markers pdl1_cd8 --data data/classification/evaluation/pdl1_cd8 --datasets "NC1/test+NC2/test" --output data/evaluator_classes/ --training_data_root data/classification/training --cross_validate GroupKFold

# Copy results of evaluation into current directory.
docker cp $CONTAINER_ID:/code/data/evaluator_classes/results ./results_classes

# Print f_scores from ./results/Summary.txt file to capture them in <<Set build description>> action below.
echo "Segmentation:<br>" > build_desc.txt
echo $(awk '{if(NR % 6 == 1 || NR % 6 == 5) { printf("&emsp;%s<br>", $0) } }' ./results/Summary.txt) >> build_desc.txt

echo "<br>Classification using model on test data:<br>" >> build_desc.txt
echo "&emsp;Ki67:<br>" >> build_desc.txt
echo $(awk '{if(NR % 6 == 1 || NR % 6 == 5) { printf("&emsp;&emsp;%s<br>", $0) } }' ./results_classes/classify_evaluate_results/Ki67/Summary.txt) >> build_desc.txt
echo "&emsp;Epith:<br>" >> build_desc.txt
echo $(awk '{if(NR % 6 == 1 || NR % 6 == 5) { printf("&emsp;&emsp;%s<br>", $0) } }' ./results_classes/classify_evaluate_results/epith/Summary.txt) >> build_desc.txt
echo "&emsp;CD8:<br>" >> build_desc.txt
echo $(awk '{if(NR % 6 == 1 || NR % 6 == 5) { printf("&emsp;&emsp;%s<br>", $0) } }' ./results_classes/classify_evaluate_results/cd8/Summary.txt) >> build_desc.txt
echo "&emsp;PDL1:<br>" >> build_desc.txt
echo $(awk '{if(NR % 6 == 1 || NR % 6 == 5) { printf("&emsp;&emsp;%s<br>", $0) } }' ./results_classes/classify_evaluate_results/pdl1/Summary.txt) >> build_desc.txt

echo "<br>Classification cross-prediction:<br>" >> build_desc.txt
echo "&emsp;Ki67:<br>" >> build_desc.txt
echo $(awk '{if(NR % 6 == 1 || NR % 6 == 5) { printf("&emsp;&emsp;%s<br>", $0) } }' ./results_classes/cross_predict_results/Ki67/Summary.txt) >> build_desc.txt
echo "&emsp;Epith:<br>" >> build_desc.txt
echo $(awk '{if(NR % 6 == 1 || NR % 6 == 5) { printf("&emsp;&emsp;%s<br>", $0) } }' ./results_classes/cross_predict_results/epith/Summary.txt) >> build_desc.txt
echo "&emsp;CD8:<br>" >> build_desc.txt
echo $(awk '{if(NR % 6 == 1 || NR % 6 == 5) { printf("&emsp;&emsp;%s<br>", $0) } }' ./results_classes/cross_predict_results/cd8/Summary.txt) >> build_desc.txt
echo "&emsp;PDL1:<br>" >> build_desc.txt
echo $(awk '{if(NR % 6 == 1 || NR % 6 == 5) { printf("&emsp;&emsp;%s<br>", $0) } }' ./results_classes/cross_predict_results/pdl1/Summary.txt) >> build_desc.txt

echo "<br>Classify from scratch:<br>" >> build_desc.txt
echo "&emsp;Ki67:<br>" >> build_desc.txt
echo $(awk '{printf("&emsp;&emsp;%s<br>", $0)}' ./results_classes/train_test_results/Ki67/model_7_eager_torvalds_3_class_Ki67/summary_Ki67.txt) >> build_desc.txt
echo "&emsp;Epith:<br>" >> build_desc.txt
echo $(awk '{printf("&emsp;&emsp;%s<br>", $0)}' ./results_classes/train_test_results/epith/model_7_eager_torvalds_3_class_epith/summary_epith.txt) >> build_desc.txt
echo "&emsp;CD8:<br>" >> build_desc.txt
echo $(awk '{printf("&emsp;&emsp;%s<br>", $0)}' ./results_classes/train_test_results/cd8/model_8_angry_ptolemy_1_all_perc_class_cd8/summary_cd8.txt) >> build_desc.txt
echo "&emsp;PDL1:<br>" >> build_desc.txt
echo $(awk '{printf("&emsp;&emsp;%s<br>", $0)}' ./results_classes/train_test_results/pdl1/model_8_angry_ptolemy_1_preproc_class_pdl1/summary_pdl1.txt) >> build_desc.txt

echo [f_scores]$(cat build_desc.txt)

if [ "$RUN_TEST" = "1" ]; then

    # Run again this time for test only
    docker exec $CONTAINER_ID python -m clb.ce_segmentation --data data/instance/training/ --datasets "T8/test" --output data/evaluator_test/

    # Copy results of evaluation for test dataset into current directory.
    docker cp $CONTAINER_ID:/code/data/evaluator_test/results ./results_testset
fi
