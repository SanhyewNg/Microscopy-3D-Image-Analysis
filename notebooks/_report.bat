@ECHO OFF
if NOT "%~4" == "" ( 
    set ki67=-p classification_ki67_file %4
) 
if NOT "%~5" == "" ( 
    set epith=-p classification_epith_file %5
)
papermill -p input_file %2 -p segmentation_file %3 %ki67% %epith% report_results.ipynb Report.ipynb & jupyter nbconvert --to html --template ./_no_output.tpl Report.ipynb --output %1 & del Report.ipynb