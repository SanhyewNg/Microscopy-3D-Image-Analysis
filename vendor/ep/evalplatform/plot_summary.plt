#===== INPUT FROM EP =====#
#output_file = <filename_without_extension>
#output_file_extension = ".svg"|".png"
#set terminal (svg | pngcairo) size 1200,800 linewidth 2 font ",22" # (depends on evaluation.ini)
#plot_title = <plot_title>

#===== GRID SETTINGS =====#
grid_xtics=3;
grid_mxtics=1;

grid_xlabel=""
grid_ylabel=""

#===== FINAL SETTINGS =====#
# Output
set terminal pngcairo font ",14"
#set terminal (svg | pngcairo | other) [options] # if you want to override evaluation.ini settings
set output output_file.output_file_extension

# Plot style
set style data lines
set key center bottom box inside horizontal noenhanced
unset grid
set grid mxtics mytics xtics ytics

# Ranges and labels
set yrange [0.5:1]
set mytics 5
set ylabel grid_ylabel

set xtics grid_xtics font ",8" nomirror rotate by 20 right
set mxtics grid_mxtics
set xlabel grid_xlabel

set title plot_title font ",32" noenhanced

#===== PLOT SETTINGS =====#

# generated dynamically
#plot data_file index 0 using 2:xtic(1) title " Precision" with lines lw 2
#	 ,data_file index 1 using 2:xtic(1) title "Recall" with lines lw 2 \
#	 ,data_file index 2 using 2:xtic(1) title "F" with lines lw 2 \
