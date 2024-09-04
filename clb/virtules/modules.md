# Virtules - Virtum modules:

These are wrappers for processings created in CLB-AI so that they can be easily called in Virtum without the need to know the CLB-AI internals.
For example Virtum calls for segmentation and it is up to virtules to pick the correct model and setting which can be inferred from the data (e.g. voxel size).

#### Abstract format:

MODULE = (Name, Script, Description, INPUT, OUTPUT)

INPUT = { param_name | * : Type.Subtype }

PARAMETERS = { param_name : Type }

OUTPUT = { param_name : Type.Subtype } 

## ENHANCEMENTS

### enhancement.py

* Name = Enhancement

* Description = "Denoise imagery with pretrained N2N model."

* Input = { input : Volume,   enhancements: List of Enhancements}

* Enhancements = { name: Str, channel : List of Int } 

* Output = { output_file : Volume }

Remarks:
* Only denoise enhancement is implemented.


## SEGMENTATION

### segment.py

* Name = Segment DAPI

* Description = "Segment nuclei in DAPI channel."

* Input =  { input_file : Volume.Dapi }

* Parameters = { channel : Int,
                 start_z : Int,
                 stop_z : Int }

* Output = { output_file : Volume.Segm }

Remarks:
* segment.py currently takes directory as it wants to save both TIF and UFF outputs.


## CLASSIFY

### classify_panCK.py

* Name = Classify panCK

* Description = "Classify panCK cells."

* Input =  { input_file : Volume,
             segment_dir : Str to Volume.Segm-s }

* Parameters = { channel : List of Volume.PanCK,
                 start_z : Int,
                 stop_z : Int }

* Output = { output_file : Volume.Class.PanCK }

Remarks:
* classify_panCK.py currently takes directory as it wants to save both TIF and UFF outputs.
* For now we assume that panck_channel is single channel file.

### classify_ki67.py

* Name = Classify ki67

* Description = "Classify ki67 cells."

* Input =  { input_file : Volume,
             segment_dir : Str to Volume.Segm-s }

* Parameters = { channel : List of Volume.Ki67,
                 start_z : Int,
                 stop_z : Int }

* Output = { output_file : Volume.Class.Ki67 }

 Remarks:
* classify_ki67.py currently takes directory as it wants to save both TIF and UFF outputs.
* For now we assume that ki67_channel is single channel file.

### classify_pdl1.py

* Name = Classify pdl1

* Description = "Classify pdl1 cells."

* Input = { input_file: Volume,
            segment_dir : Str to Volume.Segm-s }

* Parameters = { channels : List of Volume.pdl1,
                 start_z : Int,
                 stop_z : Int }

* Output = { output_file : Volume.Class.pdl1 }

 Remarks:
* classify_pdl1.py currently takes directory as it wants to save both TIF and UFF outputs.

### classify_cd8.py

* Name = Classify cd8

* Description = "Classify cd8 cells."

* Input = { input_file : Volume,
            segment_dir: Str to Volume.Segm-s }

* Parameters = { channel : List of Volume.cd8,  
                 start_z : Int,
                 stop_z : Int }

* Output = { output_file : Volume.Class.cd8 }

 Remarks:
* classify_cd8.py currently takes directory as it wants to save both TIF and UFF outputs.


## STATISTICS

### stats_nuclei.py

* Name = Stats nuclei

* Description = "Calculate nuclei statistics for provided datasets."

* Input = { input_file : Volume,
            classes : Dict of Str: Volume.Class,
            segment_dir : Str  }

* Parameters = { channels : List of Ints,
                 channels_names : List of Str,
                 start_z : Int, 
                 stop_z : Int,

* Output = { output_file : Csv.Volume, 
             output_file : Csv.Morphology, 
             output_file : Csv.Intensity, 
             output_file : Csv.AllStats,
             output_file : Png.Scatterplots}

### spatial_stats.py

* Name = Spatial stats

* Description = "Calculate spatial statistics."

* Input = { input_file : Volume,
            segment_dir : Str, 
            tested_classes_paths : List of Volume.Class,
            ref_class_path: Volume.Class }

* Parameters = { ref_plot_name: Str    
                 tested_classes_names : List of Str,
                 filter_double_positives : boolean }

* Output = { output_file : Png.Lineplots,
             output_file : Csv.Distance, 
             output_file : Csv.DistanceToEmpty }



## EXPORT

### export_imaris.py

* Name = Export IMARIS

* Description = "Export selected channels to IMARIS file"

* Input = { inputs : List of Datasets}

* Dataset =  { series_path : Volume,
               channel_name : Str,
               channel : Int,
               color : Str }

* Output = { output_path : IMARIS }

Remarks:
`color` is optional and if not given segmentation is assumed.