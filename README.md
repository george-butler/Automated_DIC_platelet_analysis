# Fully automated platelet Differential Inference Contrast image analysis via deep learning. 

**Carly Kempster<sup>1,3</sup>, George Butler<sup>1,2,3</sup>, Elina Kuznecova<sup>1</sup>, Kirk A. Taylor<sup>1</sup>, Neline Kriek<sup>1</sup>, Gemma Little<sup>1</sup>, Marcin A. Sowa<sup>1</sup>, Louise J. Johnson<sup>1</sup>, Jonathan M. Gibbins<sup>1</sup>, and Alice Y. Pollitt<sup>1</sup>**

<sup><sup>1</sup>School of Biological Sciences, University of Reading, Reading, UK</sup>

<sup><sup>2</sup>The Brady Urological Institute, Johns Hopkins School of Medicine, Baltimore, USA</sup>

<sup><sup>3</sup>Both authors contributed equally</sup>
	
![segmentation_example](/example_images/DIC_and_colourised_mask.png)
	
This repository enables the morphology of indiviudal platelets to be quantified within a platelet spreading assay. This work is an modification of the brilliant [Usiigaci](https://github.com/oist/Usiigaci) platform originally developed by Tsai *et al.* to track cell migration within phase contrast time-lapse microsopy. We have altered and retrained the mask regional convolutional neural network (Mask R-CNN) to segment the morphology of individual platelets from within Differential Inference Contrast (DIC) images. Furthermore, we have also repurposed the [Usiigaci](https://github.com/oist/Usiigaci) tracking GUI to function as quality control panel by which inaccurately segmented platelets can be removed from future analysis. 


## Installation 
The necessary dependencies are the same as the original Usiigaci platform and as such a thorough installation guide can be found [here](https://github.com/oist/Usiigaci#dependencies).
	

## How to make your own training data and trained network

1. Download the LOCI pulgin
	
	 Assuming that you already have FIJI installed, download the latest LOCI plugin from [here](https://sites.imagej.net/LOCI/plugins/LOCI/). Once downloaded, save the LOCI plugin to the FIJI plugin folder and then restart FIJI. If successful, the ROI Map function should now appear at Plugins>LOCI>ROI Map.

2. Manual curation 
	
	1) Open each ‘.tif’ file with ImageJ by dragging and dropping into the ImageJ toolbar
	2) Click Analyze>Tools>ROI Manager
	3) Select the "show all" and "labels" check boxes on the ROI manager - this will keep the outline of each platelet visible so you can keep track of where you are
	4) Draw around each platelet in the image using the pen-pad and save the outline with the ROI Manager
	5) Once all of the platelets have been drawn round, click Move>Save on the ROI Manager
	6) Save as "RoiSet" in the same directory as the ".tif" file
	7) In the same directory , save the image by selecting File>Save As>PNG
	8) Name the file "phase.png"
	9) Generate the labeled image by selecting Plugins>LOCI>ROI Map
	10) In the same directory, save the colourful image by selecting File>Save As>PNG
	11) Name the file "labeled.png"
	12) You should then have a directory that looks like this:
	   ![training_directory](/example_images/training_file_structure.png)
	13) Finally, repeat for all of the directories within the training set, ensuring the contents of each directory is as the same as the above, and all of the files are named correctly. 

3. Preprocess the training data
	
	Run the preprocess_data.py in "/mask_R-CNN/preprocess_data.py" to convert the coloured mask, "labeled.png", into gray scale 8 bit image, "instance_ids.png". The location of the training data, "mask_R-CNN/training_data", can be set in line 44 
	
4. Training the network
	
	Finally run the train.py script in "/mask_R-CNN/train.py" to train the segmentation model. The location of the training data directory can be set in line 258. Likewise, the location of the final tracking weights can be set in line 289.

## Mask R-CNN Segmentation

1. Download the trained weights
	
	The 3 trained network weights be found at "/mask_R-CNN/DIC_training_weights" and relate to networks that have been trained with 120 manual curated images.

	Note: Platelets were imaged by Köhler illuminated Nomarski differential interference contrast (DIC) optics using a Nikon eclipse Ti2 inverted microscope, equipped with a Nikon DS-Qi2 camera, and visualised using a 100x oil immersion objective lens. NIS Elements software was used for image capture. The original 16-bit DIC images with dimensions 2424x2424 were then rescaled and coverted to 970x970 8-bit images in ImageJ. 


2. Data structure
	
	The DIC images collected from a given experiment can be stored, and then processed, from within one directory. However, within the parent directory an individual subdirectory should still be used for each DIC image as seen in the "/mask_R-CNN/inference_example/"


3. Segmenting an image
	
	The individual images are then segmented via the inference.py script in "/mask_R-CNN/inference.py".
	
	The path to the parent directory can be set at line 287. The script then searches through each of the nested directories to segment each of the individual DIC images.
	
	The paths to the trained model weights are then set in lines 290:292

3. Running the inference script
	
	Navigative to the directory within which the inference.py script is stored and type 'python inference.py' into the terminal. If you are not familar with navigating via the terminal then please refer to this beginners [guide](https://ubuntu.com/tutorials/command-line-for-beginners#1-overview), or [here](https://docs.microsoft.com/en-us/windows-server/administration/windows-commands/windows-commands#command-line-reference-a-z) if using a Windows machine.

4. Output
	
	The segmented output will then be stored in a directory with the suffix "_mask_avg" that forms the basis of the qualtify control phase, an example of the output structure can be seen in "/DIC_example_data/test1/"
	
## Quality control panel

The quality control panel was based on the Usiigaci tracking GUI and therefore many of the same libraries must also be installed. 

##### Prerequisites:
1. Install required packages
	
	pip install pyqt5 pyqtgraph pims pandas pillow imageio--ffmpeg scikit-image==0.16.2


2. Modify the Imageitem class of PyQtGraph
	
	Overwrite the ImageItem.py into python/site-packages/pyqtgraph/graphicsItems folder

	If working in the virtual environment *tensorflow*:
	it will be under /home/username/tensorflow/lib/python3.5/site-packages/pyqtgraph/graphicitems/

3. Posthoc filter 
	
	A posthoc filter was applied such that objects containing an area of <250 pixels, or any pixels within a 10 pixel range of the image edge, were automatically removed to increase the speed of the quality control process. This filter can be adjusted by modifying lines 22 - 35 in the cell_features.py script, "/platelet_quality_control/cell_features.py"

### Using the qualtify control panel:

1. Launch the quality control panel 
										
	The quality control panel can be launch by navigating to the quality control directory, "/platelet_quality_control", and typing "python qc_panel.py". 


2. Selecting an image
										
	Once the panel has been launch, an indivdual image can be selected my clicking on the "Open Folder" button at the top right of the panel. A pop out window will then launch enabling you to navigate to the location of the indivdual image. An example can be seen by navigating to "/DIC_example_data/test1/raw/" and then clicking "open" at the bottom right of the pop out window. If the "raw" and "raw_mask_avg" subdirectories are stored with in the same parent directory then the raw DIC image will appear on the left of the quality control panel and the segmented image will appear on the right.

3. Run the posthoc filter
										
	Next, if you click the "Colour Mask" button on the upper right of the panel the segmented image will be automatically filtered and a colourised image will appear inplace of the previous greyscale image. Individual platelets can then be further removed by deselecting a given ID on the right hand panel

4. Saving the output
										
	Finally, the output can be saved by clicking the "Save selection" button at the top right the quality control panel which saves the colourised mask and the morphological metrics, "morphological_metrics.csv".	Note that the colourised mask can also be saved without the overlayed platelet ids by deselecting the "Show id's" button in the upper left hand corner of the quality control panel. Similarly the pixel to micro scale can also be adjusted by changing the "Pixel scale" underneath the "Show id's" button. 			

## Licenses
Usiigaci is released under MIT license 

TensorFlow is open-sourced under Apache 2.0 opensource license. 

Keras is released under MIT license

The copyright of PyQT belong to Riverbank Computing ltd.

Pandas is released under [BSD 3-Clause License](http://pandas.pydata.org/pandas-docs/stable/overview.html?highlight=bsd). Copyright owned by AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team. 

Trackpy is released under [BSD 3-Clause License](https://github.com/soft-matter/trackpy/blob/master/LICENSE). Copyright owned by trackpy contributors.

NumPy and SciPy are released under BSD-new License

Scikit-image is released under [modified BSD license](https://github.com/scikit-image/scikit-image)

PIMS is released under [modified BSD license](https://github.com/soft-matter/pims/blob/master/license.txt)

Matplotlib is released under [Python Software Foundation (PDF) license](https://matplotlib.org/)

Seaborn is released under [BSD 3-clause license](https://github.com/mwaskom/seaborn/blob/master/LICENSE)

ffmpeg is licensed under [LGPL 2.1 license](https://www.ffmpeg.org/legal.html)

PyQtGraph is released under MIT license
