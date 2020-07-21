# NucleiSeg

Set of python scripts for nuclei segmentation/tracking in microscopic 2D/3D images, with use of parallel processing(joblib) and based on edge detection method,
works with tiff files(multitiff). There are several modes for different kinds of data input, use segmentation_controller(mode, path_to_files):

'tile': returns list of seperate 3d nuclei in numpy arrays extracted from a tilescan(previolsy stiched), can work with multiple channels

'live': returns list of seperate 3d nuclei in numpy arrays but as a timeseries of tracked nuclei, extracted from multiposition imaging(multiple files), multiple channels

'data','NB': loads previously segmented images of nuclei saved in multiple files \n

