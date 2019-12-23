# scalygrapic - scalable image graphic machine learning
<details>
  <summary> 
   Operation - numerical generator of chaotic, fractal images.
  </summary>
  <br>
  a. Clone this repository
  
  ```bash scripting
    git clone https://github.com/dlanier/scalygraphic.git
  ```
  b. Edit a yaml file to set your image resolutions and number of images
  
Requires Python 3.5 or more
 </details>

------
 # Generate a set of images for train / validate / test 
 (this example is for super-resolution)
<details>
  a. Copy the file scalygraphic/data/run_files/create_scaled_image_set.yml to your run (or test) directory
  ```bash scripting
  #in the directory with the cloned repo
  mkdir -p run_dir/results
  cp scalygraphic/data/run_files/create_scaled_image_set.yml run_dir/anew_image_set.yml
  ```
  b. Edit the newly copied file to set the run parameters for the desired data set.
  ```bash scripting
  #main function in src/scalygraphic.py calls method defined here with these run parameters
  method:               scaled_images_dataset

  #number of pairs of images
  number_of_image_sets: 100
  
  #small scale size
  small_scale_rows:     128
  small_scale_cols:     128
  
  #matching large scale image size
  large_scale_rows:     256
  large_scale_cols:     256

  #where to write the results
  results_directory:    ./run_dir/results

  #max number of iterations for the algorithm (larger is slower)
  it_max:               64
  #image diagonal multiples (larger is slower, smaller may produce artifacts)
  scale_dist:           10

  #false color if true
  greyscale:            False
  #constrain image generation to use one equation only
  use_one_eq:           False
  ```
  c. Call the main function from the command line with the edited .yml file
  ```bash scripting
  #Note that the run_file is in the run_directory
  python3 ./scalygraphic/src/scalygraphic.py -run_directory ./run_dir/ -run_file anew_image_set.yml
  #Check that the hash-named images begin to appear in the run_dir/results directory
  ```
</details>
------
