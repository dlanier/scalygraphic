# scalygrapic - collected graphical equations with special properties to generate images for machine learning
<details>
  <summary> 
   Numerical generator of chaotic, fractal images
  </summary>
  <br>
  a. Clone this repository
  
  ```bash scripting
    git clone https://github.com/dlanier/scalygraphic.git
  ```
  b. Use a jupyter notebook to edit a yaml file to set your image resolutions and number of images
  
Requires Python 3.5 or more
 </details>

------
 
* image generator object has:
  * function name - the equation
  * parameter set - fixed parameters passed to the equation
  * complex plain frame - fixed center point and corners (rotation & plain-scale)
  * colormap scheme
  * pixel size - resolution (pixels per frame width)
* image generator object is:
  * a generator of images that look the same but with different resolutions
  * loadable and saveable
  * printable - parameters to displayable object
* image generator will:
  * return image with specified pixel size
  * write sets of (same_name) images with diffent sizes into directories
  * write sets with rotated shifted or scled variations
  * wirte sets with color variations
* image generator construction with:
  * data-set parameters
  * project-time-place-owner identity parameters
  * directory struture - layout parameters
* image generator may:
  * load-save parameters with dataframes
  * report fractal dimension and other image statistics for any parameter set
  
  
# Yeah ImageGenerator
