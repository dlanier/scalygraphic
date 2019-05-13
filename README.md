# scalygrapic - collected graphical equations with special properties to generate images for machine learning

* image generator object has:
  * function name - the equation
  * parameter set - fixed parameters passed to the equation
  * complex plain frame - fixed center point and corners (rotation & plain-scale)
  * pixel size - resolution (pixels per frame width)
* image generator object is:
  * a generator of images that look the same but with different resolutions
  * loadable and saveable
  * printable - parameters to displayable object
* image generator will:
  * return image with specified pixel size
  * write sets of (same_name) images with diffent sizes into directories
  * write sets with rotated shifted or scled variations
  
# Yeah ImageGenerator
