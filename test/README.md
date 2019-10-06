# Create a scaled images dataset using make from this directory
```
make env_setup
```
Edit ./run_dir/create_scaled_image_set.yml (copied by env above) to set:
* the number of image sets
* scale sizes 
* results dir
* any others execpt "method"

```
make run_scaled_images_dataset
```
Command line messages appear, the images begin to appear in the results dir specified
