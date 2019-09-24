import sys

sys.path.insert(0, '../src')
from im_scale_products import get_run_directory_and_run_file, get_run_parameters

def create_scaled_images_dataset(run_parameters):
    from im_scale_products import scaled_images_dataset
    scaled_images_dataset(run_parameters)

SELECT = {'scaled_images_dataset': create_scaled_images_dataset}

def main():
    run_directory, run_file = get_run_directory_and_run_file(sys.argv)
    run_parameters = get_run_parameters(run_directory, run_file)

    SELECT[run_parameters['method']](run_parameters)

if __name__ == "__main__":
    main()
    