import os
import SimpleITK as sitk
import segmentation_functions as seg_func
import multiprocessing

def splitext(path):
    filename, ext = os.path.splitext(path)
    if ext == ".gz":
        filename, ext = os.path.splitext(filename)
        ext += ".gz"
    return filename, ext


def procces_file(file, clean):
    name, ext = splitext(file)

    print("Proccesing file: {}".format(file))
            
    if ext not in [".mha", ".nrrd", ".nii.gz"]:
        print("WARNING: Omitting file {}".format(file))
    else:
        seg_func.make_mesh(file, name+".obj", clean=clean)

if __name__ == "__main__":

    # input_dir = '/home/taha/Downloads/Panacea/mri_to_ct/mri_to_ct/24_12_24_final/infer/saved'
    input_dir = '/home/taha/Downloads/Panacea/dataset/TEST/CT'
    clean = True

    if os.path.isfile(input_dir):
        procces_file(input_dir, clean)

    elif os.path.isdir(input_dir):
        args_to_parallel = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                args_to_parallel.append([os.path.join(root, file), clean])

        pool = multiprocessing.Pool(processes=7)
        pool.starmap(procces_file, args_to_parallel)
        pool.close()
    else:
        print("ERROR: Input must be file or directory")
        exit(0)
