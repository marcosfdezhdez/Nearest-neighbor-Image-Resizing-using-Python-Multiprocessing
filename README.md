# Nearest-neighbor-Image-Resizing-using-Python-Multiprocessing

## Project Overview
This project focuses on implementing a simple image resizer using the nearest-neighbor algorithm. The goal is to compare a sequential version with a parallel implementation based on Python’s multiprocessing module. The task consists of scaling an RGB image to a new resolution, which is a very common operation in computer vision pipelines.

## Repository Contents
In the repository you can find a Report which deeply analyses the project, including an introduction of the goals and the characteristics, the explanation of the algorithm used and parallelization strategy, and an analysis of the results and the corresponding discussion. 

## For running the code
Firstly, to run it you must have installed numpy Pillow. You can install it with this command: pip install numpy Pillow.
After that, place the input image (input.png) in the same folder as main.py.
Inside the code, specify the input file name here (line 109):  input_path = "input.png"           # put here any test image 
You can also change the sie of the output image and the number of processors.
Run the program with python main.py
| Parameter     | Description                                                                                 |
| ------------- | ------------------------------------------------------------------------------------------- |
| num_processes | Number of processes used in multiprocessing (1, 2, 4, 8)                                    |
| input_path    | Path to the input image (default: input.png)                                                |
| output_size   | Target dimensions of the resized image (1000×1000, 2000×2000, 3000×3000 in the experiments) |
| scale_factors | Derived from input/output sizes for nearest-neighbor coordinate mapping                     |

Dataset:
The dataset used consists of a single local image (input.png)
