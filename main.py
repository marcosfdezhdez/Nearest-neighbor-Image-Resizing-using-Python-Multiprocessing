#IMAGE RESIZING BY MARCOS FERNANDEZ
from PIL import Image
import numpy as np
import time
from multiprocessing import Pool, cpu_count


def resize_nn_sequential(img_in: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """
    Sequential nearest-neighbor image resize.
    img_in: numpy array with shape (H_in, W_in, C)
    new_h, new_w: target height and width
    """
    H_in, W_in, C = img_in.shape

    # I allocate the output image with the desired resolution
    img_out = np.empty((new_h, new_w, C), dtype=np.uint8)

    # Double loop over the output image
    # For each output pixel I compute the corresponding location in the input image using a simple "rule of three"
    for y_out in range(new_h):
        y_in = int(y_out * H_in / new_h)

        for x_out in range(new_w):
            x_in = int(x_out * W_in / new_w)

            # Copy the entire RGB vector (all channels at once)
            img_out[y_out, x_out] = img_in[y_in, x_in]

    return img_out


def _resize_chunk(args):
    """
    Worker function executed in each process.
    It computes a contiguous block of rows [y_start, y_end) of the output image.

    I keep all arguments in a single tuple because Pool.map only passes one object.
    """
    img_in, new_h, new_w, y_start, y_end = args
    H_in, W_in, C = img_in.shape

    # Height of this chunk 
    chunk_h = y_end - y_start

    # Local output buffer for this chunk. Each process writes only here, so there are no race conditions
    chunk = np.empty((chunk_h, new_w, C), dtype=np.uint8)

    # local_y runs from 0 to chunk_h-1, while y_out is the global row index in the final output image
    for local_y, y_out in enumerate(range(y_start, y_end)):
        y_in = int(y_out * H_in / new_h)

        for x_out in range(new_w):
            x_in = int(x_out * W_in / new_w)
            chunk[local_y, x_out] = img_in[y_in, x_in]

    # I return the starting row so that the main process knows where to place this chunk inside the final image
    return y_start, chunk


def resize_nn_parallel(img_in: np.ndarray, new_h: int, new_w: int, n_procs: int) -> np.ndarray:
    """
    Parallel nearest-neighbor resize using Python multiprocessing.

    Parallelization strategy:
    - I partition the output image by rows.
    - Each process computes a disjoint block of rows.
    - Finally, the main process assembles all row blocks into the final image.
    """
    H_in, W_in, C = img_in.shape

    # Global output buffer that will hold the final image.
    # Only the main process writes here, after collecting all chunks
    img_out = np.empty((new_h, new_w, C), dtype=np.uint8)

    # Compute the chunk size: how many rows per process (ceil division)
    chunk_size = (new_h + n_procs - 1) // n_procs

    # Build the list of tasks. Each task describes one block of rows
    tasks = []
    for p in range(n_procs):
        y_start = p * chunk_size
        y_end = min(new_h, (p + 1) * chunk_size)

        if y_start >= new_h:
            break

        # Each task contains all the information the worker needs
        tasks.append((img_in, new_h, new_w, y_start, y_end))

    # I use a process pool to run all the row-chunks in parallel.
    # pool.map() basically:
    #   - sends each task to a worker process
    #   - waits until all workers finish
    #   - returns all the (y_start, chunk) results
    with Pool(processes=n_procs) as pool:
        results = pool.map(_resize_chunk, tasks)

    # Assemble the final image by placing each chunk in the right position
    for y_start, chunk in results:
        y_end = y_start + chunk.shape[0]
        img_out[y_start:y_end, :, :] = chunk

    return img_out


def main():
    # Input and output paths (I keep them simple for the experiments)
    input_path = "input.png"           # put here any test image
    output_seq_path = "output_seq.png"
    output_par_path = "output_par.png"

    # Target resolution for the resized image.
    # I can change this during the experiments to see how the runtime scales (1000, 2000 and 3000 were studied)
    new_w, new_h = 3000, 3000

    # You can change it as you wish (1, 2, 4 and 8 were studied)
    n_procs = 8

    # Load the input image as RGB and convert it to a numpy array
    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img)

    # --- Sequential version ---
    t0 = time.perf_counter()
    resized_seq = resize_nn_sequential(img_np, new_h, new_w)
    t1 = time.perf_counter()
    seq_time = t1 - t0
    print(f"[Sequential]   time = {seq_time:.4f} s")

    # --- Parallel version ---
    t2 = time.perf_counter()
    resized_par = resize_nn_parallel(img_np, new_h, new_w, n_procs)
    t3 = time.perf_counter()
    par_time = t3 - t2
    print(f"[Parallel x{n_procs}] time = {par_time:.4f} s")

    # Validation: the parallel output should be identical to the sequential one
    same_output = np.array_equal(resized_seq, resized_par)
    print("Outputs equal? ", same_output)

    # Save both images (mainly to visually check that the resize looks correct)
    Image.fromarray(resized_seq).save(output_seq_path)
    Image.fromarray(resized_par).save(output_par_path)

    # If I want, I can print the speedup here
    if par_time > 0:
        speedup = seq_time / par_time
        print(f"Speedup ≈ {speedup:.2f}x")


if __name__ == "__main__":

    main()

