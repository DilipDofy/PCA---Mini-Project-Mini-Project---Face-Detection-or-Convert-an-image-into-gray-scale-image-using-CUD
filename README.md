# PCA-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD

## AIM:
To develop and implement a CUDA-based GPU program that converts a color image into a grayscale image, and to compare the performance between GPU and CPU implementations.

## APPARATUS REQUIRED:
**SOFTWARE**

1.Google Colab

2.Python 3.x

3.Libraries: opencv-python, cupy-cuda12x, numpy

**HARDWARE**

1.NVIDIA GPU (CUDA supported)

2.Internet connection (for Colab runtime)

**Input**

Any color image file (.jpg, .png, etc.)

## THEORY:
Grayscale conversion reduces a three-channel color image (Red, Green, Blue) into a single intensity channel by eliminating color information while preserving luminance. The standard luminance equation used is:

Y=0.299R+0.587G+0.114B

CUDA enables parallel computation by executing thousands of threads simultaneously on GPU cores. Using GPU (via CuPy in Python), pixel-wise grayscale conversion can be computed in parallel, significantly improving performance compared to CPU execution.

## PROCEDURE:
1.Open Google Colab and enable GPU runtime (Runtime → Change runtime type → GPU → Save).

2.Install required libraries: !pip install opencv-python cupy-cuda12x

3.Upload a color image (files.upload() in Colab).

4.Read the image using OpenCV and convert it into a CuPy GPU array.

5.Apply the grayscale formula using GPU parallel computation: Y = 0.114 * B + 0.587 * G + 0.299 * R

6.Convert the result back to the CPU and save it as grayscale_gpu.png.

7.Compare GPU time with CPU time using OpenCV’s cv2.cvtColor().

8.Display the original and grayscale images for visual verification.

## PROGRAM:
```
Developed by: DILIP M P
Reg.no: 212223230048

!pip -q install -U opencv-python "cupy-cuda12x"
!nvidia-smi || echo "GPU not detected"

from google.colab import files
uploaded = files.upload()
in_path = list(uploaded.keys())[0]

import cv2, cupy as cp, numpy as np, time
from google.colab.patches import cv2_imshow

img_bgr = cv2.imread(in_path, cv2.IMREAD_COLOR)
d_img = cp.asarray(img_bgr, dtype=cp.float32)
weights = cp.array([0.114, 0.587, 0.299], dtype=cp.float32)

start = cp.cuda.Event(); end = cp.cuda.Event()
start.record()
gray_gpu = cp.tensordot(d_img, weights, axes=([2],[0]))
gray_gpu = cp.clip(gray_gpu + 0.5, 0, 255).astype(cp.uint8)
end.record(); end.synchronize()
gpu_ms = cp.cuda.get_elapsed_time(start, end)

out_gpu = cp.asnumpy(gray_gpu)
cv2.imwrite("grayscale_gpu.png", out_gpu)

t0 = time.perf_counter()
gray_cpu = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
t1 = time.perf_counter()
cpu_ms = (t1 - t0) * 1000

print(f"\nCPU Time: {cpu_ms:.3f} ms")
print(f"GPU Time: {gpu_ms:.3f} ms")
print(f"Speed-Up ≈ {cpu_ms/gpu_ms:.2f}×")

print("\nOriginal Image:")
cv2_imshow(img_bgr)

print("\nGrayscale Image (GPU Output):")
cv2_imshow(out_gpu)
```

## OUTPUT:

<img width="175" height="68" alt="image" src="https://github.com/user-attachments/assets/6aa12a92-a344-4925-a4bb-37545cb430db" />


## COLOUR IMAGE:

<img width="754" height="451" alt="image" src="https://github.com/user-attachments/assets/e1ca3b7f-ccf0-41a5-b7d8-ae994a762940" />



## GRAYSCALE IMAGE:

<img width="755" height="448" alt="image" src="https://github.com/user-attachments/assets/14dbc066-af37-4dd0-a4bc-9d14e8f0862a" />


## RESULT:
1.The color image was successfully converted into a grayscale image using GPU-based parallel computation.

2.Execution on GPU was significantly faster than CPU for large images.

3.The output image (grayscale_gpu.png) was saved and displayed in Colab.

## CONCLUSION:
The experiment demonstrates how CUDA-based GPU programming can accelerate image processing tasks. By performing pixel-level computations in parallel, the grayscale conversion achieved substantial speed improvement over the traditional CPU method.
