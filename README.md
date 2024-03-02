# face-swap

This is an OpencV-based face swapping program. 

![](https://github.com/lizhunan/asset/blob/main/face-swap/pre.gif?raw=true)

<p align="lift">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-4caf50.svg" alt="License"></a>
</a>
</p>

## Requirement

The code has used as few complex third-party libraries as possible, and tried to reduce the complexity of setting up the environment. However, there are still some required libraries that need to be imported:

- opencv-python           4.7.0.72
- dlib                    19.24.2
- numpy                   1.21.6
- matplotlib              3.5.3

As a real-time face-swapping program, some necessary hardware is required, and the author's test environment is shown in the following table:

|CPU|GPU|Memory|OS|Camera|
|---|---|---|---|---|
|Intel(R) Core(TM) i7-1065G CPU @ 1.30GHz|Intel(R) Iris(R) Plus Graphics(Inessential)|16G|Windows 11|build-in(Essential)|

## How to Use

1. Clone the code from Github.
2. Download the `shape_predictor_68_face_landmarks.dat` which is a model file provided by dlib for face detection with 68 face landmarks.
3. Move the model file to the `./model`.
4. Run:
     `python run.py data/target_01.jpg`. 

The `target` is the only parameter that specifies the base image to be fused. In this example, I chose the photo of the Mona Lisa(`./data/target_01.jpg`).

## License and Citations

The source code is published under MIT license, see [license file](./LICENSE) for details.