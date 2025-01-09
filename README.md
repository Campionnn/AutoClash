# AutoClash
## Jujutsu Infinite "macro" to automatically win domain clashes
### Does NOT inject or modify the game in any way. It only takes screenshots of the game to automatically detect when a domain clash is occurring and when to click

## Notes
- Only extensively tested on 1080p and 1440p resolutions, but should work on other resolutions as well
- May have inconsistencies if your game does not run at a consistent 30fps
- Currently only works for Windows, but can maybe support other platforms if requested (NOT MOBILE)

## Running from pre-built binaries
1. Download `AutoClash.exe` from the latest release [here](https://github.com/Campionnn/AutoClash/releases/latest)
2. Run the executable
3. Keep the console window open in the background while you play the game

## Building from source
1. Clone the repository
2. Two options for building
- For building without static linking (will need to have the opencv dll in the same folder as the executable or in the PATH)
  1. Follow the instructions to install OpenCV for Rust [here](https://github.com/twistedfall/opencv-rust/blob/master/INSTALL.md)
  2. Set the environment variables in the instructions
  3. Run `cargo build --release`
  4. The executable will be in `target/release/AutoClash.exe`
- For building with static linking (will have a larger executable size, but will not need the opencv dll)
  - Note: These instructions might be slightly off because I went through a very long debugging process to get this to work and I don't remember the exact steps
  - Will need to install CMake before (and maybe other dependencies)
  1. Download the following zip files for OpenCV [opencv-4.8.1.zip](https://github.com/opencv/opencv/archive/refs/tags/4.8.1.zip) and [opencv_contrib-4.8.1.zip](https://github.com/opencv/opencv_contrib/archive/refs/tags/4.8.1.zip)
  2. Unzip both of these files in a folder so it looks like this:
  ```
    opencv
    ├── opencv-4.8.1
    └── opencv_contrib-4.8.1
    ```
  3. Create a new `build` folder in the `opencv` folder
  4. Open a command prompt in the `build` folder
  5. Run the following commands and change the `D:/opt/opencv` to the path where you want to install OpenCV
  ```
  cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=NO -DCMAKE_INSTALL_PREFIX="D:/opt/opencv" -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF -DWITH_V4L=OFF  -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DOPENCV_EXTRA_MODULES_PATH="../opencv_contrib-4.8.1/modules" ../opencv-4.8.1
  cmake --build . --target install --config Release --parallel 8
  cmake --install . --prefix D:/opt/opencv
    ```
  6. Set the following environment variables
  ```
  OPENCV_MSVC_CRT=static
  OPENCV_INCLUDE_PATHS=D:/opt/opencv/include
  OPENCV_LINK_LIBS=opencv_core481,opencv_imgcodecs481,opencv_imgproc481,opencv_highgui481,ippiw,ippicvmt,ittnotify,zlib
  OPENCV_LINK_PATHS=D:/opt/opencv/staticlib
    ```
  7. Run `cargo build --release`
  8. The executable will be in `target/release/AutoClash.exe`