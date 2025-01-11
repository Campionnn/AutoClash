use winapi::shared::windef::{RECT, HBITMAP};
use winapi::um::winuser::{FindWindowA, GetWindowRect, GetDC, ReleaseDC};
use winapi::um::wingdi::{
    CreateCompatibleDC, CreateCompatibleBitmap, SelectObject, DeleteDC, DeleteObject,
    BitBlt, GetDIBits, BITMAPINFO, BITMAPINFOHEADER, DIB_RGB_COLORS, SRCCOPY,
};
use std::ptr::null_mut;
use std::mem::zeroed;
use opencv::{core::{self, Mat, MatTraitConst, Point, Scalar, Vector, BORDER_DEFAULT}, imgproc, prelude::*,
imgcodecs, highgui
};
use opencv::core::{Rect, Rect_, Vec3b};
use enigo::{Enigo, Button, Settings, Direction, Mouse};

const BGR_VALUES: [[i32; 3]; 4] = [
    [81, 78, 244],   // pointer
    [237, 188, 48],  // click_perfect
    [129, 57, 33],   // click_good
    [43, 19, 11],    // click_bad
];

const POINTER_LOW: [i32; 3] = [60, 58, 148];
const POINTER_HIGH: [i32; 3] = [81, 78, 244];
const PERFECT_LOW: [i32; 3] = [150, 124, 30];
const PERFECT_HIGH: [i32; 3] = [237, 188, 50];
const GOOD_LOW: [i32; 3] = [92, 41, 24];
const GOOD_HIGH: [i32; 3] = [129, 57, 33];
const BAD_LOW: [i32; 3] = [39, 17, 10];
const BAD_HIGH: [i32; 3] = [43, 19, 11];

fn buf_to_mat(buf: Vec<u32>, width: i32, height: i32) -> opencv::Result<Mat> {
    // Convert buf into a Vec for BGRA format
    let mut bgra_buf: Vec<u8> = Vec::with_capacity((width * height * 4) as usize);

    for &pixel in &buf {
        let b = (pixel & 0xFF) as u8;
        let g = ((pixel >> 8) & 0xFF) as u8;
        let r = ((pixel >> 16) & 0xFF) as u8;
        let a = ((pixel >> 24) & 0xFF) as u8;

        bgra_buf.push(b);
        bgra_buf.push(g);
        bgra_buf.push(r);
        bgra_buf.push(a);
    }

    // Create a Mat from the BGRA buffer
    let mat = unsafe {
        Mat::new_rows_cols_with_data_unsafe(
            height,
            width,
            core::CV_8UC4,
            bgra_buf.as_ptr() as *mut _,
            core::Mat_AUTO_STEP,
        )?
    };

    let mat = mat.clone();

    // Convert to BGR
    let mut bgr_mat = Mat::default();
    imgproc::cvt_color(&mat, &mut bgr_mat, imgproc::COLOR_BGRA2BGR, 0)?;

    Ok(bgr_mat)
}

fn screenshot() -> opencv::Result<Mat> {
    let window_title = "Roblox\0";
    let hwnd = unsafe { FindWindowA(null_mut(), window_title.as_ptr() as *const i8) };
    if hwnd.is_null() {
        return Ok(Mat::default());
    }

    let mut rect: RECT = unsafe { zeroed() };

    unsafe {
        if GetWindowRect(hwnd, &mut rect) == 0 {
            return Ok(Mat::default());
        }

        let width = rect.right - rect.left;
        let height = rect.bottom - rect.top;
        if width <= 0 || height <= 0 {
            return Ok(Mat::default());
        }

        let hdc_window = GetDC(hwnd);
        if hdc_window.is_null() {
            return Ok(Mat::default());
        }

        // Create a memory device context compatible with the window DC
        let hdc_mem = CreateCompatibleDC(hdc_window);
        if hdc_mem.is_null() {
            ReleaseDC(hwnd, hdc_window);
            return Ok(Mat::default());
        }

        // Create a compatible bitmap for the memory DC
        let hbitmap: HBITMAP = CreateCompatibleBitmap(hdc_window, width, height);
        if hbitmap.is_null() {
            DeleteDC(hdc_mem);
            ReleaseDC(hwnd, hdc_window);
            return Ok(Mat::default());
        }

        // Select the compatible bitmap into the memory DC
        let old_bitmap = SelectObject(hdc_mem, hbitmap as winapi::shared::windef::HGDIOBJ);
        if old_bitmap.is_null() {
            DeleteObject(hbitmap as _);
            DeleteDC(hdc_mem);
            ReleaseDC(hwnd, hdc_window);
            return Ok(Mat::default());
        }

        // Perform a BitBlt from the window DC to the memory DC
        if BitBlt(hdc_mem, 0, 0, width, height, hdc_window, 0, 0, SRCCOPY) == 0 {
            SelectObject(hdc_mem, old_bitmap);
            DeleteObject(hbitmap as _);
            DeleteDC(hdc_mem);
            ReleaseDC(hwnd, hdc_window);
            return Ok(Mat::default());
        }

        // Prepare the BITMAPINFO structure
        let mut bitmap_info = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: width,
                biHeight: -height,
                biPlanes: 1,
                biBitCount: 32,
                biCompression: 0,
                ..zeroed()
            },
            ..zeroed()
        };

        let mut buf: Vec<u32> = vec![0; (width * height) as usize];

        // Retrieve the pixel data
        GetDIBits(
            hdc_mem,
            hbitmap,
            0,
            height as u32,
            buf.as_mut_ptr() as *mut _,
            &mut bitmap_info,
            DIB_RGB_COLORS,
        );

        // Cleanup
        SelectObject(hdc_mem, old_bitmap);
        DeleteObject(hbitmap as _);
        DeleteDC(hdc_mem);
        ReleaseDC(hwnd, hdc_window);

        let bgr_mat = buf_to_mat(buf, width, height).expect("Failed to convert buffer to Mat!");
        Ok(bgr_mat)
    }
}

fn filter_image_by_bgr(image: &Mat, bgr_values: &[[i32; 3]; 4], tolerance: i32) -> opencv::Result<Mat> {
    let mut mask = Mat::zeros(image.rows(), image.cols(), core::CV_8UC1)?.to_mat()?;

    for bgr in bgr_values {
        let lower_bound = Vector::from_slice(&[
            (bgr[0] - tolerance).max(0) as u8,
            (bgr[1] - tolerance).max(0) as u8,
            (bgr[2] - tolerance).max(0) as u8,
        ]);
        let upper_bound = Vector::from_slice(&[
            (bgr[0] + tolerance).min(255) as u8,
            (bgr[1] + tolerance).min(255) as u8,
            (bgr[2] + tolerance).min(255) as u8,
        ]);

        let mut temp_mask = Mat::default();
        core::in_range(&image, &lower_bound, &upper_bound, &mut temp_mask)?;

        let mut new_mask = Mat::default();
        core::bitwise_or(&mask, &temp_mask, &mut new_mask, &Mat::default())?;
        mask = new_mask;
    }

    let mut result = Mat::default();
    core::bitwise_and(&image, &image, &mut result, &mask)?;
    Ok(result)
}

fn find_clash(image: &Mat) -> opencv::Result<Option<(i32, i32, i32, i32)>> {
    let mut filtered_image = filter_image_by_bgr(&image, &BGR_VALUES, 3)?;

    let kernel4x4 = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        core::Size::new(4, 4),
        Point::new(-1, -1),
    )?;

    let mut temp_filtered_image = Mat::default();
    imgproc::morphology_ex(
        &filtered_image,
        &mut temp_filtered_image,
        imgproc::MORPH_OPEN,
        &kernel4x4,
        Point::new(-1, -1),
        1,
        BORDER_DEFAULT,
        Scalar::default(),
    )?;
    filtered_image = temp_filtered_image;

    let mut gray = Mat::default();
    imgproc::cvt_color(&filtered_image, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    let mut binary;
    let mut binary_temp = Mat::default();
    imgproc::threshold(&gray, &mut binary_temp, 1.0, 255.0, imgproc::THRESH_BINARY)?;
    binary = binary_temp; // Update binary

    let mut dilated_binary = Mat::default();
    let kernel5x5 = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        core::Size::new(5, 5),
        Point::new(-1, -1),
    )?;
    imgproc::dilate(
        &binary,
        &mut dilated_binary,
        &kernel5x5,
        Point::new(-1, -1),
        4,
        BORDER_DEFAULT,
        Scalar::default(),
    )?;
    binary = dilated_binary;

    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours(
        &binary,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    let mut areas: Vec<(usize, f64)> = contours
        .iter()
        .enumerate()
        .map(|(i, c)| (i, imgproc::contour_area(&c, false).unwrap()))
        .collect();
    areas.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (idx, _) in areas {
        let rect = imgproc::bounding_rect(&contours.get(idx)?)?;
        let area = rect.area();
        if area < 1000 {
            continue;
        }
        let aspect_ratio = rect.width as f64 / rect.height as f64;
        if (5.3..=6.4).contains(&aspect_ratio) {
            let clash_rect = image.roi(rect)?;
            
            let clash_rect = clash_rect.roi(Rect::new(0, rect.height / 2, rect.width, 1))?;
            
            let mut pointer_count = 0;
            let mut perfect_count = 0;
            let mut good_count = 0;
            let mut bad_count = 0;
            
            let image = clash_rect.try_clone()?;
            
            for col in 0..image.cols() {
                let pixel = image.at_2d::<Vec3b>(0, col)?;
                if pixel[0] >= POINTER_LOW[0] as u8 && pixel[0] <= POINTER_HIGH[0] as u8 &&
                    pixel[1] >= POINTER_LOW[1] as u8 && pixel[1] <= POINTER_HIGH[1] as u8 &&
                    pixel[2] >= POINTER_LOW[2] as u8 && pixel[2] <= POINTER_HIGH[2] as u8 {
                    pointer_count += 1;
                } else if pixel[0] >= PERFECT_LOW[0] as u8 && pixel[0] <= PERFECT_HIGH[0] as u8 &&
                    pixel[1] >= PERFECT_LOW[1] as u8 && pixel[1] <= PERFECT_HIGH[1] as u8 &&
                    pixel[2] >= PERFECT_LOW[2] as u8 && pixel[2] <= PERFECT_HIGH[2] as u8 {
                    perfect_count += 1;
                } else if pixel[0] >= GOOD_LOW[0] as u8 && pixel[0] <= GOOD_HIGH[0] as u8 &&
                    pixel[1] >= GOOD_LOW[1] as u8 && pixel[1] <= GOOD_HIGH[1] as u8 &&
                    pixel[2] >= GOOD_LOW[2] as u8 && pixel[2] <= GOOD_HIGH[2] as u8 {
                    good_count += 1;
                } else if pixel[0] >= BAD_LOW[0] as u8 && pixel[0] <= BAD_HIGH[0] as u8 &&
                    pixel[1] >= BAD_LOW[1] as u8 && pixel[1] <= BAD_HIGH[1] as u8 &&
                    pixel[2] >= BAD_LOW[2] as u8 && pixel[2] <= BAD_HIGH[2] as u8 {
                    bad_count += 1;
                }
            }
            
            if pointer_count > 0 && perfect_count > 2 && good_count > 5 && bad_count > image.size()?.width / 3 {
                return Ok(Some((rect.x, rect.y, rect.width, rect.height)));
            }
        }
    }

    Ok(None)
}

fn  get_pixel_vec(rect: Rect) -> opencv::Result<Vec<usize>> {
    let image = screenshot()?;
    let image = image.roi(rect)?.try_clone()?;

    let mut pixels = Vec::new();
    for col in 0..image.cols() {
        let pixel = image.at_2d::<Vec3b>(0, col)?;
        let mut found = false;
        if pixel[0] >= POINTER_LOW[0] as u8 && pixel[0] <= POINTER_HIGH[0] as u8 &&
            pixel[1] >= POINTER_LOW[1] as u8 && pixel[1] <= POINTER_HIGH[1] as u8 &&
            pixel[2] >= POINTER_LOW[2] as u8 && pixel[2] <= POINTER_HIGH[2] as u8 {
            pixels.push(0);
            found = true;
        } else if pixel[0] >= PERFECT_LOW[0] as u8 && pixel[0] <= PERFECT_HIGH[0] as u8 &&
            pixel[1] >= PERFECT_LOW[1] as u8 && pixel[1] <= PERFECT_HIGH[1] as u8 &&
            pixel[2] >= PERFECT_LOW[2] as u8 && pixel[2] <= PERFECT_HIGH[2] as u8 {
            pixels.push(1);
            found = true;
        } else if pixel[0] >= GOOD_LOW[0] as u8 && pixel[0] <= GOOD_HIGH[0] as u8 &&
            pixel[1] >= GOOD_LOW[1] as u8 && pixel[1] <= GOOD_HIGH[1] as u8 &&
            pixel[2] >= GOOD_LOW[2] as u8 && pixel[2] <= GOOD_HIGH[2] as u8 {
            pixels.push(2);
            found = true;
        } else if pixel[0] >= BAD_LOW[0] as u8 && pixel[0] <= BAD_HIGH[0] as u8 &&
            pixel[1] >= BAD_LOW[1] as u8 && pixel[1] <= BAD_HIGH[1] as u8 &&
            pixel[2] >= BAD_LOW[2] as u8 && pixel[2] <= BAD_HIGH[2] as u8 {
            pixels.push(3);
            found = true;
        }
        else if pixel[0] >= 200 && pixel[1] >= 200 && pixel[2] >= 200 &&
            (col as f64 / (image.cols() as f64) < 0.05 && col as f64 / (image.cols() as f64) > 0.95) {
            found = true;
        }
        if !found {
            pixels.push(9);
        }
    }

    Ok(pixels)
}

fn find_indices(bar: &[usize], value: usize) -> Option<(usize, usize)> {
    let start = bar.iter().position(|&x| x == value)?;
    let end = bar.iter().rposition(|&x| x == value)?;
    Some((start, end))
}

fn find_pointer_pos(rect: Rect_<i32>, ended: &mut bool) -> opencv::Result<usize> {
    let mut count = 0;
    let mut pointer_start = 0;

    loop {
        if count > 10 {
            *ended = true;
            break;
        }

        let meter_vec = get_pixel_vec(rect)?;

        if meter_vec.iter().filter(|&&x| x == 0).count() == 0 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            count += 1;
            continue;
        }

        let (start, end) = find_indices(&meter_vec, 0).unwrap();
        pointer_start = (start + end) / 2;
        break;
    }

    Ok(pointer_start)
}

fn find_speed(rect: Rect_<i32>, ended: &mut bool) -> opencv::Result<f64> {
    let time1 = std::time::Instant::now();
    let pos1 = find_pointer_pos(rect, &mut *ended)?;
    std::thread::sleep(std::time::Duration::from_millis(30));

    let time2 = std::time::Instant::now();
    let pos2 = find_pointer_pos(rect, &mut *ended)?;

    let elapsed1 = time2.duration_since(time1).as_millis() as f64;
    let speed1 = (pos2 as i32 - pos1 as i32) as f64 / elapsed1;

    Ok(speed1)
}

fn main() -> opencv::Result<()> {
    let mut enigo = Enigo::new(&Settings::default()).unwrap();
    println!("Waiting for domain clash to start");

    loop {
        let image = screenshot()?;
        if image.empty() {
            std::thread::sleep(std::time::Duration::from_millis(1000));
            continue;
        }

        match find_clash(&image)? {
            Some((x, y, w, h)) => {
                println!("Detected Clash");
                let rect = Rect::new(x, y + h / 2, w, 1);
                // std::thread::sleep(std::time::Duration::from_millis(100));

                let mut count = 0;
                let mut ended = false;
                let mut speeds: Vec<f64> = Vec::new();

                loop {
                    if count > 10 {
                        println!("Clash ended");
                        break;
                    }

                    let velocity = find_speed(rect, &mut ended)?;
                    speeds.push(velocity.abs());

                    if ended {
                        println!("Clash ended");
                        break;
                    }

                    let mut speeds = speeds.clone();
                    speeds.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let speed = speeds[speeds.len() / 2];
                    let velocity = if velocity > 0.0 { speed } else { -speed };

                    let meter_vec = get_pixel_vec(rect)?;

                    if meter_vec.iter().filter(|&&x| x == 0).count() == 0 ||
                        meter_vec.iter().filter(|&&x| x == 1).count() == 0 {
                        std::thread::sleep(std::time::Duration::from_millis(10));
                        count += 1;
                        continue;
                    }
                    count = 0;

                    let (start, end) = find_indices(&meter_vec, 0).unwrap();
                    let current_pointer_pos = (start as i32 + end as i32) / 2;

                    let (start, end) = find_indices(&meter_vec, 1).unwrap();
                    let click_region_center = (start as i32 + end as i32) / 2;

                    let distance;
                    if current_pointer_pos < click_region_center {
                        distance = if velocity > 0.0 {
                            click_region_center - current_pointer_pos
                        } else {
                            (click_region_center - current_pointer_pos) + (current_pointer_pos * 2)
                        };
                    } else {
                        distance = if velocity > 0.0 {
                            (current_pointer_pos - click_region_center) + ((meter_vec.len() as i32 - current_pointer_pos) * 2)
                        } else {
                            current_pointer_pos - click_region_center
                        };
                    }
                    
                    let time = (distance as f64 / speed) - 0.0;

                    if time < 200.0 {
                        std::thread::sleep(std::time::Duration::from_millis(time as u64));
                        println!("speed: {}, velocity: {}, distance: {}", speed, velocity, distance);
                        println!("time: {}", time);
                        enigo.button(Button::Left, Direction::Click).expect("");
                        speeds.clear();
                    }
                }
            }
            None => {}
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}