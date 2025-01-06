use winapi::shared::windef::{RECT, HBITMAP};
use winapi::um::winuser::{FindWindowA, GetWindowRect, GetDC, ReleaseDC};
use winapi::um::wingdi::{
    CreateCompatibleDC, CreateCompatibleBitmap, SelectObject, DeleteDC, DeleteObject,
    BitBlt, GetDIBits, BITMAPINFO, BITMAPINFOHEADER, DIB_RGB_COLORS, SRCCOPY,
};
use std::ptr::null_mut;
use std::mem::zeroed;
use opencv::{core::{self, Mat, MatTraitConst, Point, Scalar, Vector, BORDER_DEFAULT}, imgproc, prelude::*};
use opencv::boxed_ref::BoxedRef;
use opencv::core::{Rect, Vec3b};
use enigo::{Enigo, Button, Settings, Direction, Mouse};

const BGR_VALUES: [[i32; 3]; 4] = [
    [81, 78, 244],   // pointer
    [237, 188, 48],  // click_perfect
    [129, 57, 33],   // click_good
    [43, 19, 11],    // click_bad
];

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

// fn show_image(image: &Mat) -> opencv::Result<()> {
//     highgui::imshow("Image", image)?;
//     highgui::wait_key(0)?;
//     Ok(())
// }

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

fn count_bgr_values(image: &BoxedRef<Mat>, bgr_values: &[[i32; 3]; 4], tolerance: u8) -> opencv::Result<Vec<i32>> {
    let mut counts = vec![0; bgr_values.len()];

    for row in 0..image.rows() {
        for col in 0..image.cols() {
            let pixel = image.at_2d::<Vec3b>(row, col)?;

            for (index, &target_bgr) in bgr_values.iter().enumerate() {
                if (pixel[0] as i16 - target_bgr[0] as i16).abs() <= tolerance as i16 &&
                    (pixel[1] as i16 - target_bgr[1] as i16).abs() <= tolerance as i16 &&
                    (pixel[2] as i16 - target_bgr[2] as i16).abs() <= tolerance as i16 {
                    counts[index] += 1;
                    break;
                }
            }
        }
    }

    Ok(counts)
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

            let bgr_counts = count_bgr_values(&clash_rect, &BGR_VALUES, 3)?;
            if bgr_counts[0] > (0.001 * area as f32) as i32 && // pointer
                bgr_counts[1] > (0.01 * area as f32) as i32 && // click_perfect
                bgr_counts[2] > (0.03 * area as f32) as i32 && // click_good
                bgr_counts[3] > (0.4 * area as f32) as i32 {   // click_bad
                return Ok(Some((rect.x, rect.y, rect.width, rect.height)));
            }
        }
    }

    Ok(None)
}

fn infer_regions(mut sequence: Vec<usize>) -> Vec<usize> {
    let mut changed = true;

    // Keep iterating until no more changes can be made
    while changed {
        changed = false;
        let len = sequence.len();
        let mut updates: Vec<(usize, usize)> = Vec::new();

        // Handle edges first if they contain 9's
        if sequence[0] == 9 {
            // Find first non-9 value from left
            for i in 1..len {
                if sequence[i] != 9 {
                    updates.push((0, sequence[i]));
                    changed = true;
                    break;
                }
            }
        }

        if sequence[len-1] == 9 {
            // Find first non-9 value from right
            for i in (0..len-1).rev() {
                if sequence[i] != 9 {
                    updates.push((len-1, sequence[i]));
                    changed = true;
                    break;
                }
            }
        }

        // Scan through the sequence looking for 9's
        let mut i = 1;
        while i < len-1 {
            if sequence[i] == 9 {
                // Count consecutive 9's
                let mut nine_count = 1;
                let start_pos = i;
                while i + nine_count < len && sequence[i + nine_count] == 9 {
                    nine_count += 1;
                }

                // Handle single 9
                if nine_count == 1 {
                    // Look left for the nearest non-9 value
                    for j in (0..start_pos).rev() {
                        if sequence[j] != 9 {
                            updates.push((start_pos, sequence[j]));
                            changed = true;
                            break;
                        }
                    }
                    i += 1;
                    continue;
                }

                // Find left and right values for groups of 9's
                let mut left_value = None;
                let mut right_value = None;

                // Look left
                for j in (0..start_pos).rev() {
                    if sequence[j] != 9 {
                        left_value = Some(sequence[j]);
                        break;
                    }
                }

                // Look right
                for j in (start_pos + nine_count)..len {
                    if sequence[j] != 9 {
                        right_value = Some(sequence[j]);
                        break;
                    }
                }

                // Fill in values based on available sides
                match (left_value, right_value) {
                    (Some(left), Some(right)) => {
                        // For odd number of 9's, fill all but the last one
                        let fill_count = if nine_count % 2 == 1 {
                            nine_count - 1
                        } else {
                            nine_count
                        };

                        let mid_point = start_pos + fill_count / 2;

                        // Fill from left to midpoint
                        for j in start_pos..mid_point {
                            updates.push((j, left));
                            changed = true;
                        }

                        // Fill from right to midpoint (if even count)
                        if nine_count % 2 == 0 {
                            for j in mid_point..start_pos + fill_count {
                                updates.push((j, right));
                                changed = true;
                            }
                        }
                    },
                    (Some(value), None) | (None, Some(value)) => {
                        // If only one side has a value, use it for all positions
                        for j in 0..nine_count {
                            updates.push((start_pos + j, value));
                            changed = true;
                        }
                    },
                    (None, None) => {} // Can't infer any values
                }

                i += nine_count;
                continue;
            }
            i += 1;
        }

        // Apply all updates after scanning
        for (index, value) in updates {
            sequence[index] = value;
        }
    }

    sequence
}

fn  get_pixel_vec(rect: Rect) -> opencv::Result<Vec<usize>> {
    let image = screenshot()?;
    // let image = imgcodecs::imread("screenshot2.png", imgcodecs::IMREAD_COLOR)?;
    let image = image.roi(rect)?.try_clone()?;

    let mut pixels = Vec::new();
    for col in 0..image.cols() {
        let pixel = image.at_2d::<Vec3b>(0, col)?;
        let mut found = false;
        for (index, &target_bgr) in BGR_VALUES.iter().enumerate() {
            if pixel[0] == target_bgr[0] as u8 &&
                pixel[1] == target_bgr[1] as u8 &&
                pixel[2] == target_bgr[2] as u8 {
                pixels.push(index);
                found = true;
                break;
            }
        }
        if !found {
            pixels.push(9);
        }
    }

    pixels = infer_regions(pixels);

    Ok(pixels)
}

fn find_indices(bar: &[usize], value: usize) -> Option<(usize, usize)> {
    let start = bar.iter().position(|&x| x == value)?;
    let end = bar.iter().rposition(|&x| x == value)?;
    Some((start, end))
}

fn main() -> opencv::Result<()> {
    let mut enigo = Enigo::new(&Settings::default()).unwrap();
    loop {
        let image = screenshot()?;
        if image.empty() {
            std::thread::sleep(std::time::Duration::from_millis(100));
            continue;
        }

        match find_clash(&image)? {
            Some((x, y, w, h)) => {
                println!("Detected Clash");
                let rect = Rect::new(x, y + h / 2, w, 1);

                let mut count = 0;
                loop {
                    if count > 10 {
                        println!("Clash ended");
                        break;
                    }

                    let meter_vec = get_pixel_vec(rect)?;
                    if meter_vec.iter().filter(|&&x| x == 2).count() == 0 ||
                        meter_vec.iter().filter(|&&x| x == 0).count() == 0 {
                        std::thread::sleep(std::time::Duration::from_millis(10));
                        count += 1;
                        continue;
                    }
                    count = 0;

                    let (start, end) = find_indices(&meter_vec, 0).unwrap();
                    let current_pointer_pos = (start + end) / 2;

                    // Find the size of the regions
                    let ones = meter_vec.iter().filter(|&&x| x == 1).count();

                    let click_start;
                    let click_end;
                    if ones as f64 / meter_vec.len() as f64 > 0.03 {
                        let (start, end) = find_indices(&meter_vec, 1).unwrap();
                        click_start = start;
                        click_end = end;
                    } else {
                        println!("using 2s");
                        let (start, end) = find_indices(&meter_vec, 2).unwrap();
                        let length = end - start;
                        click_start = start + length / 4;
                        click_end = end - length / 4;
                    }
                    if current_pointer_pos > click_start && current_pointer_pos < click_end {
                        enigo.button(Button::Left, Direction::Click).expect("");
                        std::thread::sleep(std::time::Duration::from_millis(200));
                    }
                }
            }
            None => {}
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}