clear all;

%{
(i)	Devise and implement a “QR code reader” using a short Matlab script to 
generate the correct 29 x 29 binary array to represent the QR code 
    (0 represents a black cell; 1 represents a white cell).

%}




%Read in image
qr_1 = imread("QR_1.jpg");

qr_2 = imread('QR_2.jpg');

disp("One");
one = qr_code_reader(qr_1);
%medRes2 = medfilt2(one, [2,2]);
disp("Two");
two = treshold_img(qr_2);


qr_2_resized = imresize(qr_2, 0.5);
three = treshold_img(qr_2_resized);
three_cell_size = get_cells_size_in_img(qr_2_resized);

qr_2_resized_third= imresize(qr_2, [222,NaN]);
four = treshold_img(qr_2_resized_third);


%cell_size_one = get_cell_size_in_img(one);
%cell_size_two = get_cell_size_in_img(two);
%cell_size_x = cell_size_one(1);
%cell_size_y = cell_size_one(2);
%calculated_qr_cells = calculate_qr_cells(cell_size_x, cell_size_y, one);
%cell_size_x1 = cell_size_two(1);
%cell_size_y2 = cell_size_two(2);


%imshow(medRes1);
%SECOND METHOD FOR CALCULATING CELL SIZES FOR SMALLER IMAGES

disp("two");
cell_size = find_cell_size(two);
disp("three");
cell_size_2 = find_cell_size(three);
disp("Four");
cell_size_3 = find_cell_size(four);
%SECOND METHOD FOR CALCULATING CELL SIZES FOR SMALLER IMAGES

f = figure;
subplot(2,3,1), imshow(two), title("Original Sized QR");
subplot(2,3,2), imshow(cell_size), title("Resized QR Using New QR Cell Calculator (Accurate)");
subplot(2,3,3), imshow(three), title("1/2 Sized QR");
subplot(2,3,4), imshow(cell_size_2), title("Resized QR Using New Cell Calculator (Accurate)");
subplot(2,3,5), imshow(four), title("1/3 Sized QR");
subplot(2,3,6), imshow(cell_size_3), title("Resized QR Using New Cell Calculator (Inaccurate)");


%ORIGINAL METHOD FOR CALCULATING CELL SIZES FOR SMALLER IMAGES
five = qr_code_reader(qr_2);
six = qr_code_reader(qr_2_resized);
seven= qr_code_reader(qr_2_resized_third);
qr_2_resized_two_thirds = imresize(qr_2, [444, NaN]);
eight = qr_code_reader(qr_2_resized_two_thirds); 
qr_2_resized_three_quarters = imresize(qr_2, [222, NaN]);
nine = qr_code_reader(qr_2_resized_three_quarters);


sp = figure('Name', 'Thresholded Image');
subplot(2,5,1), imshow(qr_2), title("QR 2 (Original Size)");
subplot(2,5,2), imshow(five), title("QR 1/1 Resulting 37x37 Image");
subplot(2,5,3), imshow(qr_2_resized), title("QR 2 (Half of Original Size)");
subplot(2,5,4), imshow(six), title("QR 2 1/2 Resulting 37x37 Image");
subplot(2,5,5), imshow(qr_2_resized_third), title("QR 2 (1/4 original size)");
subplot(2,5,6), imshow(seven), title("QR 2 1/4 Resulting 37x37 Image");
subplot(2,5,7), imshow(qr_2_resized_two_thirds), title("QR 2 (2/3 original size)")
subplot(2,5,8), imshow(eight), title("QR 2 2/3 Accurate 37x37")
subplot(2,5,9), imshow(qr_2_resized_two_thirds), title("QR 2 (1/3 original size)");
subplot(2,5,10), imshow(nine), title("QR 2 1/3 Accurate 37x37");



%NEW METHOD TO CALCULATE CELL SIZES FOR SMALLER IMAGES
test = get_cells_size_in_img(two);
disp("TEST " + test);
test2 = get_cells_size_in_img(three);
disp("TEST 2 " + test2);


%Function to generate a binary array to represent QR code
function qr_code = qr_code_reader(image)
    [rows, cols] = size(image);
    automatic_threshold_value = get_automatic_threshold_value(image);
    img_after_thresholding = apply_automatic_thresholding(image, automatic_threshold_value);
    cropped_img = get_cropped_img(img_after_thresholding);
    cell_size = get_cell_size_in_img(cropped_img);
    disp("Cell Size: " + cell_size);
    cell_size_x = cell_size(1);
    cell_size_y = cell_size(2);
    dbl_cell_size_x = double(cell_size_x);
    dbl_cell_size_y = double(cell_size_y);
    dbl_rows = double(rows);
    dbl_cols = double(cols);
    width = dbl_rows/dbl_cell_size_x;
    height = dbl_cols/dbl_cell_size_y;
    cells_x = floor(width);
    cells_y = floor(height);
    disp(cells_x + " " + cells_y);
    binarized_img = imbinarize(cropped_img);
    resized_binary_img = imresize(binarized_img, [cells_x, cells_y]);

    qr_code = resized_binary_img;
end

function thresholded_img = treshold_img(image)
    [rows, cols] = size(image);
    automatic_threshold_value = get_automatic_threshold_value(image);
    img_after_thresholding = apply_automatic_thresholding(image, automatic_threshold_value);
    cropped_img = get_cropped_img(img_after_thresholding);

    thresholded_img = cropped_img;
end



%SECOND METHOD FOR CALCULATING CELL SIZES FOR SMALLER IMAGES
function cell_size_in_img = find_cell_size(img)
    [rows, cols] = size(img);

    first_white_cell_dimension_c = get_cell_size_in_col(img, rows, cols);
    first_white_cell_dimension_r = get_cell_size_in_row(img, rows, cols);
    disp(first_white_cell_dimension_c + " " + first_white_cell_dimension_r);
    double_cell_size_c = double(first_white_cell_dimension_c);
    double_cell_size_r = double(first_white_cell_dimension_r);
    double_rows = double(rows);
    double_cols = double(cols);
    width = double_cols/double_cell_size_c;
    height = double_rows/double_cell_size_r;
    cells_x = round(width);
    cells_y = round(height);
    disp(cells_x + " " + cells_y);
    binarized_img = imbinarize(img);
    if (cells_y < cells_x)
        resized_binary_img = imresize(binarized_img, [cells_y, cells_y]);
    elseif (cells_y >= cells_x)
        resized_binary_img = imresize(binarized_img, [cells_x, cells_x]);

    end
        
   cell_size_in_img = resized_binary_img;
end




function cell_size_on_row = get_cell_size_in_col(img, rows, cols)

    first_white_pixel_in_c = 0;

    for c=1: cols
        if (c ~= cols)
            if (img(1,c) == 255)
                if (first_white_pixel_in_c == 0)
                    first_white_pixel_in_c = c;
                end
            end
        end
    end


    first_black_pixel_after_white = 0;

    for c=first_white_pixel_in_c: cols
        if (img(1,c) == 0)
            if (first_black_pixel_after_white == 0)
                first_black_pixel_after_white = c;
            end
        end
    end
    disp(first_white_pixel_in_c);
    disp(first_black_pixel_after_white);

    cell_size_on_row = (first_black_pixel_after_white -1) - first_white_pixel_in_c;
end

function cell_size_on_col = get_cell_size_in_row(img, rows, cols)
    first_white_cell_in_r = 0;

    for r=1: rows
        if (img(r,1) == 255)
            if(first_white_cell_in_r == 0)
                first_white_cell_in_r = r;
            end
        end
    end

    disp(first_white_cell_in_r);
    first_black_pixel_after_white = 0;

    for r2=first_white_cell_in_r: cols
        if (first_black_pixel_after_white == 0)
            if (img(r2,1) == 0)
                first_black_pixel_after_white = r2;
            end
        end
    end
    
    cell_size_on_col = (first_black_pixel_after_white -1) - first_white_cell_in_r;
end

function blurred_img = blur_img_for_wiener_filter(img)
    PSF = fspecial('motion',13, 13);
    Idouble = im2double(img);
    blurred_img = imfilter(Idouble,PSF,'conv','circular');
end

function restored_blurred_img = restore_blur_using_wienr_filter(blurred_img)
    PSF = fspecial('motion',13, 13);
    restored_blurred_img = deconvwnr(blurred_img, PSF);
    restored_blurred_img = im2uint8(restored_blurred_img);
end

function cleaned_of_sp_noise = clean_sp_noise(sp_img, cell_size_x, cell_size_y)
    [rows, cols] = size(sp_img);
    
    %Pixels per cell
    pixels_per_cell_x = floor(double(rows)/double(cell_size_x));
    pixels_per_cell_y = floor(double(cols)/double(cell_size_y));
    disp(cell_size_x + " " + cell_size_y);
    
    img_copy = sp_img;
    wiener_2 = sp_img;
    another_img = sp_img;
    %Loop through image in steps of cell_sizes
    for r=1: pixels_per_cell_x:rows
        upper_x = r + pixels_per_cell_x;
        %if (upper_x <= rows)
            %disp("r: " + r + " upper_x: " + upper_x);
         for c=1: pixels_per_cell_y: cols
            upper_y = c + pixels_per_cell_y;
            if (upper_y <= cols) && (upper_x <= rows)
                pixel_range = sp_img(r: upper_x, c: upper_y);
                img_after_medfilt2 = medfilt2(pixel_range);
                img_after_weiner_2 = wiener2(pixel_range);
                %disp(median(img_after_medfilt2(:)));
                img_copy(r: upper_x, c: upper_y) = img_after_medfilt2;
                wiener_2(r: upper_x, c: upper_y) = img_after_weiner_2;

            else    
                rows_minus_r = rows - r;
                cols_minus_c = cols - c;
                pixel_range = sp_img(r: r + rows_minus_r, c: c+cols_minus_c);
                img_after_weiner_2 = wiener2(pixel_range);
                wiener_2(r: r + rows_minus_r, c: c + cols_minus_c) = img_after_weiner_2;
            end
                %disp(upper_y < pixels_per_cell_y);
        end
        %end
    end
    
    cleaned_of_sp_noise = wiener_2;


end


function cell_size = get_cell_size(image, cells_x, cells_y)
    [rows, cols] = size(image);
    cell_size_x = uint16(rows/cells_x);
    cell_size_y = uint16(cols/cells_y);
    disp(cell_size_x);
    disp(cell_size_y);
    cell_size = [cell_size_x, cell_size_y];
end






function automatic_threshold_value = get_automatic_threshold_value(image)
    greylevel_struct = total_num_greylevel_vals(image);
    cummulative_greylevel_struct = get_cummulative_greylevels(greylevel_struct);
    mean_greylevel_struct = get_mean_greylevel_struct(greylevel_struct, cummulative_greylevel_struct);
    size_of_image = size(image);
    total_pixels = size_of_image(1)* size_of_image(2);
    threshold_val = get_threshold_value(cummulative_greylevel_struct, mean_greylevel_struct, total_pixels);
    automatic_threshold_value = threshold_val;
end


function cropped_img = get_cropped_img(img)
    [rows, cols] = size(img);
    %disp(rows + " " +  cols);
    top_left_val = img(rows(1), cols(1));
    if (top_left_val == 255)
        top_row_crop = crop_top_row(img, rows, cols);
        %disp("Need to crop top rows at row " + top_row_crop);
        img = img(top_row_crop:rows, :);
    end

    [rows, cols] = size(img);
    bottom_left_val = img(rows, cols);
    %disp(bottom_left_val);
    if (bottom_left_val == 255)
            bottom_row_crop = crop_bottom_row(img, rows, cols);
            %disp("Need to crop bottom rows at row " + bottom_row_crop);
            img = img(1:bottom_row_crop, :);
    end


    [rows, cols] = size(img);
    top_left_val = img(rows(1), cols(1));
    if (top_left_val == 255)
        left_col_crop = crop_left_col(img, rows, cols);
        %disp("Need to crop left columns at col " + left_col_crop);
        img = img(:, left_col_crop:cols);
    end


    [rows, cols] = size(img);
    top_right_val = img(1, cols);
    if (top_right_val == 255)
        right_col_crop = crop_right_col(img, rows, cols);
        %disp(right_col_crop);
        %disp("Need to crop right columns at col " + right_col_crop);
        img = img(:, 1: right_col_crop);
    end

    cropped_img = img;

end

 
function top_row_crop = crop_top_row(img, rows, cols)
    top_row_crop = 0;

    for c=1:cols
        if (top_row_crop ~= 0)
            break
        end
        for r=1:rows
            if (top_row_crop ~= 0)
                break
            
            elseif (img(r, c) == 0)
                top_row_crop = r;
                break
            end
        end
    end
end


function bottom_row_crop = crop_bottom_row(img, rows, cols)
    bottom_row_crop = 0;

    for c=cols: -1:1
       if (bottom_row_crop ~=0)
           break
       end
       for r=rows: -1:1
            if (bottom_row_crop ~=0)
                break
            elseif (img(r, c) == 0)
                bottom_row_crop = r;
                
            end
       end
    end
end


function left_col_crop = crop_left_col(img, rows, cols)
    left_col_crop = 0;

    for r=1: rows
        if (left_col_crop ~=0)
            break
        end
        for c=1: cols
            if (left_col_crop ~=0)
                break
            
            elseif (img(r,c) == 0)
                left_col_crop = c;
            end
        end
    end
end


function right_col_crop = crop_right_col(img, rows, cols)
    right_col_crop = 0;

    for r=rows: -1:1
        if (right_col_crop ~=0)
            break
        end
        for c=cols: -1:1
            if (right_col_crop ~=0)
                break
            
            elseif (img(r,c) == 0)
                right_col_crop = c;
            end
        end
    end
end


%CELL SIZE
function cell_size = get_cell_size_in_img(img)
    cell_size = get_top_left_first_cell_coords(img);

end


%Function that will find the first white pixel in the image, and by doing
%so, finding the number of pixels that will correspond to one cell in the
%qr code 
function top_left_cell_coords = get_top_left_first_cell_coords(img)
    r = 0;
    c = 0;
    [rows, cols] = size(img);
    %Only iterate until first quarter of the image
    for i=1: uint8(rows/4)
        if (r ==0 && c ==0)
            if (img(i,i) ==255)
                r = i;
                c = i;
                %disp("First white cell found");
            end
        end
    end
    %{
    %Check if there are surrounding black cells
    if (img(r, c-1) == 255)
        %disp("Another black cell found at different position");
        if (img(r-1, c) == 255)
            %disp("Second black cell found at different position");
            r = r-1;
            c = c-1;
        end
    end
    %}
    %Minus one from both r & c as to get the last black pixel before the
    %first white pixel
    top_left_cell_coords = [r-1,c-1];

end

function cell_size = get_cells_size_in_img(img)
    first_white_pixel_in_c = get_first_white_pixel_in_c(img);
    disp(first_white_pixel_in_c);
    first_black_pixel_after_first_white_pixel = get_first_black_pixel_after_white_pixel(img, first_white_pixel_in_c);
    disp(first_black_pixel_after_first_white_pixel);
    first_white_pixels_end_col = first_black_pixel_after_first_white_pixel-1;

    [rows, cols] = size(img);
    disp(rows + " " + cols);

    cell_size = first_white_pixels_end_col - first_white_pixel_in_c;
end

function first_white_pixel_in_c = get_first_white_pixel_in_c(img)
    [rows, cols] = size(img);

    first_white_pixel_in_c = 0;

    for c=1: cols
        if (img(1,c) == 255)
            if (first_white_pixel_in_c == 0)
                first_white_pixel_in_c = c;
                
            end
        end
    end
end


function first_black_pixel_after_white = get_first_black_pixel_after_white_pixel(img, white_pixel)
    [rows, cols] = size(img);
    
    first_black_pixel_after_white = 0;

    for c=white_pixel: cols
        if (img(1,c) == 0)
            if (first_black_pixel_after_white == 0)
            first_black_pixel_after_white = c;
            end
        end
    end
    
    first_black_pixel_after_white = first_black_pixel_after_white;
end

function image_after_thresholding = apply_automatic_thresholding(image, automatic_threshold_value)
    [rows, cols] = size(image);
    %disp(rows + " " + cols);

    for r=1:rows
        for c=1: cols
            if (image(r, c) <= automatic_threshold_value)%-10)
                image(r,c) = 0;

            else
                image(r,c) = 255;
            end
            
        end
    end

    image_after_thresholding = image;

end

%Function that returns a struct holding the greylevel values that make up
%our image as well as the total number of pixels that correspond to said greylevel
function greylevel_struct = total_num_greylevel_vals(image)
    %Find total number of pixels at greylevel g
    greylevel_struct.greylevel = [];
    greylevel_struct.count = [];

    %Populate graylevel.value with all possible graylevel vals
    for i=0:255
        greylevel_struct.greylevel(i+1) = i;
    end 


    %Populate graylevel.count with 256 default values of 0
    for i=0: 255
        greylevel_struct.count(i+1) = 0;
    end
   

    %Get size of image
    [row, col] = size(image);

    
    for r=1:row
        for n=0:255
            %{
            Count the number of times the graylevel n occurs in
            the current row (r) across all columns
            %}
            n_occurs = sum(image(r, :)==n);

            %{
            If n_occurs isn't equal to 0, increment the current pixel's 
            count level by the number of times n occurs in the current row
            %}
            if (n_occurs ~= 0)
                greylevel_struct.count(n+1) =  greylevel_struct.count(n+1) + n_occurs;
            end
        end
    end 
    
   %greylevel_struct_zeros_removed is a struct which will hold all
   %greylevel values and corresponding total number of pixels ONLY for
   %greylevels with a total number of pixels > 0
   greylevel_struct_zeros_removed = get_greylevel_struct_without_zeros(greylevel_struct);

   %Re-Initializing greylevel_struct to greylevel_struct_zeros_removed so
   %we don't calculate greylevel values where the corresponding number of
   %pixels is equal to 0
   greylevel_struct = greylevel_struct_zeros_removed;
end



%Function that will be used to pop all greylevel values as well as the
%total count of pixels at that greylevel in the image if the total count is
%0
function greylevel_struct_zeros_removed = get_greylevel_struct_without_zeros(greylevel_struct)
    %Initializing struct with two arrays
    %greylevel array will hold greylevel values only if their pixel count
    %is > 0
    greylevel_struct_zeros_removed.greylevel = [];
    %count array will hold the total count of pixels at the greylevel if
    %the count is > 0
    greylevel_struct_zeros_removed.count = [];

    %Getting size of our greylevel struct 
    %Will be used as an upper bound in the for loop
    size_of_greylevel_struct = size(greylevel_struct.count);

    %n will mark the index at which we want to append new elements to the
    %greylevel and count arrays in the
    %greylevel_struct_zeros_removed_struct
    n = 1;
    
    %Loop through the entire greylevel_struct
    for i = 1: size_of_greylevel_struct(2)
        %If the pixel count is not 0, enter
        if (greylevel_struct.count(i) ~= 0)
            greylevel_struct_zeros_removed.greylevel(n) = greylevel_struct.greylevel(i); 
            greylevel_struct_zeros_removed.count(n) = greylevel_struct.count(i);
            n = n + 1;
        end
    end
end




%Function that will calculate the cummulative total for all greylevels in
%greylevel_struct
function cummulative_greylevels = get_cummulative_greylevels(greylevel_struct)
    %Array to hold greylevel values
    cummulative_greylevels.greylevel = [];
    
    %{
    Array to hold the cummulative count in correspondance with the
    greylevel at the same index value in cummulative_greylevels.greylevel
    %}
    cummulative_greylevels.count = [];
    
    %Integer to hold the cummulative count
    cummulative_counter = 0;

    %
    size_of_greylevel_struct = size(greylevel_struct.greylevel);
    for n=1:size_of_greylevel_struct(2)
        %Integer to hold the array index we want to append our values to
        current_arr_index = size(cummulative_greylevels.greylevel)+1;
        current_arr_index = current_arr_index(2);

        %
        if (greylevel_struct.count(1, n) ~=0)
            %Increment cummulative_counter by the current greylevel's
            %value
            cummulative_counter = cummulative_counter + greylevel_struct.count(n);
            
            %Add greylevel to appropriate index in
            %cummulative_greylevels.greylevel
            cummulative_greylevels.greylevel(current_arr_index) = greylevel_struct.greylevel(n);
            
            %Append cummulative value of current greylevel to appropriate 
            %index in cummulative_greylevels.count 
            cummulative_greylevels.count(current_arr_index) = cummulative_counter;
            
            %disp(greylevel_struct.greylevel(n) + ": " + greylevel_struct.count(n) + ": " + cummulative_counter);
        end
    end
end


%Function that wil calculate the mean greylevel for each greylevel present
%in the image
function mean_greylevel_struct = get_mean_greylevel_struct(greylevel_struct, cummulative_greylevel_struct)
    %Initializing struct of two arrays which will hold:
    %1. each greylevel value present in the image
    mean_greylevel_struct.greylevel = greylevel_struct.greylevel;
    %2. The mean greylevel calculated
    mean_greylevel_struct.mean = [];

    %Finding size of the mean_greylevel_struct.greylevel array - which will
    %provide the number of times our loop should perform the calculation of
    %finding mean greylevels
    size_of_mean_greylevel_struct = size(mean_greylevel_struct.greylevel);
    size_of_mean_greylevel_struct = size_of_mean_greylevel_struct(2);
  
    for i=1: size_of_mean_greylevel_struct        
        %Variable to hold sum of all greylevel values up to index i in
        %size_of_mean_greylevel_struct - will re-initialize each time loop
        %runs
        sum = 0;

        %Loop to calculate sum of all pixels in range 0 to i
        for n=0 : i-1
            %Increment sum variable by index n multiplied by the number of pixels
            %At the appropriate greylevel in greylevel_struct (needs to be
            %n+1 because matlab array indexes start at 1, and n starts at
            %0)
            sum = sum + (n * greylevel_struct.count(n+1));
            
             %Uncomment to be assured we are working with appropriate
             %elements
             %{
             if (i <=10)%|| i >=130)
                disp("Greylevel: " + mean_greylevel_struct.greylevel(i) + " Index of sum: " + n +  ...
                    " Num of pixels at Greylevel: "  +  greylevel_struct.count(n+1) + " Incrementing Sum Value: " + sum);
            end 
             %}

        end
        %Add mean greylevel value to appropriate index in
        %mean_greylevel_struct.mean array
        mean_greylevel_struct.mean(i) = sum / cummulative_greylevel_struct.count(i);
        
        
        %Uncomment to be assured/demonstrate that we are returned the appropriate mean
        %greylevel value
        %{
        if (i <= 10)% || i >= 130)
            disp("greylevel : " + mean_greylevel_struct.greylevel(i) + " sum: " + sum +  " cummulative_grey_level: " + cummulative_greylevel_struct.greylevel(i)  +" cummulative_grey val: " + cummulative_greylevel_struct.count(i) + " mean_grey: " + mean_greylevel_struct.mean(i));
        end
        %}
    end
end

%NEEDS WORK
function threshold_value = get_threshold_value(cummulative_greylevel_struct, mean_greylevel_struct, total_pixels)
    max_mean_greylevel = get_max_mean_greylevel(mean_greylevel_struct);
    
    size_of_mean_struct = size(mean_greylevel_struct.mean);
    %Declaring & Initializing variable that will hold the number of times
    %we should loop
    upper_bound = size_of_mean_struct(2);

    results_arr = [];
    for i=1:upper_bound
        squared_mean_greylevel_minus_max_greylevel_mean = get_squared_mean_greylevel_minus_max_greylevel_mean(max_mean_greylevel, mean_greylevel_struct.mean(i));
        quotient_of_cummulative_val = get_quotient_of_cummulative_val(total_pixels, cummulative_greylevel_struct.count(i));
        %product = int64(squared_mean_greylevel_minus_max_greylevel_mean)*int64(quotient_of_cummulative_val);
        product = squared_mean_greylevel_minus_max_greylevel_mean*quotient_of_cummulative_val;
        %disp(product);
        results_arr(i) = product;
    end
    %disp("Cummulative " + size(cummulative_greylevel_struct.count));
    %disp("Mean_level " + size(mean_greylevel_struct.mean));
    %disp("Results_arr " + size(results_arr));

    %disp(max(results_arr));

    max_value_in_results_arr = max(results_arr);
    max_val_int = max_value_in_results_arr;
    max_val_int_minus_one = (max_val_int-1);
    threshold_value = find(results_arr==(max_value_in_results_arr))-1;%-1;
end



function max_mean_greylevel = get_max_mean_greylevel(mean_greylevel_struct)
    max_mean_greylevel = max(mean_greylevel_struct.mean);
end


function quotient_of_cummulative_val = get_quotient_of_cummulative_val(total_pixels, cummulative_val)
    %disp(total_pixels + " - " + cummulative_val);
    total_pixels_minus_cummulative_val = total_pixels - cummulative_val;
    %disp(total_pixels_minus_cummulative_val);
    quotient_of_cummulative_val = cummulative_val/total_pixels_minus_cummulative_val;
    %disp(quotient_of_cummulative_val);

end

function squared_mean_greylevel_minus_max_greylevel_mean = get_squared_mean_greylevel_minus_max_greylevel_mean(max_mean_greylevel, this_mean_greylevel)
    %disp(this_mean_greylevel + " - " + max_mean_greylevel);
    squared_mean_greylevel_minus_max_greylevel_mean = (this_mean_greylevel - max_mean_greylevel)^2;
    %disp(squared_mean_greylevel_minus_max_greylevel_mean);
end