clear all;

%{
(i)	Devise and implement a “QR code reader” using a short Matlab script to 
generate the correct 29 x 29 binary array to represent the QR code 
    (0 represents a black cell; 1 represents a white cell).

%}


%Read in image
qr_1 = imread("QR_1.jpg");
one = qr_code_reader(qr_1, 29, 29);

%imshow(medRes1);
sp = figure('Name', 'Thresholded Image');
subplot(1,2,1), imshow(qr_1);
subplot(1,2,2), imshow(one);





%Function to generate a binary array to represent QR code
function qr_code = qr_code_reader(image, cells_x, cells_y)
    automatic_threshold_value = get_automatic_threshold_value(image);
    img_after_thresholding = apply_automatic_thresholding(image, automatic_threshold_value);
    binarized_img = imbinarize(img_after_thresholding);
    resized_binarized_img = imresize(binarized_img, [cells_x, cells_y]);
    qr_code = resized_binarized_img;
end




%Function that gets the automatic thresholding value
function automatic_threshold_value = get_automatic_threshold_value(image)
    greylevel_struct = total_num_greylevel_vals(image);
    cummulative_greylevel_struct = get_cummulative_greylevels(greylevel_struct);
    mean_greylevel_struct = get_mean_greylevel_struct(greylevel_struct, cummulative_greylevel_struct);
    size_of_image = size(image);
    total_pixels = size_of_image(1)* size_of_image(2);
    threshold_val = get_threshold_value(cummulative_greylevel_struct, mean_greylevel_struct, total_pixels);
    automatic_threshold_value = threshold_val;
end





%Function that applies the automatic thresholding value across the img
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

%Function that gets the thresholding value by analysing cummulative
%greylevels, mean greylevels and total pixels
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



%Get the max mean greylevel from mean_greylevel struct
function max_mean_greylevel = get_max_mean_greylevel(mean_greylevel_struct)
    max_mean_greylevel = max(mean_greylevel_struct.mean);
end


%Function to get quotient of cummulative pval
function quotient_of_cummulative_val = get_quotient_of_cummulative_val(total_pixels, cummulative_val)
    %disp(total_pixels + " - " + cummulative_val);
    total_pixels_minus_cummulative_val = total_pixels - cummulative_val;
    %disp(total_pixels_minus_cummulative_val);
    quotient_of_cummulative_val = cummulative_val/total_pixels_minus_cummulative_val;
    %disp(quotient_of_cummulative_val);

end

%Refer to function name for description
function squared_mean_greylevel_minus_max_greylevel_mean = get_squared_mean_greylevel_minus_max_greylevel_mean(max_mean_greylevel, this_mean_greylevel)
    %disp(this_mean_greylevel + " - " + max_mean_greylevel);
    squared_mean_greylevel_minus_max_greylevel_mean = (this_mean_greylevel - max_mean_greylevel)^2;
    %disp(squared_mean_greylevel_minus_max_greylevel_mean);
end