function hyperspectral_processing_modified()
    %% Main function: generate 15 types of degraded hyperspectral images
    % Path settings
    folder_HR = 'E:\Xiongan\test\clean\'; % Path to clean images
    base_save_path = 'E:\Xiongan\test\degraded\'; % Root path for saving degraded images

    % Define the degradation types and subtypes
    degradations = {'haze', 'blur', 'noise', 'bandMissing'};
    noise_subtypes = {'gauss', 'impulse', 'stripe'};
    bandmissing_subtypes = {'complete', 'bandwise', 'partial'};

    % Generate the 15 degradation combinations (binary 1 to 15)
    for combo = 1:15
        % Determine which degradation types are applied
        haze_flag = bitget(combo, 1);        % bit 1: haze
        blur_flag = bitget(combo, 2);        % bit 2: blur
        noise_flag = bitget(combo, 3);       % bit 3: noise
        bandMissing_flag = bitget(combo, 4); % bit 4: bandMissing

        % Build the folder name
        folder_name = '';
        if haze_flag, folder_name = [folder_name 'h_']; end
        if blur_flag, folder_name = [folder_name 'b_']; end
        if noise_flag, folder_name = [folder_name 'n_']; end
        if bandMissing_flag, folder_name = [folder_name 'bm_']; end
        folder_name = folder_name(1:end-1); % Remove the trailing underscore
        save_folder = fullfile(base_save_path, folder_name);
        if ~exist(save_folder, 'dir'), mkdir(save_folder); end

        % Process the images
        for i = 1:10
            % Read the clean image
            filepath = fullfile(folder_HR, [' ',int2str(i), '.tif']);
            hyperspectral_image = read_hyperspectral_image(filepath);
            

            % Initialize the image to be processed
            process_image = hyperspectral_image;

            % Randomly select the noise and bandMissing subtypes
            if noise_flag
                noise_type = noise_subtypes{randi(3)};
            else
                noise_type = 'clean';
            end
            if bandMissing_flag
                bandmissing_type = bandmissing_subtypes{randi(3)};
            else
                bandmissing_type = 'clean';
            end

            % Apply the degradations in order
            if haze_flag
                process_image = add_haze(process_image);
                haze_status = 'haze';
            else
                haze_status = 'clean';
            end
            if blur_flag
                process_image = blur_hyperspectral_image(process_image, 5);
                blur_status = 'blur';
            else
                blur_status = 'clean';
            end

            if noise_flag
                switch noise_type
                    case 'gauss'
                        process_image = add_gaussian_noise(process_image, 0, randi([5,25],1));
                    case 'impulse'
                        process_image = add_impulse_noise(process_image);
                    case 'stripe'
                        process_image = add_stripe_noise(process_image, 0.2, 0.1, 0.2, 0.1, 25);
                end
            end
            if bandMissing_flag
                switch bandmissing_type
                    case 'complete'
                        process_image = add_complete_bandmissing(process_image);
                    case 'bandwise'
                        process_image = add_bandwise_bandmissing(process_image);
                    case 'partial'
                        process_image = add_deadline_noise(process_image, 0.1, 0.05, 0.1);
                end
            end

            % Build the file names
            file_name = sprintf('%d_%s_%s_%s_%s.tif', i, haze_status, blur_status, noise_type, bandmissing_type);
            filepng_name = sprintf('%d_%s_%s_%s_%s.png', i, haze_status, blur_status, noise_type, bandmissing_type);
            save_path = fullfile(save_folder, file_name);
            save_rgb_path = fullfile(strcat(save_folder,'/rgb/'), filepng_name);
            if ~exist(strcat(save_folder,'/rgb/')), mkdir(strcat(save_folder,'/rgb/')); end

            process_rgb_image = extract_and_normalize_rgb(process_image, 120,76,36, 1.5);
            % gf5 59, 38, 20  aviris 37, 19, 8 enmap 48,30,16 xiongan 120
            % 76 36
            imwrite(process_rgb_image,save_rgb_path);
            process_image = uint16(process_image * (2^14 - 1));
            save_hyperspectral_image(save_path, process_image, 16);
        end
    end
end

%% Helper functions
function image = read_hyperspectral_image(filepath)
    % Read the hyperspectral image
    image = imread(filepath);
    image = double(image);
    
    % Check the maximum value of the original image
    max_original = max(image(:));
    fprintf('原始图像最大值: %f\n', max_original);
    
    % Normalize to [0, 1] to prevent overflow
    image = image / (2^14 - 1);
    
    % Keep the values strictly within [0, 1]
    
    % Report the maximum value after normalization
    max_normalized = max(image(:));
    fprintf('归一化后图像最大值: %f\n', max_normalized);
end

function hazy_img = add_haze(img)
%% Impulse
fprintf("Hazy img ")
% Assume img is a hyperspectral image (H x W x B, where B is the number of bands)
[H, W, B] = size(img); % Get the image height, width, and number of bands

% Create an airlight matrix the same size as img
airlight = zeros(size(img), 'like', img);

% Fill the airlight matrix
for b = 1:B
    anum = 0.9 + 0.2 * rand(); % Generate a random number between 0.85 and 0.95
    dif = -0.01 + (0.01 - (-0.01)) * rand(); % Generate a random number between -0.03 and 0.03
    airlight(:, :, b) = min(1, (anum + dif)*max(max(img(:,:,b)))); % All pixels of a band share the same value (anum + dif);
end

% Create an mbtrans_img matrix the same size as img
mbtrans_img = zeros(size(img), 'like', img);

% Define the band wavelength distribution
Bandwavelength = linspace(400,1000,256);
% Read the image and process it
img_path = fullfile('E:\code_history\RSMD-IR\data\Landsat\train\singletranspng\landsat8trans', sprintf('%d.png', randi([1, 1000])));
trans_img = imread(img_path);
trans_img = trans_img(129:384,129:384);%(209:320,209:320);%(193:320,193:320);%(129:384,129:384);%(1:512,1:512);
trans_img = double(trans_img) / 255; % Normalize the image data to [0, 1]

% Assume trans_img is a single-band image (H x W)
trans_img = repmat(trans_img, [1, 1, B]); % Replicate the single-band image across all bands

w = 0.8; % Weight parameter
trans_img = 1 - w * trans_img;

% Process trans_img and fill mbtrans_img
lambdanum = 1 * rand(); % Generate a random number between 0 and 4
count=0;
for b = 1:B
    fprintf(1, repmat('\b',1,count));
    count=fprintf(1,'%d: %d', B, b);
    if b == 120 % Red-light band; gf5: 59, aviris: 37, enmap: 48, xiongan: 120
        mbtrans_img(:, :, b) = trans_img(:, :, 1);
        %fprintf('trans_img max: %f, min: %f\n', max(max(trans_img(:, :, 1))), min(min(trans_img(:, :, 1))));
    else % Other bands
        bwnum = (Bandwavelength(b) / Bandwavelength(1))^lambdanum;
        mbtrans_img(:, :, b) = trans_img(:, :, 1) .^ (bwnum);
    end
end
% Clamp mbtrans_img to the range [0.03, 1]
mbtrans_img = max(min(mbtrans_img, 1), 0.03);

% Compute the final hazy image
hazy_img = img .* mbtrans_img + (1 - mbtrans_img) .* airlight;

% Print the max/min values of mbtrans_img and the hazy image
fprintf('mbtrans_img max: %f, min: %f\n', max(mbtrans_img(:)), min(mbtrans_img(:)));
fprintf('airlight_img max: %f, min: %f\n', max(airlight(:)), min(airlight(:)));
fprintf('img max: %f, min: %f\n', max(hazy_img(:)), min(hazy_img(:)));
end

function [sig_x, sig_y, theta] = generate_random_gaussian_params()
    %% Generate random Gaussian kernel parameters
    sig_x = round(rand * 3.8 + 0.2, 2);
    sig_y = round(rand * 3.8 + 0.2, 2);
    theta = round(rand * pi, 2);
end

function kernel = generate_bivariate_gaussian(kernel_size, sig_x, sig_y, theta)
    %% Generate a bivariate Gaussian kernel
    [grid_x, grid_y] = meshgrid(-floor(kernel_size / 2):floor(kernel_size / 2));
    grid = cat(3, grid_x, grid_y);

    rotation_matrix = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    sigma_matrix = diag([sig_x^2, sig_y^2]);
    covariance_matrix = rotation_matrix * sigma_matrix * rotation_matrix';

    inv_cov = inv(covariance_matrix);
    det_cov = det(covariance_matrix);

    kernel = zeros(kernel_size, kernel_size);
    for i = 1:kernel_size
        for j = 1:kernel_size
            x = squeeze(grid(i, j, :));
            kernel(i, j) = exp(-0.5 * x' * inv_cov * x);
        end
    end

    kernel = kernel / sum(kernel(:));
end

function blurred_image = blur_hyperspectral_image(image, kernel_size)
    %% Generate a random blur kernel and apply it to the hyperspectral image
    [sig_x, sig_y, theta] = generate_random_gaussian_params();
    kernel = generate_bivariate_gaussian(kernel_size, sig_x, sig_y, theta);
    
    [height, width, bands] = size(image);
    blurred_image = zeros(size(image));

    for band = 1:bands
        blurred_image(:, :, band) = conv2(image(:, :, band), kernel, 'same');
    end

    blurred_image = min(max(blurred_image, 0), 1);
end


function noisy_image = add_gaussian_noise(im_input, mean_val, sigma)
    %% Add Gaussian noise
    %% Gauss
    [W, H, Band] = size(im_input);
    maxi = max(max(max(im_input)));
    fprintf("Gauss ing ")
    count = 0;
    noiseSigma = sigma*rand(1,Band)*maxi/255.;
    for i=1:Band
        noisy_image(:, :, i) = im_input(:, :, i) + noiseSigma(i)*randn(size(im_input(:, :, i)));
        fprintf(1, repmat('\b',1,count));
        count=fprintf(1,'%d: %d', Band, i);
    end
end

function noisy_image = add_impulse_noise(im_input)
    %% Impulse        
    fprintf("Impulse ing ")
    noisy_image=im_input;
    [W, H, Band] = size(im_input);
    p_density = 0.2;
    band_i = ceil(Band*rand(1,ceil(Band*p_density)));
    count = 0;
    ratios = [0.05,0.025];
    idx = randi(length(ratios), length(band_i), 1);
    ratio = ratios(idx);
    for i=1:length(band_i)
        %ratio(i)
        noisy_image(:,:,band_i(i)) = imnoise(im_input(:,:,band_i(i)),'salt & pepper',ratio(i));
        %impluse = noisy_image(:,:,band_i(i))-im_input(:,:,band_i(i))
        fprintf(1, repmat('\b',1,count));
        count=fprintf(1,'%d: %d', length(band_i), i);
    end
    fprintf('\n')
end

function im_s_noise = add_stripe_noise(im_label, s_density, s_static, s_var, w_density, w_width_max)
% Get the dimensions of the input image
% Parameter description:
% im_label: input image of size W x H x Band.
% s_density: density of bands selected for stripe noise. 0.1
% s_static: ratio controlling the static stripe thickness. 0.1
% s_var: variation in stripe thickness. 0.5
% w_density: density of wide stripes. 0.1
% w_width_max: maximum width of wide stripes. 25
    [W, H, Band] = size(im_label);
    im_input = im_label;  % Assign the original image data to im_input

    %% Add noise
    % Randomly select the noisy bands; their number is determined by the density s_density
    band_s = ceil(Band * rand(1, ceil(Band * s_density)));  % Noisy bands randomly chosen according to density s_density
    band_w = ceil(Band * rand(1, ceil(Band * w_density)));  % Wide-stripe bands randomly chosen according to density w_density

    %% Apply the stripe noise
    fprintf("正在添加条纹噪声... ");
    count = 0;
    
    % Number of stripes per band
    % stripnum_thic: number of stripes per band, between 5% and 65%
    stripnum_thic = ceil(H * s_static) + ceil(H * s_var * rand(1, length(band_s)));
    
    % Add stripe noise to each noisy band
    for i = 1:length(band_s)
        loc_thic = ceil((H - 2) * rand(1, stripnum_thic(i)));  % Randomly select the stripe locations
        % t_thic: stripe intensity, based on the band mean with random +/- noise
        t_thic = mean(im_input(:,:,band_s(i)), [1, 2]) * randsrc(1, 1, [-1, 1]);
        % Add the stripe noise at the corresponding positions
        im_input(:, loc_thic, band_s(i)) = im_input(:, loc_thic, band_s(i)) + t_thic;
        
        % Print progress info
        fprintf(1, repmat('\b', 1, count));
        count = fprintf(1, '%d: %d', length(band_s), i);
    end
    fprintf('\n');

    % Apply the wide-stripe noise
    count = 0;
    % Number of wide stripes per band, between 1 and 3
    stripnum_wid = ceil(1 + 2 * rand(1, length(band_w)));
    
    % Add wide-stripe noise to each band
    for i = 1:length(band_w)
        loc_wid = sort(ceil((H - w_width_max) * rand(1, stripnum_wid(i))));  % Randomly select the stripe locations
        w_width = ceil(w_width_max * rand(1, stripnum_wid(i)));  % Randomly select the stripe widths
        t = w_width + loc_wid;  % Compute the stripe end positions
        
        % Handle overlapping stripes: ignore a stripe that overlaps its predecessor
        for d = 1:length(loc_wid) - 1
            if t(d) >= loc_wid(d + 1)
                w_width(d + 1) = 0; loc_wid(d + 1) = 0; t(d + 1) = 0;
            end
        end
        % Remove invalid stripes
        w_width(w_width == 0) = [];
        loc_wid(loc_wid == 0) = [];
        t(t == 0) = [];
        
        % t_wid: wide-stripe intensity, based on the band mean with random +/- noise
        t_wid = mean(im_input(:,:,band_w(i)), [1, 2]) * randsrc(1, 1, [-1, 1]);
        
        % Add the wide-stripe noise at the corresponding positions
        for j = 1:length(loc_wid)
            im_input(:, loc_wid(j):t(j), band_w(i)) = im_input(:, loc_wid(j):t(j), band_w(i)) + t_wid;
        end
        
        % Print progress info
        fprintf(1, repmat('\b', 1, count));
        count = fprintf(1, '%d: %d', length(band_w), i);
    end
    fprintf('\n');

    % Output the noisy image
    im_s_noise = im_input;
end

function noisy_image = add_deadline_noise(im_input,d_density,d_static,d_var)
    %% Deadline
    fprintf("Deadline ing ")
    [W, H, Band] = size(im_input);
    noisy_image=im_input;
    maxi = max(max(max(im_input)));
    count = 0;
    band_d = ceil(Band*rand(1,ceil(Band*d_density))); % d_static = 0.05; d_var = 0.1;
    dead_num = ceil(H*d_static)+ceil(H*d_var*rand(1,length(band_d)));     % 5-15% number of deadline in these bands   
    for i=1:length(band_d)
        loc_d = ceil((H-2)*rand(1,dead_num(i)));
        noisy_image(:,loc_d,band_d(i)) = 0;
        fprintf(1, repmat('\b',1,count));
        count=fprintf(1,'%d: %d', length(band_d), i);
    end
    fprintf('\n')
end

function image = add_complete_bandmissing(image)
    % Complete band missing: randomly choose 10% of the bands and set them to zero
    [~, ~, bands] = size(image);
    num_missing = round(0.1 * bands);
    missing_bands = randperm(bands, num_missing);
    image(:, :, missing_bands) = 0;
    image = max(min(image, 1), 0); % Clamp to [0, 1]
end

function image = add_bandwise_bandmissing(image, K, missing_row_ratio)
    % ADD_BANDWISE_BANDMISSING drops rows from selected bands of the hyperspectral image
    % Inputs:
    % image - 3D array (height x width x bands)
    % K - number of bands to which missing rows are applied (optional; default 10% of the bands)
    % missing_row_ratio - ratio of rows to set to zero (optional; default 0.1)
    % Outputs:
    % image - the image after band-wise band missing is applied

    [height, ~, bands] = size(image);

    % Set default values if not provided
    if nargin < 2
        K = round(0.1 * bands);
    end
    if nargin < 3
        missing_row_ratio = 0.1;
    end

    % Randomly select K bands
    selected_bands = randperm(bands, K);

    % Determine number of missing rows
    missing_rows = round(height * missing_row_ratio);

    % Randomly select rows to set to zero
    missing_row_indices = randperm(height, missing_rows);

    % Set selected rows in selected bands to zero
    for band = selected_bands
        image(missing_row_indices, :, band) = 0;
    end

    % Ensure image values are within [0,1]
    image = max(min(image, 1), 0);
end

function save_hyperspectral_image(savepath, im, type)
% type indicates the variable type: uint8:16; uint16:32; double:64; single:0
disp('Storing Tiff.......');
t = Tiff(savepath,'w');
% Image size information (these two fields are straightforward)
tagstruct.ImageLength=size(im,1); % Image length (rows)
tagstruct.ImageWidth=size(im,2);  % Image width (columns)
% Photometric interpretation; see Section 3.1 below for details
tagstruct.Photometric = Tiff.Photometric.MinIsWhite;
% Number of bits per pixel; single is single-precision floating point (32-bit)
% Specifies how the data type is interpreted
switch type
    case 8
        tagstruct.BitsPerSample =8;
        tagstruct.SampleFormat =Tiff.SampleFormat.UInt;
    case 16
        tagstruct.BitsPerSample =16;
        tagstruct.SampleFormat = Tiff.SampleFormat.UInt;
    case 32
        tagstruct.BitsPerSample =32;
        tagstruct.SampleFormat =Tiff.SampleFormat.UInt;
%     case 32
%         tagstruct.BitsPerSample = 32;
%         tagstruct.SampleFormat  = Tiff.SampleFormat.IEEEFP;
    case 0
        tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
        tagstruct.BitsPerSample = 32;
    case 64
        tagstruct.BitsPerSample =64;
        tagstruct.SampleFormat =Tiff.SampleFormat.IEEEFP;
end
% Bands per pixel; usually 1 or 3 for ordinary images, but often > 3 for remote sensing imagery
tagstruct.SamplesPerPixel =size(im,3);
tagstruct.RowsPerStrip = 1;
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
% Software that created the image
tagstruct.Software = 'MATLAB';
% Set the tags of the Tiff object
t.setTag(tagstruct);
% The header is ready; start writing the data
t.write(im);
% Close the Tiff file
t.close;
end

function rgb_image = extract_and_normalize_rgb(image, r_band, g_band, b_band, brightness_factor)
    %% Extract and enhance the RGB bands
    rgb_image = image(:, :, [r_band, g_band, b_band]);
    rgb_image = rgb_image * brightness_factor;
    rgb_image = rgb_image / max(rgb_image(:)); % Normalize to [0, 1]
end
