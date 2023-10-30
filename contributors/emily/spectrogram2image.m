function im = spectrogram2image(s, outfile)
% Based on Jess's spectrum to image: direct map decibel scaled spectrogram
% to an image with colormap jet. (use compute_spectrogram to get s).

% Use colormap jet
CMAP = jet;

% ... index the values of s and show the result
N = size(CMAP, 1);
im = rescale(flipud(s), 1, N); 
% imshow(im, CMAP);

% ... save the result if a filename is given
if nargin > 1 && ~isempty(outfile)
    imwrite(im, CMAP, outfile);
end

