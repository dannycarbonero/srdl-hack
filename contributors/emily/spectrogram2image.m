function varargout = spectrogram2image(s, outfile)
% Based on Jess's spectrum to image: direct map decibel scaled spectrogram
% to an image with colormap jet. (use compute_spectrogram to get s).

% Use colormap jet
CMAP = jet(256);

% ... index the values of s and show the result
N = size(CMAP, 1);
im = ind2rgb(round(rescale(flipud(s), .5, N)), CMAP); 
if nargout == 0
    imshow(im, CMAP);
else 
    varargout = {im};
end

% ... save the result if a filename is given
if nargin > 1 && ~isempty(outfile)
    imwrite(im, outfile);
end
