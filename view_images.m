function view_images(im, i, len, wid)
im_view = reshape(im(:,i), [len wid]);
im_view = im_view / max(max(im_view));
imshow(im_view);