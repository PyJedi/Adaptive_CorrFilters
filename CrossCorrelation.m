A = imread('normal.jpg'); % input frame 1
B = imread('occlusion.jpg'); % input frame 2
C = imread('outofplanerotation.jpg'); % input frame 3
T = imread('template.jpg'); % template

% Converting images into greyscale
a = im2bw(A);
b = im2bw(B);
c = im2bw(C);
t = im2bw(T);

a1 = double(a);
b1 = double(b);
c1 = double(c);
t1 = double(t);

% cross correlating frames and the template
x1 = xcorr2(a1,t1);
x2 = xcorr2(b1,t1);
x3 = xcorr2(c1,t1);

% plot of x1
[ssr,snd] = max(x1(:));
[ij,ji] = ind2sub(size(x1),snd);
figure
plot(x1(:))
title('Cross-Correlation')
hold on
hold off

% plot of x2
[ssr,snd] = max(x2(:));
[ij,ji] = ind2sub(size(x2),snd);
figure
plot(x2(:))
title('Cross-Correlation')
hold on
hold off

% plot of x3
[ssr,snd] = max(x3(:));
[ij,ji] = ind2sub(size(x3),snd);
figure
plot(x3(:))
title('Cross-Correlation')
hold on
hold off



