function [psnr] = psnr_t(markedIM,coverimage,row,column)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
mse=0;
for i=1:row*column
    mse=mse+(markedIM(i)-double(coverimage(i)))*(markedIM(i)-double(coverimage(i)));
end
mse=mse/(row*column);
psnr=10*log10(65535^2/mse);

end

