clc;clear;close all;

dataimage1={'peppers.tif','fishingboat.tif','st1x20.tif','z1x25.tif'};
    
%% coverimage
coverimage=imread('z1x25.tif');
coverimage=double(imresize(coverimage,[512 512]));  
%rgb2gray
figure;
subplot(231),imagesc(coverimage); colormap gray;
axis off; axis square;
%title('载体图');
imwrite(uint8(coverimage),'z1x25PREY.tif','Compression','none')
%% haar
% [LLorig,LHorig,HLorig,HHorig] = haart2(coverimage,3);
% subplot(232),imagesc(LLorig); colormap gray;
% title('Level 3 Haar Approximation');
% axis off; axis square;
% [row, column]=size(LLorig);

%% lifting-based haar 
n=2;
[~,m]=size(coverimage);
imgwave=liftwavedec2(coverimage,m,n);
% subplot(232),imagesc(imgwave); colormap gray;
% title('B ');
% axis off; axis square;
copy1=imgwave;
LL=imgwave(1:m/(n^2),1:m/(n^2));
[row, column]=size(LL);

%% 水印
watermark = imread('WATERMARK13.png');
watermark = imresize(watermark,[row/4 column/2]); %watermark = imresize(watermark,[row/2 column/2]);
watermark = imbinarize(watermark);
subplot(233),imagesc(watermark); colormap gray;
title('C1');axis off; axis square;
%imwrite(watermark,'watermarkRW.tif','Compression','none')
% figure;
% imagesc(watermark); colormap gray;
% axis off;
%% 水印加k
k=randi([0 1],row/4,column/2);  
watermarkXOR=xor(k,watermark);

%% 卷积码&&Turbo码
trellis = poly2trellis(4,[17 13]); %1111   1011   k/n=1/2
watermarkRESHAPE=reshape(watermarkXOR,row*column/8,1);
intrlvrIndices = randperm(row*column/8);
turboEN = comm.TurboEncoder('TrellisStructure',trellis,'InterleaverIndices',intrlvrIndices);
codedData= step(turboEN,watermarkRESHAPE);
[lengthOFturbo,~]=size(codedData);
codedData = convenc(watermarkRESHAPE,trellis); %卷积码

%% tian 嵌入
markedLL=zeros(column);
j=1;
for i=1:2:row*column
    if j<=lengthOFturbo
        x=LL(i);y=LL(i+1);
        L=floor((x+y)/2); %矩阵向下数
        h=2*(x-y)+double(codedData(j));  
        x1=L+floor((h+1)/2);
        y1=L-floor(h/2);
        markedLL(i)=x1;markedLL(i+1)=y1;
        j=j+1;
    else
        markedLL(i)=LL(i);markedLL(i+1)=LL(i+1);
    end
end
copy1(1:m/(n^2),1:m/(n^2))=markedLL;
markedIM=liftwaverec2(copy1,m,n);
figure;
imagesc(markedIM); colormap gray;
%title('D1');
axis off; axis square;

imwrite(uint8(markedIM),'z1x25WCONVI.bmp');%,'Compression','none')

%% 压缩
Cdata=reshape(markedIM,1,[]);
tbl = tabulate(Cdata);
[dict,avglen] = huffmandict(tbl(:,1),tbl(:,3)*0.01);
enco = huffmanenco(Cdata,dict);

% meth = 'gbl_mmc_h'; % Method name 
% option = 'c';         % 'c' stands for compression 
% [CR,BPP] = wcompress(option,markedIM,'tryCOM.tif',meth,'bpp',4);



%% 解压缩
dsig = huffmandeco(enco,dict);
isequal(Cdata,dsig)
markedIMC=reshape(dsig,[512,512]);


%% 逆水印
markedim=double(imread('fishingboatWCONVI.bmp'));
imgwave2 = liftwavedec2(double(markedim),m,n);
LL1=imgwave2(1:m/(n^2),1:m/(n^2));
riverLL=zeros(column); 
j=1;
for i=1:2:row*column
    if j<=lengthOFturbo
        x=LL1(i);y=LL1(i+1);  
        L=floor((x+y)/2); %矩阵向下数
        h1=floor((x-y)/2);
        watermarkREVER(j)=2*h1-(x-y);
        x1=L+floor((h1+1)/2);
        y1=L-floor(h1/2);
        riverLL(i)=x1;riverLL(i+1)=y1;
        j=j+1;
    else
        riverLL(i)=LL1(i);riverLL(i+1)=LL1(i+1);
    end
end
imgwave2(1:m/(n^2),1:m/(n^2))=riverLL;
isequal(riverLL,LL)
watermarkREVER=reshape(watermarkREVER,lengthOFturbo,1);
watermarkREVER=logical(abs(watermarkREVER));
isequal(watermarkREVER,codedData)

%% 卷积码&&Turbo解码
% tbdepth = 20; % A rate 1/2 code has a traceback depth of 5(ConstraintLength – 1).
% decodedData1 = vitdec(watermarkREVER,trellis,tbdepth,'trunc','hard');
% decodedData1=reshape(decodedData1,[row/2,column/2]);
% isequal(decodedData1,watermarkXOR)

turboDE = comm.TurboDecoder('TrellisStructure',trellis,'InterleaverIndices',intrlvrIndices, ...
    'NumIterations',4);
release(turboDE);
decodedData1 = step(turboDE,double(watermarkREVER));
% decodedData=reshape(decodedData,[row/2,column/2]);
isequal(logical(decodedData1),watermarkRESHAPE)
decodedData1=reshape(decodedData1,[row/4,column/2]);
watermarkFINAL=xor(decodedData1,k);
isequal(watermarkFINAL,watermark)
subplot(235),imagesc(watermarkFINAL); colormap gray;title('watermarkFINAL');
axis off; axis square;

%imwrite(watermarkFINAL,'z1x25RW.tif','Compression','none')

% imgwave(1:256,1:256)=LL;
% isequal(imgwave2,imgwave)
reverIM = liftwaverec2(imgwave2,m,n);
isequal(reverIM,coverimage)
subplot(236),imagesc(reverIM); colormap gray;title('REVER IM');
axis off; axis square;

%% psnr
mse=0;
for i=1:m*m
    cha=(markedim(i)-coverimage(i))*(markedim(i)-coverimage(i));
    mse=mse+cha;
end
mse=mse/(m*m);
psnr=10*log10(255^2/mse);
nc=NC9(watermark,markedim,m,n,row,column,trellis,k,intrlvrIndices,lengthOFturbo);

%% median filter
medianATTACK=medfilt2(markedIM);
ncMEDIAN=NC9(watermark,medianATTACK,m,n,row,column,trellis,k,intrlvrIndices);
mse=0;
for i=1:m*m
    mse=mse+medianATTACK(i)-coverimage(i);
end
mse=mse/(m*m);
psnrMEDIAN=10*log10(255^2/mse);

%% gaussian noise
gaussianATTACK=imnoise(markedIM,'gaussian');
ncGAUSSIAN=NC9(watermark,gaussianATTACK,m,n,row,column,trellis,k,intrlvrIndices);
mse=0;
for i=1:m*m
    mse=mse+gaussianATTACK(i)-coverimage(i);
end
mse=mse/(m*m);
psnrgaussian=abs(10*log10(255^2/mse));

%% salt & pepper 
saltpepperATTACK=imnoise(markedIM,'salt & pepper') ;
ncSALTPEPPER=NC9(watermark,saltpepperATTACK,m,n,row,column,trellis,k,intrlvrIndices);
mse=0;
for i=1:m*m
    mse=mse+saltpepperATTACK(i)-coverimage(i);
end
mse=mse/(m*m);
psnrsaltpepper=abs(10*log10(255^2/mse));

%% 攻击
dataimage={'fishingboatW_JPEG_15.bmp','fishingboatW_JPEG_20.bmp','fishingboatW_JPEG_25.bmp','fishingboatW_JPEG_30.bmp' ...
    ,'fishingboatW_JPEG_35.bmp','fishingboatW_JPEG_40.bmp','fishingboatW_JPEG_50.bmp','fishingboatW_JPEG_60.bmp'...
    ,'fishingboatW_JPEG_70.bmp','fishingboatW_JPEG_80.bmp','fishingboatW_JPEG_90.bmp','fishingboatW_JPEG_100.bmp'};
dataimage2={'fishingboatW_MEDIAN_3.bmp','fishingboatW_MEDIAN_5.bmp','fishingboatW_MEDIAN_7.bmp','fishingboatW_MEDIAN_9.bmp',...
    'fishingboatW_NOISE_0.bmp','fishingboatW_NOISE_20.bmp','fishingboatW_NOISE_40.bmp','fishingboatW_NOISE_60.bmp',...
    'fishingboatW_NOISE_80.bmp','fishingboatW_NOISE_100.bmp'};

[~,lengthofimage]=size(dataimage);
[~,lengthofimage2]=size(dataimage2);
for i=1:1:lengthofimage
    markedimage=double(imread(strrep(dataimage{i},'fishingboatW','z1x25WCONVI')));
    disp(strrep(dataimage{i},'fishingboatW','z1x25WCONVI'));
    nc=NC9(watermark,markedimage,m,n,row,column,trellis,k,intrlvrIndices,lengthOFturbo)
end
for i=1:1:lengthofimage2
    markedimage=double(imread(strrep(dataimage2{i},'fishingboatW','z1x25WCONVI')));
    disp(strrep(dataimage2{i},'fishingboatW','z1x25WCONVI'));
    nc=NC9(watermark,markedimage,m,n,row,column,trellis,k,intrlvrIndices,lengthOFturbo)
end

%% 打印结果
A1=["orignal","median filter","gaussian noise","salt & pepper";
    psnr,psnrMEDIAN,abs(psnrgaussian),abs(psnrsaltpepper);
    nc,ncMEDIAN,ncGAUSSIAN,ncSALTPEPPER];
formatSpec = '%17s:  PSNR  %9.4f   NC  %9.4f \n';
fprintf(formatSpec,A1);
