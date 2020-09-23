%%
clc;clear;close all;
%% 使用Otsu方法确定ROI
I=imread('ankle.tif');
coverimage=imadjust(I); 
level = graythresh(coverimage);
BWmask = imbinarize(coverimage,level);
newCOVimage=coverimage;
coverimage_Otsu = coverimage;
coverimage_Otsu(~BWmask) = 255;
figure;subplot(121),imagesc(coverimage_Otsu); colormap gray;
axis off; axis equal;title('E');
%% 使用SVM区分ROI和RONI %%%%%%%%%%%%%%%%%%%%%
%% try
coverimage = dicomread('brain2.dcm');
%coverimage=imadjust(I);
figure,imshow(coverimage,[]);
%imwrite(coverimage,'MYTRY1.tiff');
% C=dicomread('ankle_newseries.dcm');


TrainData_RONI = zeros(20,1,'double');
TrainData_ROI = zeros(20,1,'double');

% NROI鼠标点击采样二十个样本点
for i = 1:20
    [x,y] = ginput(1);
    hold on;
    plot(x,y,'r*');
    x = int16(x);
    y = int16(y);
    TrainData_RONI(i,1) = coverimage(x,y);
end
% ROI采样
msgbox("采集ROI");pause;
for i = 1:20
    [x,y] = ginput(1);
    hold on;
    plot(x,y,'ro');
    x = int16(x);
    y = int16(y);
    TrainData_ROI(i,1) = coverimage(x,y);
end
% 分类 NROI=0 ROI=1
TrainLabel = [zeros(length(TrainData_RONI),1); ...
    ones(length(TrainData_ROI),1)];

TrainData = [TrainData_RONI;TrainData_ROI];
% Best hyperparameter
svmModel=fitcsvm(TrainData,TrainLabel,'KernelFunction','rbf','OptimizeHyperparameters','auto',...
      'HyperparameterOptimizationOptions',struct('ShowPlots',true,'Repartition',true));

% 创建测试集
[row, column]=size(coverimage);
TestData = double(reshape(coverimage,row*column,1));

% 预测
TestLabel = predict(svmModel, TestData);
ind = reshape(TestLabel,[row column]);
ind = logical(ind);
coverimage_ROI = coverimage;
coverimage_ROI(~ind) = max(coverimage(:));

% level = graythresh(coverimage);
% BWmask = imbinarize(coverimage,level);
% 
% coverimage_Otsu = coverimage;
% coverimage_Otsu(~BWmask) = max(coverimage(:));
figure;
% subplot(221);imshow(coverimage,[]);
imshow(coverimage,[]);
% figure;
% imshow(coverimage_Otsu,[]);
figure;
imshow(coverimage_ROI,[]);

%% RSA加密水印
k=1000;
primeNUMBER=primes(k);
r1=0;r2=0;p=0;q=0;l=length(primeNUMBER);
while(r1==r2 )
    r1 = randi(l);
    r2 = randi(l);
    p=primeNUMBER(r1);
    q=primeNUMBER(r2);
end
n=p*q;n1=(p-1)*(q-1);
e=0;cd=0;val=0;
while(cd~=1||val==0)
    e=randi(n1);
    cd=gcd(e,n1);
    val=isprime(e);
end

%% 求d
b=1;
while(1)
    if mod(b*n1+1,e)==0 && b*n1+1~=e^2
        break;
    end
    b=b+1;
end
d=(b*n1+1)/e;

%% 水印
watermark=imread('230px-CUMT_logo.jpg');
watermark = rgb2gray(watermark);
watermark = imresize(watermark,[row column]);
figure;
imagesc(watermark); colormap gray;title('H');
axis off; axis equal;

watermarkRSA=zeros(row,column);

%% 加密
qm=dec2bin(e);len=length(qm);
for i=1:row
    for j=1:column
        m=double(watermark(i,j));
        %快速幂取模
        c=1;
        xz=1;
        while(xz<=len)  
            if(qm(xz)=='1')
                c=mod(mod((c^2),n)*m,n);
            elseif(qm(xz)=='0')
                c=(mod(c^2,n));
            end
            xz=xz+1;
        end
        watermarkRSA(i,j)=c;
    end
end
subplot(222),imagesc(watermarkRSA); colormap gray; title('I');
axis off; axis equal;

%% 块分解 4*4 嵌入pn
pn=randi(9,[row column]);
blockWise=zeros(4,4);sum=0;pntestNumber=1;imagewatermarked=double(coverimage);
for i=1:4:column-4
    for j=((i-1)*row+1):4:i*row-4
        tip=1;
        for k=1:16
            if ind(j+k-1-floor(k/4)*4+floor(k/4)*column)==0
                tip=0;
            end
            blockWise(k)=double(coverimage_ROI(j+k-1-floor(k/4)*4+floor(k/4)*column));
        end
        if tip==1
            LL= liftwavedec2(blockWise,4,2);
            sum = sum + 1;
            tissue=dec2bin(watermark(j));
            leng=length(tissue);
            if tissue(leng)=='0'
                LLoriged=LL(1)+pn(sum);
                pntestresults(pntestNumber)=pn(sum);
                pntestNumber=pntestNumber+1;
            else
                LLoriged=LL(1);
            end
            LL(1)=LLoriged;
            blockWiseED=liftwaverec2(LL,4,2);
            for k=1:16
                imagewatermarked(j+k-1-floor(k/4)*4+floor(k/4)*column)=blockWiseED(k);
            end
        end
    end
end
imagesc(imagewatermarked); colormap gray; 
axis off; axis equal;

%%
figure,imshow(imagewatermarked,[]);

dicomwrite(int16(imagewatermarked),'brainW.dcm');

gg=uint16(imagewatermarked);

iw =uint16(imagewatermarked);

imwrite(uint16(imagewatermarked),'brain11W.jpg','BitDepth',16,'Mode','lossless');
gg1=imread('brain2W.pgm');
%% 逆水印
% I1=imagewatermarked;
I1=gg1;
watermarkRErsa=zeros(row,column);
qm1=dec2bin(d);len1=length(qm1);
for i=1:row
    for j=1:column
        c=round(watermarkRSA(i,j)); %取整
        nm=1;
        xy=1;
        while(xy<=len1)    
            if(qm1(xy)=='1')
                nm=mod(mod((nm^2),n)*c,n);
            elseif(qm1(xy)=='0')
                nm=(mod(nm^2,n));
            end
            xy=xy+1;    
        end
        watermarkRErsa(i,j)=nm;
    end
end


isequal(watermark,watermarkRErsa)  

%%
blockWise=zeros(4,4);sum1=0;pntestNumber1=1;recoverimage=I1;
for i=1:4:column-4
    for j=((i-1)*row+1):4:i*row-4
        tip=1;
        for k=1:16
            if ind(j+k-1-floor(k/4)*4+floor(k/4)*column)==0
                tip=0;
            end
            blockWise(k)=I1(j+k-1-floor(k/4)*4+floor(k/4)*column);
        end
        if tip==1
            LL= liftwavedec2(blockWise,4,2);
            sum1 = sum1 + 1;
            tissue=dec2bin(watermarkRErsa(j));
            leng=length(tissue);
            if tissue(leng)=='0'
                LLorigRE=LL(1)-pn(sum1);
                pntestresults(pntestNumber1)=pn(sum1);
                pntestNumber1=pntestNumber1+1;
            else
                LLorigRE=LL(1);
            end
            LL(1)=LLorigRE;
            blockWiseRE=liftwaverec2(LL,4,2);
            for k=1:16
                recoverimage(j+k-1-floor(k/4)*4+floor(k/4)*column)=blockWiseRE(k);
            end
        end
    end
end

isequal(recoverimage,coverimage)
% subplot(235),imagesc(recoverimage); colormap gray; title('Recover Image');
% axis off; axis equal;

nocorrect=0;
for i=1:512*512
        if recoverimage(i)~=coverimage(i)
            nocorrect=nocorrect+1;
        end
end
%% psnr
psnr=psnr_t(imagewatermarked,coverimage,row,column);
nc=NC(gg1,coverimage,watermarkRErsa ,row,column,pntestresults,ind);

%% wiener
wienerATTACK=wiener2(imagewatermarked);
ncwiener=NC(wienerATTACK,coverimage,watermarkRErsa ,row,column,pntestresults,ind);
%% median filter
medianATTACK=medfilt2(imagewatermarked);
%psnrMEDIAN=psnr_t(medianATTACK,coverimage,row,column);
ncMEDIAN=NC(medianATTACK,coverimage,watermarkRErsa ,row,column,pntestresults,ind);
%% imboxfilt
imboxfiltATTACK=imboxfilt(imagewatermarked);
ncimboxfilt=NC(imboxfiltATTACK,coverimage,watermarkRErsa ,row,column,pntestresults,ind);
%% gaussian noise
gaussianATTACK=imnoise(imagewatermarked,'gaussian');
%psnrGAUSSIAN=psnr_t(gaussianATTACK,coverimage,row,column);
ncGAUSSIAN=NC(gaussianATTACK,coverimage,watermarkRErsa ,row,column,pntestresults,ind);

%% salt & pepper 
saltpepperATTACK=imnoise(imagewatermarked,'salt & pepper') ;
%psnrSALTPEPPER=psnr_t(saltpepperATTACK,coverimage,row,column);  
ncSALTPEPPER=NC(saltpepperATTACK,coverimage,watermarkRErsa ,row,column,pntestresults,ind);
%% imnoise(I,'speckle')
speckleATTACK=imnoise(imagewatermarked,'speckle');
ncspeckle=NC(speckleATTACK,coverimage,watermarkRErsa ,row,column,pntestresults,ind);
%% roifilt
h = fspecial('laplacian');
roifiltATTACK=roifilt2(h,imagewatermarked,ind);
ncroifilt=NC(roifiltATTACK,coverimage,watermarkRErsa ,row,column,pntestresults,ind);
%% 打印结果
A1=["orignal","median filter","gaussian noise","salt & pepper";
    psnr,psnrMEDIAN,psnrGAUSSIAN,psnrSALTPEPPER;
    nc,ncMEDIAN,ncGAUSSIAN,ncSALTPEPPER];
formatSpec = '%17s:  PSNR  %9.4f   NC  %9.4f \n';
fprintf(formatSpec,A1);
