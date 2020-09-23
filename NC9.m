function [NC] = NC9(w,markedIM,m,n,row,column,trellis,k,intrlvrIndices,lengthOFturbo)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
% [LLr,LHw,HLw,HHw] = haart2(markedIM,3);
% imgwave2 = liftwavedec2(markedIM,m,n);
% LL1=imgwave2(1:m/(n^2),1:m/(n^2));
% riverLL=zeros(column); %watermarkREVER = zeros(column/2 ,column);
% j=1;
% for i=1:2:row*column
%     x=LL1(i);y=LL1(i+1);  %    x=LL1(i);y=LL1(i+1);
%     L=floor((x+y)/2); %矩阵向下数
%     h1=floor((x-y)/2);
%     watermarkREVER(j)=2*h1-(x-y);
%     x1=L+floor((h1+1)/2);
%     y1=L-floor(h1/2);
%     riverLL(i)=x1;riverLL(i+1)=y1;
%     j=j+1;
% end
% % imgwave2(1:m/(n^2),1:m/(n^2))=riverLL;
% 
% watermarkREVER=reshape(watermarkREVER,row*column/2,1);
% watermarkREVER=logical(abs(watermarkREVER));

imgwave2 = liftwavedec2(markedIM,m,n);
LL1=imgwave2(1:m/(n^2),1:m/(n^2));
riverLL=zeros(column); %watermarkREVER = zeros(column/2 ,column);
j=1;
for i=1:2:row*column
    if j<=lengthOFturbo
        x=LL1(i);y=LL1(i+1);  %    x=LL1(i);y=LL1(i+1);
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
% imgwave2(1:m/(n^2),1:m/(n^2))=riverLL;
% isequal(riverLL,LL)
watermarkREVER=reshape(watermarkREVER,lengthOFturbo,1);
watermarkREVER=logical(abs(watermarkREVER));
% isequal(watermarkREVER,codedData)

%% 卷积码解码
tbdepth = 20; % A rate 1/2 code has a traceback depth of 5(ConstraintLength – 1).
decodedData1 = vitdec(watermarkREVER,trellis,tbdepth,'trunc','hard');
% decodedData1=reshape(decodedData,[row/4,column/2]);
% 
% watermarkFINAL=xor(decodedData,k);

% turboDE = comm.TurboDecoder('TrellisStructure',trellis,'InterleaverIndices',intrlvrIndices, ...
%     'NumIterations',4);
% release(turboDE);
% decodedData1 = step(turboDE,double(watermarkREVER));

% decodedData=reshape(decodedData,[row/2,column/2]);
% sum=0;
% for i=1:1024
%     if decodedData1(i)~=watermarkRESHAPE(i)
%         sum=sum+1;
%     end
% end
% 
% isequal(logical(decodedData1),watermarkRESHAPE)
decodedData1=reshape(decodedData1,[row/4,column/2]);
watermarkFINAL=xor(decodedData1,k);

% imgwave(1:256,1:256)=LL;
% isequal(imgwave2,imgwave)
% reverIM = liftwaverec2(imgwave2,m,n);



a=0;
b=0;
for i=1:row/4
    for j=1:column/2
        a=a+double(w(i,j))*watermarkFINAL(i,j);
        b=b+double(w(i,j))^2;
    end
end

NC=a/b;
 
end
