function [NC] = NC(w,coverimage,watermarkRErsa,row,column,pn,ind)
%DERSA 此处显示有关此函数的摘要
%   此处显示详细说明

%%

blockWise=zeros(4,4);blockWise1=zeros(4,4);sum1=0;
for i=1:4:column-4
    for j=((i-1)*row+1):4:i*row-4
        tip=1;
        for k=1:16
            if ind(j+k-1-floor(k/4)*4+floor(k/4)*column)==0
               tip=0;
            end
            blockWise(k)=w(j+k-1-floor(k/4)*4+floor(k/4)*column);
            blockWise1(k)=coverimage(j+k-1-floor(k/4)*4+floor(k/4)*column);
        end
        if tip==1
            LL= liftwavedec2(blockWise,4,2);
            LL1= liftwavedec2(blockWise1,4,2);
            tissue=dec2bin(watermarkRErsa(j));
            leng=length(tissue);
            if tissue(leng)=='0'
%                 LLorigRE=LL(1)-pn(sum1);
                sum1 = sum1 + 1;
                pn1(sum1)=LL(1)-LL1(1);

%                 pntestresults(pntestNumber1)=pn(sum1);
%                 pntestNumber1=pntestNumber1+1;
%             else
%                 LLorigRE=LL(1);
            end
%             LL(1)=LLorigRE;
%             blockWiseRE=liftwaverec2(LL,4,2);
%             for k=1:16
%                 recoverimage(j+k-1-floor(k/4)*4+floor(k/4)*column)=blockWiseRE(k);
%             end
        end
    end
end



a=0;
b=0;
for i=1:sum1
        a=a+double(pn(i))*pn1(i);
        b=b+double(pn(i))^2;
end
% if a/b>1
%     NC=abs(b/a);
% else
    NC=a/b;
% end

