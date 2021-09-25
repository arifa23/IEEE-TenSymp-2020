Image = imread('92AV3C.tif');%reads image
[r, c, b] = size(Image);%image transformed into 3D data
newRow = r*c; %compressed 
image = reshape(Image,newRow,b);%transforms 3D data into 2D
meanImage = mean(image);%calculates mean x
imageAd = bsxfun(@minus,double(image),meanImage); %calculates Adjusted data=(xi- xmean)
covMat = cov(imageAd); %generates covariance matrix
[eigVector,eigValue] = eig(covMat,'nobalance'); %generates eigen value & matrix
[eigValue,idx] = sort(diag(eigValue),'descend'); %sorts eigenvector in descending order
eigVector = eigVector(:,idx(1:1:end));
preFinal = eigVector'*imageAd';%  trans(eigenvector)* trans(Adjusted data)
finalData = preFinal';% feature vector = trans(prefinal)
imagePCA1 = bsxfun(@plus,meanImage,finalData);
imagePCA1=uint16(finalData);%transforms into unsigned 16 bit binary data
imagePCA=reshape(imagePCA1, r, c, b); %%transforms into 3D image 

for i = 1:20
    j=imagePCA(:,:,i);% first 20 pca image
 figure,colormap(gray),imagesc(j);
end

cumVar=(cumsum(eigValue))/sum(eigValue);%calculates Cumulative variance
pc=1:b;
figure,
plot( pc,cumVar)
grid on;
xlabel('Principal Components');
ylabel('Cumulative variance(%)');
title('cumulative variance explained');
multibandwrite(imagePCA, 'imagePCA.tif', 'bsq');
%reverse transformed data(all the process in opposite)
recovImage1 = eigVector*preFinal; 
recovImage2 = recovImage1';
recovImage3 = bsxfun(@plus,recovImage2,meanImage); 
recovImage4 = uint8(recovImage3);
recovImage5 = reshape(recovImage4,r,c,b);
