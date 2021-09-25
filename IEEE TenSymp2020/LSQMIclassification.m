function QMIh=LSQMIclassification(x,y,sigma_list,lambda_list,b)
%
% Least-Squares Quadratic Mutual Information for classification
%
% Estimating quadratic mutual information 
%    \int\sum_{y=1,...,c} (p_{xy(x,y)-p_x(x)p_y(y))^2 dx
% from input-output samples
%    { (x_i,y_i) | x_i\in R^d, y_i\in{1,...,c} }_{i=1}^n 
% drawn independently from a joint density p(x,y).
% p(x) and p(y) are marginal density/probability of x and y, respectively.
% 
% Usage:
%       QMIh=LSQMIclassification(x,y,sigma_list,lambda_list,b)
%
% Input:
%    x          : d by n input sample matrix
%    y          : 1 by n label vector
%    sigma_list : (candidates of) Gaussian width
%                 If sigma_list is a vector, one of them is selected by cross validation.
%                 If sigma_list is a scalar, this value is used without cross validation
%                 If sigma_list is empty/undefined, Gaussian width is chosen from
%                 some default canditate list by cross validation
%    lambda_list: (OPTIONAL) regularization parameter
%                 If lambda_list is a vector, one of them is selected by cross validation.
%                 If lambda_list is a scalar, this value is used without cross validation
%                 If lambda_list is empty, Gaussian width is chosen from
%                 some default canditate list by cross validation
%    b          : number of Gaussian centers (if empty, b=100 is used);
%
% Output:
%    QMIh        : estimated mutual information between x and y

 

if nargin<2
  error('number of input arguments is not enough!!!')
end
if nargin < 3 || isempty(sigma_list)
%   sigma_list=logspace(-2,2,9);
sigma_list=logspace(-2,2,9);
end
if nargin < 4 || isempty(lambda_list)
%   lambda_list=logspace(-3,1,9);
lambda_list=logspace(-3,1,9);
end

if nargin<5 || isempty(b)
  b = 200;
end

[d,n]=size(x);
y_list=unique(y);
b=min(b,n);
tmp=randperm(n);
center_flag=zeros(1,n);
center_flag(tmp(1:b))=1;
for y_index=1:length(y_list)
  center_y=x(:,center_flag & y==y_list(y_index));
  data(y_index).dist2=repmat(sum(x.^2,1),[size(center_y,2) 1]) ...
	    +repmat(sum(center_y.^2,1)',[1 n])-2*center_y'*x;
end
clear center_y

fold=5;
fold_index=[1:fold];
if length(sigma_list)==1 && length(lambda_list)==1
  sigma_chosen=sigma_list;
  lambda_chosen=lambda_list;
  %%%%%%%%%%%%%%%% Searching Gaussian kernel width `sigma_chosen'
  %%%%%%%%%%%%%%%% and regularization parameter `lambda_chosen' 
else
  cv_index=mod(randperm(n),fold)+1;
  n_cv=hist(cv_index,fold_index);
  for sigma_index=1:length(sigma_list)
    sigma2=sigma_list(sigma_index)^2;
    for y_index=1:length(y_list)
      Ky=exp(-data(y_index).dist2/(2*sigma2));
      cv_index_y=cv_index(y==y_list(y_index));
      Hy_sigma=(pi*sigma2)^(d/2)...
	    *exp(-data(y_index).dist2(:,center_flag & y==y_list(y_index))/(4*sigma2));
      clear hhy1_cv hhy2_cv
      for k=fold_index
        hhy1_cv(:,k)=sum(Ky(:,cv_index==k & y==y_list(y_index)),2);
        hhy2_cv(:,k)=sum(Ky(:,cv_index==k),2);
        ny_cv(k)=sum(cv_index==k & y==y_list(y_index));
      end
      for k=fold_index
        hhy_cv_tr=sum(hhy1_cv(:,fold_index~=k),2)/sum(n_cv(fold_index~=k)) ...
		  -sum(hhy2_cv(:,fold_index~=k),2) ...
		  *sum(ny_cv(fold_index~=k))/(sum(n_cv(fold_index~=k))^2);
	hhy_cv_te=hhy1_cv(:,k)/n_cv(k)-hhy2_cv(:,k)*ny_cv(k)/(n_cv(k)^2);
        for lambda_index=1:length(lambda_list)
          alphahy_cv=mylinsolve(Hy_sigma+lambda_list(lambda_index)*eye(size(Hy_sigma,1)),hhy_cv_tr);
          scores_cv(sigma_index,lambda_index,k,y_index)=alphahy_cv'*Hy_sigma*alphahy_cv-2*alphahy_cv'*hhy_cv_te;
        end % lambda
      end % k
    end % y_index
  end % sigma
  [scores_cv_tmp,lambda_chosen_index]=min(mean(sum(scores_cv,4),3),[],2);
  [score_cv,sigma_chosen_index]=min(scores_cv_tmp);
  lambda_chosen=lambda_list(lambda_chosen_index(sigma_chosen_index));
  sigma_chosen=sigma_list(sigma_chosen_index);

end %length(sigma_list)==1 && length(lambda_list)==1

%%%%%%%%%%%%%%%% Computing the final solution `QMIh'

for y_index=1:length(y_list)
  Ky=exp(-data(y_index).dist2/(2*sigma_chosen^2));
  Hy=(pi*sigma_chosen^2)^(d/2)...
	    *exp(-data(y_index).dist2(:,center_flag & y==y_list(y_index))/(4*sigma_chosen^2));
  hhy=sum(Ky(:,y==y_list(y_index)),2)/n-sum(Ky,2)*sum(y==y_list(y_index))/(n^2);
  alphahy=mylinsolve(Hy+lambda_chosen*eye(size(Hy,1)),hhy);
  %QMIh=QMIh+hhy'*alphahy-alphahy'*Hhy*alphahy/2;
  %QMIh=alphahy'*Hhy*alphahy/2-(mean(Phix,2).*mean(Phiy,2))'*alphahy+1/2;
  QMIhs(y_index)=2*alphahy'*hhy-alphahy'*Hy*alphahy;
end

QMIh=sum(QMIhs);
