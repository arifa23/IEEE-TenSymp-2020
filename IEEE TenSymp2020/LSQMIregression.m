function QMIh=LSQMIregression(x,y,sigma_list,lambda_list,b)
%
% Least-Squares Quadratic Mutual Information (with least-squares cross validation)
%
% Estimating quadratic mutual information 
%    \int\int (p_{xy(x,y)-p_x(x)p_y(y))^2 dx dy
% from input-output samples
%    { (x_i,y_i) | x_i\in R^{dx}, y_i\in R^{dy} }_{i=1}^n 
% drawn independently from a joint density p_{xy}(x,y).
% p_x(x) and p_y(y) are marginal densities of x and y, respectively.
% 
% Usage:
%       [QMIh,score_cv]=LSQMIregression(x,y,sigma_list,lambda_list,b)
%
% Input:
%    x          : dx by n input sample matrix
%    y          : dy by n output sample matrix
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
%    b          : number of Gaussian centers (if empty, b=200 is used);
%
% Output:
%    QMIh       : estimated quadratic mutual information between x and y

if nargin<2
  error('number of input arguments is not enough!!!')
end

n =size(x,2);
if n~=size(y,2)
  error('x and y must have the same number of samples!!!')
end;
if nargin < 3 || isempty(sigma_list)
  sigma_list=logspace(-2,2,9);
end
if nargin < 4 || isempty(lambda_list)
  lambda_list=logspace(-3,1,9);
end
if nargin<5 || isempty(b)
  b = 200;
end

b=min(n,b);
dx=size(x,1);
dy=size(y,1);

%Gaussian centers are randomly chosen from samples
center_index=randperm(n);
center_index=center_index(1:b);
xxsum=sum(x.^2,1);
yysum=sum(y.^2,1);

dist2_x=repmat(xxsum,[b 1])+repmat(xxsum(1,center_index)',[1 n])-2*x(:,center_index)'*x;
dist2_y=repmat(yysum,[b 1])+repmat(yysum(1,center_index)',[1 n])-2*y(:,center_index)'*y;

clear xxsum yysum x y 

if length(sigma_list)==1 && length(lambda_list)==1
  sigma_chosen=sigma_list;
  lambda_chosen=lambda_list;
else
  %%%%%%%%%%%%%%%% Searching Gaussian kernel width `sigma_chosen'
  %%%%%%%%%%%%%%%% and regularization parameter `lambda_chosen' 
  fold=5;
  fold_index=[1:fold];
  cv_index=mod(randperm(n),fold)+1;
  n_cv=hist(cv_index,fold_index);

  for sigma_index=1:length(sigma_list)
    sigma2=sigma_list(sigma_index)^2;
    Phix_sigma=exp(-dist2_x/(2*sigma2));
    Phiy_sigma=exp(-dist2_y/(2*sigma2));
    for k=fold_index
      hh1_cv(:,k)=sum(Phix_sigma(:,cv_index==k).*Phiy_sigma(:,cv_index==k),2);
      hh2_cv(:,k)=sum(Phix_sigma(:,cv_index==k),2);
      hh3_cv(:,k)=sum(Phiy_sigma(:,cv_index==k),2);
    end
    H_sigma=(pi*sigma2)^((dx+dy)/2)...
	    *exp(-(dist2_x(:,center_index)+dist2_y(:,center_index))/(4*sigma2));
    for k=fold_index
      hh_cv_tr=sum(hh1_cv(:,fold_index~=k),2)/sum(n_cv(fold_index~=k)) ...
	       -sum(hh2_cv(:,fold_index~=k),2).*sum(hh3_cv(:,fold_index~=k),2)/(sum(n_cv(fold_index~=k))^2);
      hh_cv_te=hh1_cv(:,k)/n_cv(k)-hh2_cv(:,k).*hh3_cv(:,k)/(n_cv(k)^2);
      for lambda_index=1:length(lambda_list)
        alphah_cv=mylinsolve(H_sigma+lambda_list(lambda_index)*eye(b),hh_cv_tr);
        scores_cv(sigma_index,lambda_index,k)=alphah_cv'*H_sigma*alphah_cv-2*hh_cv_te'*alphah_cv;
      end % fold
    end % lambda
  end % sigma
  [scores_cv_tmp,lambda_chosen_index]=min(mean(scores_cv,3),[],2);
  [score_cv,sigma_chosen_index]=min(scores_cv_tmp);
  lambda_chosen=lambda_list(lambda_chosen_index(sigma_chosen_index));
  sigma_chosen=sigma_list(sigma_chosen_index);
end %length(sigma_list)==1 && length(lambda_list)==1

%%%%%%%%%%%%%%%% Computing the final solution `MIh'
Phix=exp(-dist2_x/(2*sigma_chosen^2));
Phiy=exp(-dist2_y/(2*sigma_chosen^2));
H=(pi*sigma_chosen^2)^((dx+dy)/2)...
  *exp(-(dist2_x(:,center_index)+dist2_y(:,center_index))/(4*sigma_chosen^2));
hh=mean(Phix.*Phiy,2)-mean(Phix,2).*mean(Phiy,2);
alphah=mylinsolve(H+lambda_chosen*eye(b),hh);

QMIh=2*hh'*alphah-alphah'*H*alphah;
