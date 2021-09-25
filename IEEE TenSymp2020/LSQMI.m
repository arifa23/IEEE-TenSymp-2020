 clear all;

%  rand('state',0);
%  randn('state',0);
% 
load FINAL_train.txt;
test = FINAL_train;
clear FINAL_train;
y=test(:,1);
test(:,1:2)=[];
[n,m] = size(test);

for i=1:220;
dataset=1;
switch dataset
 case 1 % dependent (regression)
  %n=200;
  X = test(:,i)';
  Y = y';  
  y_type=0; % regression
 case 2 % independent (regression)
  X = test(:,i)';
  Y = y';
 
  y_type=0; % regression
 case 3 % dependent (classification)
  X = test(:,i)';
  Y = y';
 
  y_type=1; % classification
 case 4 % independent (classification)
  X = test(:,i)';
  Y = y';
  y_type=1; % classification
end

 

if y_type==0
  QMIh(i)=LSQMIregression(X,Y);
else
  QMIh(i)=LSQMIclassification(X,Y);
end
end
disp(sprintf('(Estimated QMI between x and y) = %g\n',QMIh));
%disp(QMIh);

 tmpMI = QMIh; 
% [spam,idx]=sort(QMIh(:),'descend');
% QMI=idx(1:20)';
[QMIh,id] = sort(QMIh,'descend');
% S = id(1);
% Z=test(:,1)';
% Q=test(:,S(1));
% 
% flag = [];
% for i = 1:m
%     flag(i) = 0;
% end
% 
% flag(id(1)) = 1;
% 
% for feature = 2:m
%     mx = -2;
%     next = 0;
%     for i = 1:m
%         if flag(i) == 0 && tmpMI(i) > .5
%             redun = 0;
%             [tmp k] = size(S);
%           D=S(1);
%             for j = 1:k
%                redun = redun + LSQMIclassification(test(:,i)',test(:,S(j))');
%             end
%             G = LSQMIclassification(test(:,i)',y') - (1/k)*redun;
%             if G > mx
%                 mx = G;
%                 next = i;
%             end
%         end
%     end
% %     if mx < 1
% %         break;
% %     end
%     
%     S(feature) = next;
%     flag(next) = 1;
% end
% id = S;

%%%%%%%%%%%%%%%%%%%%%% Displaying original 2D data
% figure(dataset)
% clf
% hold on
% 
% set(gca,'FontName','Helvetica')
% set(gca,'FontSize',12)
% plot(X,Y,'ro','LineWidth',1,'MarkerSize',8);
% xlabel('x')
% ylabel('y')
% axis([0 20 -1.2 1.9])
% title(sprintf('(Estimated QMI between x and y) = %g',QMIh))
% 
% set(gcf,'PaperUnits','centimeters');
% set(gcf,'PaperPosition',[0 0 12 9]);
% %print('-depsc',sprintf('LSQMI%g',dataset))
% print('-dpng',sprintf('LSQMI%g',dataset))
%   
