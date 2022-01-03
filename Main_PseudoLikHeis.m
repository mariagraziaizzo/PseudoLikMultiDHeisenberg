% This code finds the values of the elements of the interactions matrix J that maximize the Log-Pseudo-Likelihood function of a Boltzmann probability distribution function related to a generalized Heisenberg model with nd-dimensional spin variables defined on a graph with nNodes nodes and nT observations (lenght of the time series).
clear all
close all
%
global i D l m l2_r nd
%
output_name='Outputs Name '; %choose a name for the output files
nd=21; % dimension of the spin variables;
d=(1:21); % vector labeling the d-th component of the spin variables;
for j=1:nd;
datafiles=['Input Files Name_',num2str(d(j))]; % a single input file is available for each component of the spin variables. The input files corresponding to the d-th component of the spin variables are named 'Input Files Name_d', where 'd' is a number labeling the dimensions of the spin variables. The input files of each component contain a time series of graph's nodes configurations.;
data=importdata(datafiles);   
S(:,:,j) = data; % matrix containing the time series of spin variables. First index of the matrix: node's index. Second index of the matrix: time or configuration index. Third index of the matrix: component index of the spin variable.
end
%
nT = length(data(1,:)); % lenght of time series (number of realized configurations of spin variables available)
nNodes = length(data(:,1)); % number of nodes of the graph;
%
ci=[1];% array containing the initial values of the elements of the matrix J in the maximization algorithm. Do not put value ==0! Zero values of elements of J are used to disentagle fixed parameters and free parameters in the optimization routine.
l2=[0.13]; % array containing the values of the parameer of the regularizer of the Log-Pseudo_likelihood function one wants to try.
%
for r=1:length(l2);
    l2_r=l2(r);
for t=1:length(ci);
for i=1:nNodes; % the Log-Pseudo-Likelihood of each node is independently maximized
Jinit = zeros(size(S,1),size(S,1));
D=1-diag(diag(ones(size(S,1),size(S,1))));
Jinit(:,i)=Jinit(:,i)+ci(t); % all the elelements of J are initianalized to the same value ci
Jinit=Jinit.*D;
[l,m,jf]=find(Jinit);
Jinit_f=jf;
 options=optimoptions('fminunc','Algorithm','trust-region','Display','iter',...
     'GradObj','on','MaxFunEvals',100000,'TolX',1e-9,'TolFun',1e-5,'MaxIter',1000);
lb=Jinit_f.*0; %lower bound of J
ub=abs(Jinit_f.*inf); % upper bound of J
A = [];
b = [];
Aeq = [];
beq = [];
nonlcon = [];
objfun = @(J_f) obFun_multiDHeis(S,J_f); % it is defined the function to be minimized, objfun, given in obFun_multiDHeis.m.
[J_f,e,exitflag,output,grad] = fminunc(objfun,Jinit_f,options);% Minimization algorithm by fminunc [https://www.mathworks.com/help/optim/ug/fminunc.html]. The gradient is included. The upper and lower bounds of J are not considered here. 
%[J_f,e,exitflag,output,grad]=fmincon(objfun,Jinit_f,A,b,Aeq,beq,lb,ub,nonlcon,options); %here the upper and lower bounds of J are considered. Uncomment this line if you want to include upper and lower bounds of J.
J0=sparse(l,m,J_f,size(S,1),size(S,1));
J_t=full(J0);
J(:,i)=J_t(:,i);
end
J_ciR(:,:,t,r)=triu((J+J')./2); % the J matrix is made symmetric
%J_ci=J(:,:,t,r); % uncomment this line if you want to save the J matrix forall ci
%save (['J_',output_name,'ci=',num2str(ci(t)),'l2=',num2str(l2(r))], 'J_ci', '-ascii'); % uncomment this line if you want to save the J matrix forall ci
if t==1;
    s=t;
else
    s=t-1;
end
if  J_ciR(:,:,t,r)-J_ciR(:,:,s,r) ~= 0;  
    disp('DEPENDENCE FROM INITIAL CONDITIONS');
    figure1=figure;
    % Create axes
    axes1 = axes('Parent',figure1);
    imagesc(J_ciR(:,:,t,r));
    title(['J_',output_name,' ci=',num2str(ci(t)),' l2=',num2str(l2(r))]);
    % Create colorbar
    colorbar('peer',axes1);
    break %the script is stopped when a dependence from the initial condition of J is found
end
end
J_end=J_ciR(:,:,length(ci),r);
% save (['J_',output_name,'l2=',num2str(l2(r))], 'J_end', '-ascii'); % uncomment this line if you want to save the J matrix forall l2
figure1=figure;
axes1 = axes('Parent',figure1);
imagesc((J_end));
title(['J_',output_name,' l2=',num2str(l2(r))]);
% Create colorbar
colorbar('peer',axes1);
% saveas(figure,['J_',output_name,' l2=',num2str(l2(r)),'.fig']);
% saveas(figure,['J_',output_name,' l2=',num2str(l2(r)),'.jpg']);
%
figure1=figure;
% Create axes
axes1 = axes('Parent',figure1);
imagesc(sign(J_end));
title(['sign(J)_',output_name,' l2=',num2str(l2(r))]);
% Create colorbar
colorbar('peer',axes1);
% saveas(figure,['sign(J)_',output_name,' l2=',num2str(l2(r)),'.fig']);
% saveas(figure,['sign(J)_',output_name,' l2=',num2str(l2(r)),'.jpg']);
figure1=figure;
% Create axes
axes1 = axes('Parent',figure1);
imagesc(log10(abs(J_end)));
title(['log10(abs(J))_',output_name,' l2=',num2str(l2(r))]);
% Create colorbar
colorbar('peer',axes1);
% saveas(figure,['log10(abs(J))_',output_name,' l2=',num2str(l2(r)),'.fig']);
% saveas(figure,['log10(abs(J))_',output_name,' l2=',num2str(l2(r)),'.jpg']);
end